"""Shared OSEDiff inference (CLI and API)."""
from __future__ import annotations

import io
import os
import sys
import threading
import time
import types
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.getcwd())

from osediff import OSEDiff_test
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from ram import inference_ram as inference
from ram.models.ram_lora import ram

tensor_transforms = transforms.Compose([transforms.ToTensor()])
ram_transforms = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

AlignMethod = Literal["adain", "wavelet"]
@dataclass
class EnhanceOptions:
    upscale: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    process_size: int = 512
    align_method: Optional[AlignMethod] = None
    prompt: str = ""


@dataclass
class EnhanceResult:
    image: Image.Image
    prompt: str
    width: int
    height: int
    inference_seconds: float


def _default_paths() -> dict[str, str]:
    root = os.getcwd()
    return {
        "sd": os.path.join(root, "preset/models/stable-diffusion-2-1-base"),
        "osediff": os.path.join(root, "preset/models/osediff.pkl"),
        "ram": os.path.join(root, "preset/models/ram_swin_large_14m.pth"),
        "dape": os.path.join(root, "preset/models/DAPE.pth"),
    }


def _build_osediff_args(osediff_path: str, paths: dict[str, str]) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        pretrained_model_name_or_path=paths["sd"],
        osediff_path=osediff_path,
        ram_path=paths["ram"],
        ram_ft_path=paths["dape"],
        mixed_precision="fp16",
        merge_and_unload_lora=False,
        vae_decoder_tiled_size=224,
        vae_encoder_tiled_size=1024,
        latent_tiled_size=96,
        latent_tiled_overlap=32,
        lora_rank=4,
    )


def _align_dimension(value: int) -> int:
    return max(8, value - value % 8)


def prepare_input_image(image: Image.Image, options: EnhanceOptions) -> tuple[Image.Image, bool]:
    """Resize input for OSEDiff; returns (prepared RGB image, legacy resize_flag)."""
    input_image = image.convert("RGB")
    ori_width, ori_height = input_image.size
    resize_flag = False

    has_w = options.width is not None and options.width > 0
    has_h = options.height is not None and options.height > 0

    if has_w or has_h:
        if has_w and has_h:
            target_w = _align_dimension(options.width)
            target_h = _align_dimension(options.height)
        elif has_w:
            scale = options.width / ori_width
            target_w = _align_dimension(options.width)
            target_h = _align_dimension(int(round(ori_height * scale)))
        else:
            scale = options.height / ori_height
            target_h = _align_dimension(options.height)
            target_w = _align_dimension(int(round(ori_width * scale)))
        input_image = input_image.resize((target_w, target_h), Image.LANCZOS)
    else:
        if options.upscale is None:
            raise ValueError("Indica width/height o upscale")
        rscale = options.upscale
        if ori_width < options.process_size // rscale or ori_height < options.process_size // rscale:
            scale = (options.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize(
                (int(scale * ori_width), int(scale * ori_height)), Image.LANCZOS
            )
            resize_flag = True
        input_image = input_image.resize(
            (input_image.size[0] * rscale, input_image.size[1] * rscale), Image.LANCZOS
        )
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

    return input_image, resize_flag


# RAM tags that push SD toward heavy makeup / stylized faces
_RAM_COSMETIC_TAGS = frozenset(
    {
        "lipstick",
        "makeup",
        "makeup artist",
        "eyeshadow",
        "mascara",
        "eyeliner",
        "blush",
        "lip gloss",
        "lipgloss",
        "foundation",
        "concealer",
        "rouge",
        "glamour",
        "glamorous",
        "beauty salon",
    }
)


def _filter_ram_tags(tag_string: str) -> str:
    kept: list[str] = []
    for part in tag_string.split(","):
        tag = part.strip()
        if not tag:
            continue
        lower = tag.lower()
        if lower in _RAM_COSMETIC_TAGS:
            continue
        if any(word in lower for word in ("makeup", "lipstick", "mascara", "eyeshadow")):
            continue
        kept.append(tag)
    return ", ".join(kept)


def _validation_prompt(
    input_image: Image.Image, dape: torch.nn.Module, prompt_extra: str, weight_dtype: torch.dtype
) -> tuple[str, torch.Tensor]:
    lq = tensor_transforms(input_image).unsqueeze(0).to("cuda")
    lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
    captions = inference(lq_ram, dape)
    ram_tags = _filter_ram_tags(captions[0])
    extra = prompt_extra.strip()
    if extra:
        validation_prompt = f"{ram_tags}, {extra},"
    else:
        validation_prompt = f"{ram_tags}, natural skin, realistic photo,"
    return validation_prompt, lq


class OSEDiffService:
    """Carga modelos bajo demanda; libera VRAM tras inactividad."""

    def __init__(self, paths: Optional[dict[str, str]] = None) -> None:
        self.paths = paths or _default_paths()
        self.weight_dtype = torch.float16
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._loading = False
        self._idle_timer: Optional[threading.Timer] = None
        self._dape: Optional[torch.nn.Module] = None
        self._osediff: Optional[OSEDiff_test] = None

    @property
    def ready(self) -> bool:
        return self._dape is not None and self._osediff is not None

    @property
    def loading(self) -> bool:
        return self._loading

    def _idle_unload_seconds(self) -> int:
        raw = os.getenv("OSEDIFF_IDLE_UNLOAD_SEC", "60")
        try:
            return int(raw)
        except ValueError:
            return 60

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _unload_models(self) -> None:
        self._cancel_idle_timer()
        if self._dape is None and self._osediff is None:
            return
        print("Unloading models (VRAM)...")
        self._dape = None
        self._osediff = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _try_unload_idle(self) -> None:
        if not self._lock.acquire(blocking=False):
            self._schedule_idle_unload()
            return
        try:
            self._unload_models()
        finally:
            self._lock.release()

    def _schedule_idle_unload(self) -> None:
        self._cancel_idle_timer()
        sec = self._idle_unload_seconds()
        if sec < 0:
            return
        if sec == 0:
            threading.Thread(target=self._try_unload_idle, daemon=True).start()
            return
        self._idle_timer = threading.Timer(sec, self._try_unload_idle)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def load(self) -> None:
        with self._load_lock:
            if self._dape is not None:
                self._cancel_idle_timer()
                return
            self._loading = True
            try:
                print("Loading RAM/DAPE...")
                self._dape = ram(
                    pretrained=self.paths["ram"],
                    pretrained_condition=self.paths["dape"],
                    image_size=384,
                    vit="swin_l",
                )
                self._dape.eval()
                self._dape.to("cuda", dtype=self.weight_dtype)

                print("Loading OSEDiff...")
                args = _build_osediff_args(self.paths["osediff"], self.paths)
                self._osediff = OSEDiff_test(args)
            finally:
                self._loading = False

    def shutdown(self) -> None:
        with self._load_lock:
            with self._lock:
                self._unload_models()

    def enhance(self, image: Image.Image, options: EnhanceOptions) -> EnhanceResult:
        self.load()

        input_image, resize_flag = prepare_input_image(image, options)
        t0 = time.perf_counter()

        with self._lock:
            dape = self._dape
            assert dape is not None
            model = self._osediff
            assert model is not None
            validation_prompt, lq = _validation_prompt(
                input_image, dape, options.prompt, self.weight_dtype
            )

            with torch.no_grad():
                lq = lq * 2 - 1
                output_image = model(lq, prompt=validation_prompt)
                output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
                if options.align_method == "adain":
                    output_pil = adain_color_fix(target=output_pil, source=input_image)
                elif options.align_method == "wavelet":
                    output_pil = wavelet_color_fix(target=output_pil, source=input_image)
                if resize_flag:
                    ow, oh = image.convert("RGB").size
                    output_pil = output_pil.resize(
                        (int(options.upscale * ow), int(options.upscale * oh)), Image.LANCZOS
                    )

        self._schedule_idle_unload()

        elapsed = time.perf_counter() - t0
        return EnhanceResult(
            image=output_pil,
            prompt=validation_prompt,
            width=output_pil.width,
            height=output_pil.height,
            inference_seconds=elapsed,
        )


def result_to_png_bytes(result: EnhanceResult) -> bytes:
    buf = io.BytesIO()
    result.image.save(buf, format="PNG")
    return buf.getvalue()
