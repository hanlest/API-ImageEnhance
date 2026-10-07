import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import time
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

from osediff import OSEDiff_test
from osediff_inference import EnhanceOptions, prepare_input_image
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
    captions = inference(lq_ram, model)
    validation_prompt = f"{captions[0]}, {args.prompt},"
    
    return validation_prompt, lq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='preset/datasets/test_dataset/input', help='path to the input image')
    parser.add_argument('--output_dir', '-o', type=str, default='preset/datasets/test_dataset/output', help='the directory to save the output')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='sd model path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Ancho objetivo (px). Con --height define tamaño exacto; solo width = proporcional.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Altura objetivo (px). Con --width define tamaño exacto; solo height = proporcional.",
    )
    parser.add_argument(
        "--align_method",
        type=str,
        choices=["adain", "wavelet"],
        default=None,
        help="Corrección de color: adain o wavelet. Omitir = sin corrección.",
    )
    parser.add_argument("--osediff_path", type=str, default='preset/models/osediff.pkl')
    parser.add_argument('--prompt', type=str, default='', help='user prompts')
    parser.add_argument('--ram_path', type=str, default=None)
    parser.add_argument('--ram_ft_path', type=str, default=None)
    parser.add_argument('--save_prompts', type=bool, default=True)
    # precision setting
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    # merge lora
    parser.add_argument("--merge_and_unload_lora", default=False) # merge lora weights before inference
    # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) 
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) 
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 

    args = parser.parse_args()

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    # Load RAM/DAPE before OSEDiff to avoid peak CPU RAM when loading large checkpoints
    DAPE = ram(pretrained=args.ram_path,
            pretrained_condition=args.ram_ft_path,
            image_size=384,
            vit='swin_l')
    DAPE.eval()
    DAPE.to("cuda", dtype=weight_dtype)

    model = OSEDiff_test(args)

    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
    else:
        image_names = [args.input_image]
    
    if args.save_prompts:
        txt_path = os.path.join(args.output_dir, 'txt')
        os.makedirs(txt_path, exist_ok=True)
    
    # make the output dir
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')
    for image_name in image_names:
        # make sure that the input image is a multiple of 8
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        enhance_opts = EnhanceOptions(
            upscale=args.upscale,
            width=args.width,
            height=args.height,
            process_size=args.process_size,
            align_method=args.align_method,
        )
        input_image, resize_flag = prepare_input_image(input_image, enhance_opts)
        if args.width or args.height:
            print(f"target size -> {input_image.width}x{input_image.height}")
        bname = os.path.basename(image_name)

        # get caption
        validation_prompt, lq = get_validation_prompt(args, input_image, DAPE)
        if args.save_prompts:
            txt_save_path = f"{txt_path}/{bname.split('.')[0]}.txt"
            with open(txt_save_path, 'w', encoding='utf-8') as f:
                f.write(validation_prompt)
                f.close()
        print(f"process {image_name}, tag: {validation_prompt}".encode('utf-8'))

        # translate the image
        t0 = time.perf_counter()
        with torch.no_grad():
            lq = lq*2-1
            output_image = model(lq, prompt=validation_prompt)
            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
            if enhance_opts.align_method == "adain":
                output_pil = adain_color_fix(target=output_pil, source=input_image)
            elif enhance_opts.align_method == "wavelet":
                output_pil = wavelet_color_fix(target=output_pil, source=input_image)
            if resize_flag:
                output_pil.resize((int(args.upscale*ori_width), int(args.upscale*ori_height)))

        out_path = os.path.join(args.output_dir, bname)
        output_pil.save(out_path)
        elapsed = time.perf_counter() - t0
        print(f"inference+save: {elapsed:.2f}s -> {out_path} ({output_pil.width}x{output_pil.height})")

