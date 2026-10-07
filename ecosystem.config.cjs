const path = require("path");

module.exports = {
  apps: [
    {
      name: "API-ImageEnhance",
      script: "api.py",
      // pythonw + windowsHide: sin ventana de consola en Windows (PM2 reinicia si matas el proceso).
      interpreter: path.join(__dirname, ".venv", "Scripts", "pythonw.exe"),
      cwd: __dirname,
      windowsHide: true,
      env: {
        OSEDIFF_HOST: "0.0.0.0",
        OSEDIFF_PORT: "8010",
        // Segundos sin inferencias antes de liberar VRAM; 0 = al terminar cada tarea.
        OSEDIFF_IDLE_UNLOAD_SEC: "60",
      },
      autorestart: true,
      max_restarts: 10,
      min_uptime: "30s",
      time: true,
    },
  ],
};
