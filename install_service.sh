#!/usr/bin/env bash
# Optional systemd user-service installer for Nukumi Mobile PWA (src/web_app.py).
# Usage:
#   ./install_service.sh             # Install, enable, and start background auto-restart service
#   ./install_service.sh --uninstall # Stop, disable, and remove the background service

set -euo pipefail

SERVICE_NAME="nukumi.service"
USER_SYSTEMD_DIR="${HOME}/.config/systemd/user"
SERVICE_FILE="${USER_SYSTEMD_DIR}/${SERVICE_NAME}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
WEB_APP_SCRIPT="${PROJECT_DIR}/src/web_app.py"

if [[ "${1:-}" == "--uninstall" || "${1:-}" == "uninstall" ]]; then
    echo "Stopping and removing ${SERVICE_NAME}..."
    systemctl --user stop "${SERVICE_NAME}" 2>/dev/null || true
    systemctl --user disable "${SERVICE_NAME}" 2>/dev/null || true
    rm -f "${SERVICE_FILE}"
    systemctl --user daemon-reload
    echo "✅ ${SERVICE_NAME} uninstalled. You can still run 'python src/web_app.py' manually anytime."
    exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: Virtual environment Python not found at ${PYTHON_BIN}."
    echo "Please create the virtual environment first: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

mkdir -p "${USER_SYSTEMD_DIR}"

cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Nukumi Field Support Mobile PWA Server
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PYTHON_BIN} -u ${WEB_APP_SCRIPT}
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "${SERVICE_NAME}"

echo "================================================================="
echo "✅ Nukumi background service installed and running!"
echo "================================================================="
echo "  Service file : ${SERVICE_FILE}"
echo "  Check status : systemctl --user status nukumi"
echo "  View live log: journalctl --user -u nukumi -f"
echo "  Stop service : systemctl --user stop nukumi"
echo "  Start service: systemctl --user start nukumi"
echo "  Uninstall    : ./install_service.sh --uninstall"
echo "-----------------------------------------------------------------"
echo "  Tip: To keep your user service running after reboot even before"
echo "  you log into the desktop, run once: sudo loginctl enable-linger \$USER"
echo "================================================================="
