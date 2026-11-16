#!/usr/bin/env bash
# Instalación en una VM Ubuntu 22.04 (Compute Engine e2-medium).
# Uso: sudo bash deploy/setup_vm.sh   (ejecutar desde la copia del repo en /opt/copiloto-opciones)
set -euo pipefail

DESTINO=/opt/copiloto-opciones
USUARIO=copiloto

echo "== Dependencias del sistema"
apt-get update -y
apt-get install -y python3.11 python3.11-venv python3-pip git

echo "== Usuario de servicio y carpeta"
id -u "$USUARIO" &>/dev/null || useradd --system --create-home --shell /usr/sbin/nologin "$USUARIO"
mkdir -p "$DESTINO"
if [ "$(pwd)" != "$DESTINO" ]; then
  rsync -a --exclude .venv --exclude data --exclude logs ./ "$DESTINO"/
fi
cd "$DESTINO"
mkdir -p data logs

echo "== Entorno virtual"
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "!! Edita $DESTINO/.env con tus claves antes de arrancar los servicios."
fi
chown -R "$USUARIO":"$USUARIO" "$DESTINO"

echo "== Servicios systemd"
cp deploy/copiloto-bot.service deploy/copiloto-app.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable copiloto-bot.service copiloto-app.service
systemctl restart copiloto-bot.service copiloto-app.service

echo "== Listo. Abre el puerto 8501 en el firewall de GCP:"
echo "   gcloud compute firewall-rules create copiloto-app --allow tcp:8501 --target-tags copiloto"
echo "   (y etiqueta la VM con 'copiloto'). Estado: systemctl status copiloto-bot copiloto-app"
