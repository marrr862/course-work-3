#!/bin/bash
# Telegram bot that restarts itself if it stops, and keeps the Mac from idle-sleeping.
# Opened in Terminal at login by tools/install_bot_service.sh (Terminal already has access to this folder;
# a background launchd job does not, because macOS protects the Desktop folder).
cd "$(dirname "$0")/.." || exit 1
while true; do
  /usr/bin/caffeinate -i .venv/bin/python -u -m fakenews.telegram_bot
  echo "bot stopped ($(date '+%H:%M')), restarting in 15 s"; sleep 15
done
