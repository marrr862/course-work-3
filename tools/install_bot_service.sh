#!/bin/bash
# Start the Telegram bot automatically when you log in to the Mac.
# A login agent opens tools/bot_forever.command in Terminal; that script restarts the bot if it
# stops and prevents idle sleep while it runs (closing the lid still sleeps the Mac).
# Why Terminal: macOS does not let background launchd jobs read folders on the Desktop.
# Usage:  bash tools/install_bot_service.sh          install and start now
#         bash tools/install_bot_service.sh remove   uninstall (close the bot's Terminal window to stop it)
set -e
cd "$(dirname "$0")/.."
LABEL=am.coursework.fakenews-bot
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if [ "$1" = remove ]; then rm -f "$PLIST"; echo "removed"; exit 0; fi
chmod +x tools/bot_forever.command
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/usr/bin/open</string><string>-a</string><string>Terminal</string><string>$PWD/tools/bot_forever.command</string></array>
  <key>RunAtLoad</key><true/>
</dict></plist>
PL
pkill -f "fakenews.telegram_bot" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed: $PLIST (the bot opens in a Terminal window at login)"
