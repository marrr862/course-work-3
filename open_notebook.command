#!/bin/bash
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo 'Project Python environment is missing. See START_HERE_HY.md.'
  read -r -p 'Press Enter to close.'
  exit 1
fi
.venv/bin/python -m ipykernel install --prefix .venv --name fake-news --display-name "Fake News (project)" || exit 1
exec .venv/bin/python -m jupyterlab --ip=127.0.0.1 --ServerApp.root_dir="$PWD" Final_Practical.ipynb
