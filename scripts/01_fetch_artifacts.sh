#!/bin/bash
# Download the pre-built training pairs (render side), trained weights and eval outputs from the shared Drive folder.
# usage: scripts/01_fetch_artifacts.sh <drive-folder-url-or-id>   (see ARTIFACTS.md for the layout)
set -e; cd "$(dirname "$0")/.."
mkdir -p artifacts/render2real
if ! ls artifacts/render2real/*.tar >/dev/null 2>&1; then
  # the Drive folder is access-restricted: sign in to Google in a browser and download the .tar files into artifacts/render2real/ yourself,
  # or, if the folder owner enabled link sharing, this works unattended:
  pip install -q gdown==5.2.0 && gdown --folder "$1" -O artifacts
fi
for t in artifacts/render2real/*.tar; do [ -f "$t" ] && tar -xf "$t"; done   # unpacks into ./data ./fast/weights ./wan/lora ...
echo "next: real.mp4 targets are NOT redistributed -> scripts/02_data.sh --targets-only (needs Seamless Interaction, DATA.md)"
