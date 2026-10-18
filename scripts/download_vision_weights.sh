#!/usr/bin/env bash
# Downloads the GroundingDINO + SAM2.1 files that src/vision/object_localizer.py loads from weights/
# (Git-ignored). Run once on the vision machine, from the repository root:
#
#   scripts/download_vision_weights.sh          # GroundingDINO SwinT + SAM2.1 small (~1 GB)
#   scripts/download_vision_weights.sh large    # also SAM2.1 large (demo.vision localizer.sam2_size: large)
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p weights

fetch() {  # url dest
  if [[ -s "$2" ]]; then
    echo "have  $2"
  else
    echo "fetch $2"
    curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
  fi
}

fetch https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py \
      weights/GroundingDINO_SwinT_OGC.py
fetch https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
      weights/groundingdino_swint_ogc.pth
fetch https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
      weights/sam2.1_hiera_small.pt
if [[ "${1:-}" == "large" ]]; then
  fetch https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
        weights/sam2.1_hiera_large.pt
fi
ls -lh weights/
