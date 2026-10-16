#!/usr/bin/env bash
set -euo pipefail
# Clamp the vision tower's input resolution for a Flash-Next deployment.
#
# Usage: image-resolution-cap.sh <apply|restore|status> [TARGET_MODEL]
#
# Why this exists: the checkpoint's preprocessor_config.json sets
# size.longest_edge = 16777216 (16.7 MP), which the Qwen3-VL image path feeds
# straight into the processor as max_pixels. A 24.5 MP source image therefore
# still resizes to ~16k image tokens, and the vision tower's activation buffers
# (the F.pad in apply_fp4_marlin_linear pads N 4304 -> 4352 per MLP) then ask for
# more headroom than --mem-fraction-static 0.981 leaves. The failure is a CUDA
# OOM that kills the scheduler, not a graceful error.
#
# 4 MP (longest_edge = 4194304) sizes the largest image at ~4.0k image tokens,
# which the qualified 524288-token KV pool plus that mem fraction already carry.
# It bounds vision memory, not correctness: measured descriptions of a
# 5712x4284 photograph are identical at 4 MP and at the resolutions below it.
#
# The cap is written into TARGET_MODEL itself, never into the prepared PLE
# overlay: check_ple_nvme.py requires every ordinary overlay asset to be a
# symlink resolving to its source, so an edited copy under /prepared fails the
# preflight ("Prepared source overlay asset does not resolve to source"). Editing
# the source keeps the symlink valid; the preflight hash-pins only config.json
# and model.safetensors.index.json, so preprocessor_config.json may change.
#
# A re-flattened TARGET_MODEL loses the cap; rerun `apply` after rebuilding it.
CAP_LONGEST_EDGE="${CAP_LONGEST_EDGE:-4194304}"
ACTION="${1:?Usage: image-resolution-cap.sh <apply|restore|status> [TARGET_MODEL]}"
TARGET_MODEL="${2:-${TARGET_MODEL:-}}"
if [[ -z "$TARGET_MODEL" ]]; then
  echo "Set TARGET_MODEL or pass it as the second argument" >&2
  exit 1
fi
CONFIG="$TARGET_MODEL/preprocessor_config.json"
if [[ ! -f "$CONFIG" ]]; then
  echo "No preprocessor config at $CONFIG" >&2
  exit 1
fi
if [[ ! "$CAP_LONGEST_EDGE" =~ ^[1-9][0-9]*$ ]]; then
  echo "CAP_LONGEST_EDGE must be a positive integer number of pixels" >&2
  exit 1
fi
read -r current shortest <<<"$(python3 - "$CONFIG" <<'EOF'
import json, sys
size = json.load(open(sys.argv[1]))["size"]
print(size["longest_edge"], size["shortest_edge"])
EOF
)"
case "$ACTION" in
  status)
    echo "longest_edge=$current shortest_edge=$shortest"
    if [[ "$current" == "$CAP_LONGEST_EDGE" ]]; then echo "cap: applied"; else echo "cap: not applied"; fi
    ;;
  apply|restore)
    target="$CAP_LONGEST_EDGE"
    [[ "$ACTION" == restore ]] && target=16777216
    python3 - "$CONFIG" "$target" <<'EOF'
import json, sys
path, longest_edge = sys.argv[1], int(sys.argv[2])
config = json.load(open(path))
config["size"]["longest_edge"] = longest_edge
with open(path, "w") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
EOF
    echo "$CONFIG: longest_edge $current -> $target"
    ;;
  *)
    echo "Unknown action '$ACTION'; expected apply, restore or status" >&2
    exit 1
    ;;
esac
