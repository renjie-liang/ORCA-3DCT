#!/usr/bin/env bash
set -euo pipefail
D=./llm_engine/cells_s1sweep_b64/colipri_avgpack_b64__density
J1=$(bash "$D/s1.sh"); echo "s1=$J1"
J2=$(bash "$D/score.sh" "$J1"); echo "score=$J2"
echo "CHAIN colipri_avgpack_b64__density: $J1 -> $J2"
