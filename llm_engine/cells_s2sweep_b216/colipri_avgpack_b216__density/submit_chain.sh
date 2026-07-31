#!/usr/bin/env bash
set -euo pipefail
D=./llm_engine/cells_s2sweep_b216/colipri_avgpack_b216__density
J1=$(bash "$D/s2.sh"); echo "s2=$J1"
J2=$(bash "$D/score.sh" "$J1"); echo "score=$J2"
echo "CHAIN colipri_avgpack_b216__density: $J1 -> $J2"
