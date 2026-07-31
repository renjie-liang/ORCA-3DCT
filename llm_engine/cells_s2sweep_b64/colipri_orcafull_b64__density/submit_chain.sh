#!/usr/bin/env bash
set -euo pipefail
D=./llm_engine/cells_s2sweep_b64/colipri_orcafull_b64__density
J1=$(bash "$D/s2.sh"); echo "s2=$J1"
J2=$(bash "$D/score.sh" "$J1"); echo "score=$J2"
echo "CHAIN colipri_orcafull_b64__density: $J1 -> $J2"
