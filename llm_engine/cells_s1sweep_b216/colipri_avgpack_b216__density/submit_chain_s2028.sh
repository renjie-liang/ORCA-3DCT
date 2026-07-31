#!/usr/bin/env bash
set -euo pipefail
D=./llm_engine/cells_s1sweep_b216/colipri_avgpack_b216__density
J1=$(bash "$D/s1_seed2028.sh"); echo "s1=$J1"
J2=$(bash "$D/score_s2028.sh" "$J1"); echo "score=$J2"
echo "CHAIN colipri_avgpack_b216__density seed=2028: $J1 -> $J2"
