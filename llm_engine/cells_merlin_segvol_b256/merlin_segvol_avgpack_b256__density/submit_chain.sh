#!/usr/bin/env bash
set -euo pipefail
D=./llm_engine/cells_merlin_segvol_b256/merlin_segvol_avgpack_b256__density
J1=$(bash "$D/s1.sh"); echo "s1=$J1"
J2=$(bash "$D/score.sh" "$J1"); echo "score=$J2"
echo "CHAIN merlin_segvol_avgpack_b256__density: $J1 -> $J2"
