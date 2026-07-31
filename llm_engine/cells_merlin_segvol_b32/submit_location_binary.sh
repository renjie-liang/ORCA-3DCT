#!/usr/bin/env bash
set -euo pipefail
bash ./llm_engine/cells_merlin_segvol_b32/merlin_segvol_avgpack_b32__location_binary/submit_chain.sh
bash ./llm_engine/cells_merlin_segvol_b32/merlin_segvol_orca_b32__location_binary/submit_chain.sh
