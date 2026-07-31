#!/usr/bin/env bash
set -euo pipefail
bash ./llm_engine/cells_merlin_segvol_b256/merlin_segvol_avgpack_b256__location_binary/submit_chain.sh
bash ./llm_engine/cells_merlin_segvol_b256/merlin_segvol_orca_b256__location_binary/submit_chain.sh
