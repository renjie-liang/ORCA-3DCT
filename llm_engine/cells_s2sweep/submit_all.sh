#!/usr/bin/env bash
set -euo pipefail
bash ./llm_engine/cells_s2sweep/colipri_avgpack_b27__location/submit_chain.sh
bash ./llm_engine/cells_s2sweep/colipri_avgpack_b27__size/submit_chain.sh
bash ./llm_engine/cells_s2sweep/colipri_avgpack_b27__density/submit_chain.sh
bash ./llm_engine/cells_s2sweep/colipri_avgpack_b27__texture/submit_chain.sh
bash ./llm_engine/cells_s2sweep/colipri_orca_b27__location/submit_chain.sh
bash ./llm_engine/cells_s2sweep/colipri_orca_b27__size/submit_chain.sh
bash ./llm_engine/cells_s2sweep/colipri_orca_b27__density/submit_chain.sh
bash ./llm_engine/cells_s2sweep/colipri_orca_b27__texture/submit_chain.sh
