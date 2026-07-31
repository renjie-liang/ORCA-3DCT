#!/usr/bin/env bash
set -euo pipefail
bash ./llm_engine/cells_s2sweep_b216/colipri_dins_b216__location/submit_chain.sh
bash ./llm_engine/cells_s2sweep_b216/colipri_dins_b216__size/submit_chain.sh
bash ./llm_engine/cells_s2sweep_b216/colipri_dins_b216__density/submit_chain.sh
bash ./llm_engine/cells_s2sweep_b216/colipri_dins_b216__radiomics/submit_chain.sh
