#!/usr/bin/env bash
set -euo pipefail
bash ./llm_engine/cells_s2sweep_b64/colipri_orcafull_b64__location/submit_chain.sh
bash ./llm_engine/cells_s2sweep_b64/colipri_orcafull_b64__size/submit_chain.sh
bash ./llm_engine/cells_s2sweep_b64/colipri_orcafull_b64__density/submit_chain.sh
bash ./llm_engine/cells_s2sweep_b64/colipri_orcafull_b64__radiomics/submit_chain.sh
