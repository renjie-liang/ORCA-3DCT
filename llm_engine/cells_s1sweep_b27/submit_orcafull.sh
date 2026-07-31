#!/usr/bin/env bash
set -euo pipefail
bash ./llm_engine/cells_s1sweep_b27/colipri_orcafull_b27__location/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b27/colipri_orcafull_b27__size/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b27/colipri_orcafull_b27__density/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b27/colipri_orcafull_b27__radiomics/submit_chain.sh
