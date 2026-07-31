#!/usr/bin/env bash
set -euo pipefail
bash ./llm_engine/cells_s1sweep_b64/colipri_avgpack_b64__location/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b64/colipri_avgpack_b64__size/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b64/colipri_avgpack_b64__density/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b64/colipri_avgpack_b64__radiomics/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b64/colipri_orcabase_b64__location/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b64/colipri_orcabase_b64__size/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b64/colipri_orcabase_b64__density/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b64/colipri_orcabase_b64__radiomics/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b64/colipri_orcasin_b64__location/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b64/colipri_orcasin_b64__size/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b64/colipri_orcasin_b64__density/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b64/colipri_orcasin_b64__radiomics/submit_chain.sh
