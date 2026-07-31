#!/usr/bin/env bash
set -euo pipefail
bash ./llm_engine/cells_s1sweep_b8/colipri_avgpack_b8__location/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b8/colipri_avgpack_b8__size/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b8/colipri_avgpack_b8__density/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b8/colipri_avgpack_b8__radiomics/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b8/colipri_orcabase_b8__location/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b8/colipri_orcabase_b8__size/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b8/colipri_orcabase_b8__density/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b8/colipri_orcabase_b8__radiomics/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b8/colipri_orcasin_b8__location/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b8/colipri_orcasin_b8__size/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b8/colipri_orcasin_b8__density/submit_chain.sh
bash ./llm_engine/cells_s1sweep_b8/colipri_orcasin_b8__radiomics/submit_chain.sh
