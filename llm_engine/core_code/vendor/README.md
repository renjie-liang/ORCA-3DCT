# vendor/ — self-contained third-party code

Copied here on purpose. `eval_fast.py` used to reach OUT of this repo for `shared.metrics`:

    SHARED_ROOT = "<an external repo, now vendored here as core_code/vendor>"   # eval_fast.py:33

That repo was renamed to `AdaRAG-CT-RAW`, so the import silently pointed at nothing and report-gen eval could
not run. This is the same class of failure that let three weeks of VQA runs train on stale labels: a
cross-repo reference that nobody re-checks. Note `DTBD3D/Experiment/eval_panel/vendor/` already vendors this
exact package for exactly this reason — the vendored copies survived, the cross-repo references all rotted.

## shared/  (from <an external repo, now vendored here as core_code/vendor/shared>, 2026-07-16)

`eval_fast.py` needs three symbols; all are here and depend only on pip packages (numpy/torch/sklearn/
scipy/transformers) — no further cross-repo hops:

| symbol | file |
|---|---|
| `CLINICAL_FINDINGS` (the 18 CT-RATE findings) | `shared/metrics/medical/clinical_efficacy.py` |
| `load_clinical_labels_csv` | same |
| `compute_all_text_metrics` (BLEU/ROUGE-L/METEOR/CIDEr) | `shared/metrics/text/text_metrics.py` |

Also carries `RadBertClassifier` / `extract_findings` / `compute_clinical_f1`, i.e. the clinical-F1 path that
scores generated reports against the same 18 labels the disease family uses.

To refresh: re-copy from AdaRAG-CT-RAW and re-run report-gen eval. Do NOT replace this with an import from
another repo.
