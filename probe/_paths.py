"""Label-root path constants, factored out so `probe_families/*.py` can import them without a circular
dependency on config (config imports the family REGISTRY, the families import only these paths)."""
CTRATE = "./data/CT-RATE/dataset/multi_abnormality_labels"
RESULTS = "./results"
DATA = "./data"
MERLIN = "./data/Merlin/probe_labels"
