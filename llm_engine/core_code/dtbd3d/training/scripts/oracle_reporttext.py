"""Oracle report-text injection (Paper-2 Part-2: LLM-decode upper bound). ISOLATED helper.

Concatenates a per-volume report-text embedding (128-d TF-IDF/SVD) broadcast onto EVERY visual token
channel, making the synthetic embedding PROVABLY contain the report info. We then train report-gen on
it to measure how much the LLM can DECODE back (an ORACLE/LEAKAGE upper bound, NOT deployable).

On-the-fly (NOT precomputed): the 128-d vector is the SAME for all tokens of a volume, so precomputing
the full broadcast grid would be hundreds of GB and redundant; we store only the 128-d/volume npz
(configs/report_embed/report_tfidf_svd128.npz, ~6MB) and concat in __getitem__."""
import numpy as np

ORACLE_DIM = 128


def load_oracle_emb(npz_path: str) -> dict[str, np.ndarray]:
    z = np.load(npz_path)
    return {str(v): e.astype(np.float32) for v, e in zip(z["vids"], z["emb"])}


def _strip(vid: str) -> str:
    return vid[:-7] if vid.endswith(".nii.gz") else vid


def emb_dim(emb: dict[str, np.ndarray]) -> int:
    return int(next(iter(emb.values())).shape[0])


def inject(features: np.ndarray, volume_id: str, emb: dict[str, np.ndarray]) -> np.ndarray:
    """features [1, ...spatial..., C] -> [1, ...spatial..., C+D]; broadcast the volume's report-text
    vector (D = the embedding's own dim, e.g. 128 TF-IDF / 768 BERT) onto every token. Missing id ->
    zeros (kept in-distribution, no leak of a wrong report)."""
    D = emb_dim(emb)
    vec = emb.get(_strip(volume_id))
    if vec is None:
        vec = np.zeros(D, dtype=np.float32)
    tail = np.broadcast_to(vec, features.shape[:-1] + (D,)).astype(np.float32)
    return np.concatenate([features, tail], axis=-1)
