"""
Retrieval metrics for CT-CLIP

Computes image-to-text and text-to-image retrieval metrics:
- Recall@K (K = 5, 10, 50, 100)
- Mean Reciprocal Rank (MRR)
- Mean Rank
- Median Rank
- Label Overlap@K (Jaccard similarity of disease labels)

Based on the original CT-CLIP paper evaluation protocol.

Fail-fast principles:
- No try-except
- Direct numpy operations
- Explicit errors
"""
import numpy as np
from typing import Dict, List


def compute_recall_at_k(similarity_matrix: np.ndarray, k: int) -> float:
    """
    Compute Recall@K

    For each query (row in similarity_matrix), check if the correct answer
    (diagonal element) is in the top-K most similar items.

    Args:
        similarity_matrix: (N, M) similarity matrix
            - For I2T: similarity[i, j] = cosine(image[i], text[j])
            - For T2I: similarity[i, j] = cosine(text[i], image[j])
            - Correct match is at position [i, i] (diagonal)
        k: Top-K value

    Returns:
        recall@k: Proportion of queries where correct answer is in top-K

    Example:
        If similarity_matrix[0] = [0.9, 0.3, 0.5, 0.7]
        Top-3 indices: [0, 3, 2]
        Correct answer is 0, which is in top-3 → recall += 1
    """
    n = similarity_matrix.shape[0]
    recall_count = 0

    for i in range(n):
        # Get similarities for query i
        sims = similarity_matrix[i]

        # Sort in descending order and get top-K indices
        # argsort returns ascending order, so we reverse it
        top_k_indices = np.argsort(sims)[::-1][:k]

        # Check if correct answer (i) is in top-K
        if i in top_k_indices:
            recall_count += 1

    return recall_count / n


def compute_mrr(similarity_matrix: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank

    MRR is the average of reciprocal ranks of the first correct answer.

    Args:
        similarity_matrix: (N, M) similarity matrix

    Returns:
        mrr: Mean reciprocal rank

    Example:
        If correct answers rank at positions [1, 3, 2]:
        MRR = (1/1 + 1/3 + 1/2) / 3 = 0.611
    """
    n = similarity_matrix.shape[0]
    reciprocal_ranks = []

    for i in range(n):
        sims = similarity_matrix[i]

        # Sort in descending order
        sorted_indices = np.argsort(sims)[::-1]

        # Find rank of correct answer (i)
        # Rank starts from 1
        rank = np.where(sorted_indices == i)[0][0] + 1
        reciprocal_ranks.append(1.0 / rank)

    return np.mean(reciprocal_ranks)


def compute_mean_rank(similarity_matrix: np.ndarray) -> float:
    """
    Compute Mean Rank

    Average rank position of the correct answer.
    Lower is better.

    Args:
        similarity_matrix: (N, M) similarity matrix

    Returns:
        mean_rank: Average rank of correct answers
    """
    n = similarity_matrix.shape[0]
    ranks = []

    for i in range(n):
        sims = similarity_matrix[i]
        sorted_indices = np.argsort(sims)[::-1]

        # Find rank of correct answer (starts from 1)
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(rank)

    return np.mean(ranks)


def compute_median_rank(similarity_matrix: np.ndarray) -> float:
    """
    Compute Median Rank

    Median rank position of the correct answer.
    More robust to outliers than mean rank.

    Args:
        similarity_matrix: (N, M) similarity matrix

    Returns:
        median_rank: Median rank of correct answers
    """
    n = similarity_matrix.shape[0]
    ranks = []

    for i in range(n):
        sims = similarity_matrix[i]
        sorted_indices = np.argsort(sims)[::-1]

        # Find rank of correct answer (starts from 1)
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(rank)

    return np.median(ranks)


def compute_retrieval_metrics(
    image_latents: np.ndarray,
    text_latents: np.ndarray,
    k_values: List[int] = [5, 10, 50, 100]
) -> Dict[str, float]:
    """
    Compute complete retrieval metrics (I2T + T2I)

    This follows the original CT-CLIP paper evaluation protocol.

    Args:
        image_latents: (N, D) L2-normalized image embeddings
        text_latents: (N, D) L2-normalized text embeddings
        k_values: List of K values for Recall@K (default: [5, 10, 50, 100])

    Returns:
        Dictionary containing:
            'i2t_recall_at_5': float,
            'i2t_recall_at_10': float,
            'i2t_recall_at_50': float,
            'i2t_recall_at_100': float,
            'i2t_mrr': float,
            'i2t_mean_rank': float,
            'i2t_median_rank': float,
            't2i_recall_at_5': float,
            't2i_recall_at_10': float,
            't2i_recall_at_50': float,
            't2i_recall_at_100': float,
            't2i_mrr': float,
            't2i_mean_rank': float,
            't2i_median_rank': float

    Implementation:
        1. Compute similarity matrices (dot product of L2-normalized embeddings)
        2. For each query, the correct match is at the diagonal position
        3. Compute retrieval metrics based on ranking of diagonal elements
    """
    # Validate input shapes
    if image_latents.shape[0] != text_latents.shape[0]:
        raise ValueError(
            f"Number of images ({image_latents.shape[0]}) must equal "
            f"number of texts ({text_latents.shape[0]})"
        )

    if image_latents.shape[1] != text_latents.shape[1]:
        raise ValueError(
            f"Image embedding dim ({image_latents.shape[1]}) must equal "
            f"text embedding dim ({text_latents.shape[1]})"
        )

    # Compute similarity matrices using dot product
    # Since embeddings are L2-normalized, dot product = cosine similarity

    # Image-to-Text: similarity_i2t[i, j] = cosine(image[i], text[j])
    similarity_i2t = image_latents @ text_latents.T  # (N, N)

    # Text-to-Image: similarity_t2i[i, j] = cosine(text[i], image[j])
    similarity_t2i = text_latents @ image_latents.T  # (N, N)

    metrics = {}

    # ========================================================================
    # Image-to-Text Metrics
    # ========================================================================
    for k in k_values:
        metrics[f'i2t_recall_at_{k}'] = compute_recall_at_k(similarity_i2t, k)

    metrics['i2t_mrr'] = compute_mrr(similarity_i2t)
    metrics['i2t_mean_rank'] = compute_mean_rank(similarity_i2t)
    metrics['i2t_median_rank'] = compute_median_rank(similarity_i2t)

    # ========================================================================
    # Text-to-Image Metrics
    # ========================================================================
    for k in k_values:
        metrics[f't2i_recall_at_{k}'] = compute_recall_at_k(similarity_t2i, k)

    metrics['t2i_mrr'] = compute_mrr(similarity_t2i)
    metrics['t2i_mean_rank'] = compute_mean_rank(similarity_t2i)
    metrics['t2i_median_rank'] = compute_median_rank(similarity_t2i)

    return metrics


def compute_jaccard_similarity(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """
    Compute Jaccard similarity between two binary label vectors.

    This follows the CT-CLIP paper's label overlap calculation:
    Jaccard = |A ∩ B| / |A ∪ B| (ignoring double-zeros)

    Args:
        labels_a: (num_classes,) binary label vector
        labels_b: (num_classes,) binary label vector

    Returns:
        jaccard: Jaccard similarity (0.0 to 1.0)

    Example:
        labels_a = [1, 0, 1, 0, 0]
        labels_b = [1, 1, 0, 0, 0]

        both_one = [1, 0, 0, 0, 0] → sum = 1
        either_one = [1, 1, 1, 0, 0] → sum = 3

        jaccard = 1 / 3 = 0.333
    """
    both_one = np.sum((labels_a == 1) & (labels_b == 1))
    either_one = np.sum((labels_a == 1) | (labels_b == 1))

    if either_one == 0:
        # Both samples have no diseases, consider them identical
        return 1.0

    return both_one / either_one


def compute_label_overlap_at_k(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
    k: int
) -> float:
    """
    Compute Label Overlap@K (average Jaccard similarity of top-k retrieved samples)

    For each query, find top-k most similar samples (excluding self),
    compute Jaccard similarity of labels, and average.

    This follows the CT-CLIP paper's evaluation protocol for
    image-to-image retrieval with label-based relevance.

    Args:
        similarity_matrix: (N, N) similarity matrix (e.g., image @ image.T)
        labels: (N, num_classes) binary label matrix
        k: Top-K value

    Returns:
        label_overlap@k: Average Jaccard similarity of top-k results

    Example:
        Query 0's labels: [1, 0, 1, 0, 0]
        Top-5 retrieved samples (excluding self):
          - Sample 3: [1, 0, 1, 0, 0] → Jaccard = 1.0
          - Sample 7: [1, 1, 0, 0, 0] → Jaccard = 0.33
          - Sample 2: [0, 0, 1, 0, 0] → Jaccard = 0.5
          - Sample 9: [1, 0, 1, 1, 0] → Jaccard = 0.67
          - Sample 1: [0, 0, 0, 0, 0] → Jaccard = 0.0

        Label Overlap@5 = (1.0 + 0.33 + 0.5 + 0.67 + 0.0) / 5 = 0.5
    """
    n = similarity_matrix.shape[0]
    all_overlaps = []

    for i in range(n):
        sims = similarity_matrix[i].copy()

        # Exclude self (set to -inf so it won't be in top-k)
        sims[i] = -np.inf

        # Get top-k indices (excluding self)
        top_k_indices = np.argsort(sims)[::-1][:k]

        # Compute Jaccard similarity for each top-k result
        query_labels = labels[i]
        for idx in top_k_indices:
            jaccard = compute_jaccard_similarity(query_labels, labels[idx])
            all_overlaps.append(jaccard)

    return np.mean(all_overlaps)


def compute_label_overlap_metrics(
    image_latents: np.ndarray,
    labels: np.ndarray,
    k_values: List[int] = [1, 5, 10, 50]
) -> Dict[str, float]:
    """
    Compute Label Overlap metrics for image-to-image retrieval.

    This follows CT-CLIP's evaluation protocol: for each query image,
    retrieve top-k similar images and measure label overlap (Jaccard).

    NOTE: Following CT-CLIP's protocol, samples with all-zero labels
    (no pathologies) are excluded from the search space.

    Args:
        image_latents: (N, D) L2-normalized image embeddings
        labels: (N, num_classes) binary label matrix
        k_values: List of K values (default: [1, 5, 10, 50])

    Returns:
        Dictionary containing:
            'i2i_label_overlap_at_1': float,
            'i2i_label_overlap_at_5': float,
            'i2i_label_overlap_at_10': float,
            'i2i_label_overlap_at_50': float,
            'i2i_label_overlap_mean': float (average across all k values)
            'i2i_label_overlap_n_samples': int (number of samples after filtering)
    """
    # Validate input shapes
    if image_latents.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Number of images ({image_latents.shape[0]}) must equal "
            f"number of label rows ({labels.shape[0]})"
        )

    # Filter: Keep only samples with at least one positive label (CT-CLIP style)
    label_sums = labels.sum(axis=1)  # (N,)
    valid_mask = label_sums > 0

    filtered_latents = image_latents[valid_mask]
    filtered_labels = labels[valid_mask]

    n_filtered = filtered_latents.shape[0]

    # If no valid samples, return zeros
    if n_filtered == 0:
        metrics = {f'i2i_label_overlap_at_{k}': 0.0 for k in k_values}
        metrics['i2i_label_overlap_mean'] = 0.0
        metrics['i2i_label_overlap_n_samples'] = 0
        return metrics

    # Compute image-to-image similarity matrix on filtered data
    similarity_i2i = filtered_latents @ filtered_latents.T  # (n_filtered, n_filtered)

    metrics = {}
    overlap_values = []

    for k in k_values:
        # Ensure k doesn't exceed n-1 (excluding self)
        effective_k = min(k, n_filtered - 1)
        overlap = compute_label_overlap_at_k(similarity_i2i, filtered_labels, effective_k)
        metrics[f'i2i_label_overlap_at_{k}'] = overlap
        overlap_values.append(overlap)

    # Also compute mean across all k values
    metrics['i2i_label_overlap_mean'] = np.mean(overlap_values)
    metrics['i2i_label_overlap_n_samples'] = n_filtered

    return metrics
