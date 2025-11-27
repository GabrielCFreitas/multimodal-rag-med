import numpy as np
from math import log2
# from src.f_utils.mimic_labels import _get_gabarito, _get_gabarito_any

def jaccard(a, b):
    """a e b são conjuntos de strings (labels)."""
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / union if union > 0 else 0.0


def dcg(relevances):
    """Cálculo do DCG para uma lista de relevâncias."""
    return sum(rel / log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(relevances, k):
    """Relevances = lista de relevâncias contínuas (ex: Jaccards)."""
    relevances = relevances[:k]
    ideal = sorted(relevances, reverse=True)
    return dcg(relevances) / dcg(ideal) if dcg(ideal) > 0 else 0.0


def is_relevant(query_labels, candidate_labels):
    return len(query_labels.intersection(candidate_labels)) > 0


def evaluate_single_query(query_labels, retrieved_labels, gabarito, k):
    """
    query_labels: set de labels da query
    retrieved_labels: lista de sets de labels dos resultados (na ordem retornada)
    k: corte para cálculo das métricas
    """

    # === 1) Métrica binária de relevância ===
    relevance_binary = [
        1 if is_relevant(query_labels, labs) else 0
        for labs in retrieved_labels
    ]

    # === 2) Similaridade contínua (Jaccard) ===
    relevance_jaccard = [
        jaccard(query_labels, labs)
        for labs in retrieved_labels
    ]

    # === Precision@k ===
    prec_at_k = sum(relevance_binary[:k]) / k

    # === Recall@k ===
    total_relevant = len(gabarito)  # nº total de casos relevantes associados ao estudo.
    recall_at_k = (
        sum(relevance_binary[:k]) / total_relevant
        if total_relevant > 0 else 0.0
    )

    # === Jaccard_1@k (proporção em que jaccard=1) ===
    jaccard_1_at_k = np.mean([1 if j == 1 else 0 for j in relevance_jaccard[:k]]) if k > 0 else 0.0

    # === Jaccard@k (média) ===
    jaccard_at_k = np.mean(relevance_jaccard[:k]) if k > 0 else 0.0

    # === NDCG@k ===
    ndcg_k = ndcg_at_k(relevance_jaccard, k)

    return {
        "precision@k": prec_at_k,
        "recall@k": recall_at_k,
        "jaccard_1@k": jaccard_1_at_k,
        "jaccard@k": jaccard_at_k,
        "ndcg@k": ndcg_k
    }


def evaluate_dataset(queries, retrieved, k=10):
    """
    queries: lista de sets de labels para cada estudo de consulta
    retrieved: lista de listas de sets (resultados por query)
              retrieved[i][j] = labels do resultado j da query i
    """

    results = []

    for q_labels, r_labels in zip(queries, retrieved):
        metrics = evaluate_single_query(q_labels, r_labels, k)
        results.append(metrics)

    # Média final das métricas
    mean_metrics = {
        m: np.mean([res[m] for res in results])
        for m in results[0]
    }

    return results,mean_metrics
