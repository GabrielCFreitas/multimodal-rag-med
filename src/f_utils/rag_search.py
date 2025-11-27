import faiss
import numpy as np

def search_relevant_cases(query_emb, vector_store, ids, k=3):
    """"
    Pesquisa os estudos mais relevantes para uma determinada consulta.
    Args:
        - query_emb: embedding do estudo/imagem/texto de consulta
        - vector_store: vector store
        - k: número de casos mais relevantes a serem retornados
    Returns:
        - IDs dos casos mais relevantes
        - Casos mais relevantes
    """
    # Preparar o vetor de consulta
    if isinstance(query_emb, np.ndarray):
        # Garantir que é 2D e float32
        if query_emb.ndim == 1:
            query_embedding = query_emb.reshape(1, -1).astype(np.float32)
        else:
            query_embedding = query_emb.astype(np.float32)
    else:
        query_embedding = np.array(query_emb, dtype=np.float32).reshape(1, -1)
    
    # Normalizar o vetor de consulta
    faiss.normalize_L2(query_embedding)

    # Pesquisar os casos mais relevantes
    D, I = vector_store.search(query_embedding, k+1)  # +1 para pular o próprio caso de consulta

    # Retornar os IDs dos casos mais relevantes
    results = []
    for idx in I[0]:
        results.append(ids[idx])
        
    return results[1:k+1], I[0][1:k+1] # +1 para pular o próprio caso de consulta