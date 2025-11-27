
import torch

# def encode_text_findings(sample, processor, model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
#     """
#     Recebe um item do dataset (sample) e gera o embedding textual (Findings) com o MedSigLIP.

#     Args:
#         sample: item retornado pelo MIMICCXRStudyDataset_JPG
#         processor: AutoProcessor do MedSigLIP
#         model: AutoModel do MedSigLIP
#         device: 'cuda' ou 'cpu'

#     Retorna:
#         Tensor normalizado [1, D] (embedding textual)
#     """
#     text = sample["text"]

#     inputs = processor(
#         text=text,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=64
#     ).to(device)

#     with torch.no_grad():
#         emb = model.get_text_features(**inputs)
#         emb = emb / emb.norm(dim=-1, keepdim=True)

#     return emb

def encode_text_findings(samples, model, device):
    """
    Codifica textos de findings em batch usando tokens já processados.
    
    Args:
        samples: Lista de samples ou um sample único
        model: Modelo
        device: Device (cuda/cpu)
    
    Returns:
        Embeddings de shape (batch_size, embedding_dim)
    """
    # Se for um sample único, converte para lista
    if isinstance(samples, dict):
        samples = [samples]
    
    # Coleta os tokenized_text de todos os samples
    batch_input_ids = []
    # batch_attention_mask = []
    
    for sample in samples:
        tokenized = sample["tokenized_text"]
        batch_input_ids.append(tokenized["input_ids"].squeeze(0))
        # batch_attention_mask.append(tokenized["attention_mask"].squeeze(0))
    
    # Faz padding para o tamanho máximo do batch
    max_length = max(t.shape[0] for t in batch_input_ids)
    
    padded_input_ids = torch.zeros(len(samples), max_length, dtype=torch.long).to(device)
    # padded_attention_mask = torch.zeros(len(samples), max_length, dtype=torch.long).to(device)
    
    for i, input_ids in enumerate(batch_input_ids):
        padded_input_ids[i, :input_ids.shape[0]] = input_ids.to(device)
    
    with torch.no_grad():
        emb = model.get_text_features(padded_input_ids)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    
    return emb