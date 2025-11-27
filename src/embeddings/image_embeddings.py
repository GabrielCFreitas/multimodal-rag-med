
import torch

# def encode_images_study(sample, processor, model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
#     """
#     Recebe um item do dataset (sample) e gera o embedding médio das imagens JPG do estudo.

#     Args:
#         sample: item retornado pelo MIMICCXRStudyDataset_JPG
#         processor: AutoProcessor do MedSigLIP
#         model: AutoModel do MedSigLIP
#         device: 'cuda' ou 'cpu'

#     Retorna:
#         Tensor normalizado [1, D] (embedding médio das imagens)
#     """
#     imgs = sample["images"]  # Tensor [N,3,H,W]
#     e_imgs = []

#     for img in imgs:
#         # processor espera listas de imagens, mesmo que seja uma
#         inputs = processor(images=img, return_tensors="pt").to(device)
#         with torch.no_grad():
#             emb = model.get_image_features(**inputs)
#             emb = emb / emb.norm(dim=-1, keepdim=True)
#         e_imgs.append(emb)

#     e_imgs = torch.stack(e_imgs)  # [N,1,D]
#     e_img_pool = e_imgs.mean(dim=0)
#     e_img_pool = e_img_pool / e_img_pool.norm(dim=-1, keepdim=True)

#     return e_img_pool

def encode_images_study(samples, processor, model, device, chunk_size=512):
    """
    Codifica imagens em batch e retorna um embedding por sample (média das imagens do estudo).
    Usa model.get_image_features quando disponível (projeção + normalização).
    Processa em chunks para controlar uso de memória.

    Args:
        samples: Lista de samples ou um sample único (cada sample['images'] pode ser tensor (N,C,H,W) ou lista de imagens)
        processor: Processor do modelo (AutoProcessor / image_processor)
        model: Modelo (deve expor .get_image_features)
        device: Device (cuda/cpu)
        chunk_size: número máximo de imagens processadas por iteração

    Returns:
        Embeddings de shape (batch_size, embedding_dim)
    """
    if isinstance(samples, dict):
        samples = [samples]

    # Flatten: junta todas as imagens de todos os samples
    images_flat = []
    sample_idx = []
    for i, s in enumerate(samples):
        imgs = s["images"]
        for img in imgs:
            images_flat.append(img)
            sample_idx.append(i)

    if len(images_flat) == 0:
        emb_dim = model.config.projection_dim if hasattr(model.config, "projection_dim") else model.config.hidden_size
        return torch.zeros(len(samples), emb_dim, device=device)

    proc = getattr(processor, "image_processor", processor)

    embs_chunks = []
    with torch.no_grad():
        for start in range(0, len(images_flat), chunk_size):
            chunk = images_flat[start:start + chunk_size]
            batch = proc(chunk, return_tensors="pt")
            # move all tensors to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # usa get_image_features (projeção + normalização quando disponível)
            img_feats = model.get_image_features(**batch)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            # get_image_features normalmente retorna torch.Tensor
            if not isinstance(img_feats, torch.Tensor):
                img_feats = torch.tensor(img_feats, device=device)

            embs_chunks.append(img_feats)

    img_embs = torch.cat(embs_chunks, dim=0)  # (total_images, dim)

    # Agrupa e faz média por sample
    batch_size = len(samples)
    dim = img_embs.size(1)
    device = img_embs.device

    sums = torch.zeros(batch_size, dim, device=device)
    counts = torch.zeros(batch_size, device=device)
    for idx_img, emb in zip(sample_idx, img_embs):
        sums[idx_img] += emb
        counts[idx_img] += 1.0

    counts = counts.clamp(min=1.0)
    mean_embs = sums / counts.unsqueeze(1)

    mean_embs = mean_embs / mean_embs.norm(dim=-1, keepdim=True)

    return mean_embs