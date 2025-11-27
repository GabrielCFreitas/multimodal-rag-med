import argparse
import os
import torch
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from src.dataset.mimic_dataset import MIMICDataset


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_models(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(
        config['processor']['model'],
        token=config['processor'].get('auth_token', None)
    )
    model = AutoModel.from_pretrained(
        config['processor']['model'],
        token=config['processor'].get('auth_token', None)
    ).to(device)
    return processor, model, device


def get_image_embeddings_flat(samples, processor, model, device, chunk_size=128):
    """
    Recebe uma lista de samples e retorna embeddings por imagem (flattened),
    junto com o índice do sample e o nome da imagem.
    """
    if isinstance(samples, dict):
        samples = [samples]

    images_flat = []
    sample_idx = []
    image_names = []
    for i, s in enumerate(samples):
        imgs = s.get("images", [])
        paths = s.get("image_paths", [None] * len(imgs))
        for j, img in enumerate(imgs):
            images_flat.append(img)
            sample_idx.append(i)
            pname = None
            if paths and j < len(paths) and paths[j] is not None:
                pname = os.path.basename(paths[j])
            image_names.append(pname or f"{s.get('study_id')}_img{j}")

    if len(images_flat) == 0:
        return torch.zeros((0, 0), device=device), sample_idx, image_names

    proc = getattr(processor, "image_processor", processor)

    embs_chunks = []
    with torch.no_grad():
        for start in range(0, len(images_flat), chunk_size):
            chunk = images_flat[start:start + chunk_size]
            batch = proc(chunk, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}

            # usa get_image_features (projeção + normalização quando disponível)
            img_feats = model.get_image_features(**batch)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            # get_image_features normalmente retorna torch.Tensor
            if not isinstance(img_feats, torch.Tensor):
                img_feats = torch.tensor(img_feats, device=device)
                
            embs_chunks.append(img_feats)

    img_embs = torch.cat(embs_chunks, dim=0)
    return img_embs, sample_idx, image_names


def process_images_only(dataset, processor, model, device, batch_size=32, chunk_size=128):
    """
    Itera sobre o dataset em batches de estudos, extrai embeddings por imagem
    e retorna lista de registros por imagem.
    """
    per_image_records = []
    for batch_start in tqdm(range(0, len(dataset), batch_size),
                            desc="Processando imagens por estudo",
                            total=(len(dataset) + batch_size - 1) // batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_samples = [dataset[i] for i in range(batch_start, batch_end)]

        try:
            img_embs, sample_idx, image_names = get_image_embeddings_flat(
                batch_samples, processor, model, device, chunk_size=chunk_size
            )

            for emb, sidx, iname in zip(img_embs, sample_idx, image_names):
                s = batch_samples[sidx]
                rec = {
                    "patient_id": s.get("patient_id"),
                    "study_id": s.get("study_id"),
                    "image_name": iname,
                    "embedding_image": emb.cpu().numpy()
                }
                per_image_records.append(rec)

        except Exception as e:
            print(f"Erro no batch {batch_start}-{batch_end}: {e}")
            continue

    return per_image_records


def save_per_image_records(per_image_records, output_path):
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    # metadata as (patient_id, study_id, image_name)
    metas = []
    embs = []
    for r in per_image_records:
        metas.append((r["patient_id"], r["study_id"], r["image_name"]))
        embs.append(r["embedding_image"])

    if len(embs) == 0:
        metas = np.array([], dtype=object)
        embs = np.zeros((0, 0), dtype=float)
    else:
        metas = np.array(metas, dtype=object)
        embs = np.stack(embs, axis=0)

    np.save(out / "image_metadata.npy", metas, allow_pickle=True)
    np.save(out / "image_embeddings.npy", embs)
    np.save(out / "embeddings_per_image.npy", per_image_records, allow_pickle=True)

    return out


def main(config_path, batch_size=32, chunk_size=128):
    cfg = load_config(config_path)
    processor, model, device = load_models(cfg)

    ds_root = cfg['processor']['root_folder']
    # tokenizer not needed since we're only computing image embeddings
    dataset = MIMICDataset(ds_root, tokenizer=None)

    per_image_records = process_images_only(dataset, processor, model, device,
                                            batch_size=batch_size, chunk_size=chunk_size)

    out = cfg.get('paths', {}).get('embeddings', "artifacts/img_embeddings")
    saved = save_per_image_records(per_image_records, out)

    print(f"✓ Total imagens processadas: {len(per_image_records)}")
    print(f"✓ Salvo em: {saved}")
    return per_image_records, saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa embeddings por imagem (apenas imagens)")
    parser.add_argument("--config", type=str, default="configs/imagens_embd.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=128, help="chunk para processamento das imagens flattened")
    args = parser.parse_args()
    main(args.config, batch_size=args.batch_size, chunk_size=args.chunk_size)