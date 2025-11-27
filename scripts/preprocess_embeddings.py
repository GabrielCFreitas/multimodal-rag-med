import argparse
import torch
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from src.dataset.mimic_dataset import MIMICDataset
from src.embeddings.text_embeddings import encode_text_findings
from src.embeddings.image_embeddings import encode_images_study


def load_config(config_path):
    """Carrega arquivo de configuração."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_models(config):
    """Carrega processor e modelo."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = AutoProcessor.from_pretrained(
        config['processor']['model'], 
        token=config['processor']['auth_token']
    )
    model = AutoModel.from_pretrained(
        config['processor']['model'], 
        token=config['processor']['auth_token']
    ).to(device)
    
    return processor, model, device


def process_embeddings(dataset, processor, model, config, device, batch_size=32):
    """Processa e retorna embeddings de todo o dataset em batches."""
    embeddings_data = {
        'patient_ids': [],
        'study_ids': [],
        'e_text': [],
        'e_img': [],
        'e_study': []
    }
    
    alpha = config['model_params']['alpha']
    
    # Processa em batches
    for batch_start in tqdm(
        range(0, len(dataset), batch_size),
        desc="Processando embeddings",
        total=(len(dataset) + batch_size - 1) // batch_size
    ):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_samples = [dataset[idx] for idx in range(batch_start, batch_end)]
        
        try:
            # Gera embeddings em batch (sem processor)
            e_text = encode_text_findings(batch_samples, model, device)
            e_img = encode_images_study(batch_samples, processor, model, device)
            
            # Combina (multimodal)
            e_study = alpha * e_text + (1 - alpha) * e_img
            e_study = e_study / e_study.norm(dim=-1, keepdim=True)
            
            # Armazena dados
            for i, sample in enumerate(batch_samples):
                embeddings_data['patient_ids'].append(sample["patient_id"])
                embeddings_data['study_ids'].append(sample["study_id"])
                embeddings_data['e_text'].append(e_text[i].cpu().numpy())
                embeddings_data['e_img'].append(e_img[i].cpu().numpy())
                embeddings_data['e_study'].append(e_study[i].cpu().numpy())
                
        except Exception as e:
            print(f"Erro processando batch {batch_start}-{batch_end}: {e}")
            continue
    
    return embeddings_data

# def process_embeddings(dataset, processor, model, config, device):
#     """Processa e retorna embeddings de todo o dataset."""
#     embeddings_data = {
#         'patient_ids': [],
#         'study_ids': [],
#         'e_text': [],
#         'e_img': [],
#         'e_study': []
#     }
    
#     alpha = config['model_params']['alpha']
    
#     for idx in tqdm(range(len(dataset)), desc="Processando embeddings"):
#         try:
#             sample = dataset[idx]
            
#             # Gera embeddings
#             e_text = encode_text_findings(sample, processor, model)
#             e_img = encode_images_study(sample, processor, model)
            
#             # Combina (multimodal)
#             e_study = alpha * e_text + (1 - alpha) * e_img
#             e_study = e_study / e_study.norm(dim=-1, keepdim=True)
            
#             # Armazena dados
#             embeddings_data['patient_ids'].append(sample["patient_id"])
#             embeddings_data['study_ids'].append(sample["study_id"])
#             embeddings_data['e_text'].append(e_text.cpu().numpy())
#             embeddings_data['e_img'].append(e_img.cpu().numpy())
#             embeddings_data['e_study'].append(e_study.cpu().numpy())
            
#         except Exception as e:
#             print(f"Erro processando amostra {idx}: {e}")
#             continue
    
#     return embeddings_data


def save_embeddings(embeddings_data, output_path):
    """Salva embeddings em arquivos .npy."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / "patient_ids.npy", embeddings_data['patient_ids'])
    np.save(output_path / "study_ids.npy", embeddings_data['study_ids'])
    np.save(output_path / "e_text.npy", np.array(embeddings_data['e_text']))
    np.save(output_path / "e_img.npy", np.array(embeddings_data['e_img']))
    np.save(output_path / "e_study.npy", np.array(embeddings_data['e_study']))
    
    return output_path


def main(config_path):
    """Pipeline principal de pré-processamento de embeddings."""
    print("Carregando configuração...")
    config = load_config(config_path)
    
    print("Carregando processor e modelo...")
    processor, model, device = load_models(config)
    print(f"Usando device: {device}")
    
    print("Carregando dataset...")
    dataset = MIMICDataset(
        config['processor']['root_folder'], 
        tokenizer=processor.tokenizer
    )
    
    print(f"Processando {len(dataset)} amostras...")
    embeddings_data = process_embeddings(dataset, processor, model, config, device)
    
    output_path = config['paths']['embeddings']
    print(f"Salvando embeddings em {output_path}...")
    save_embeddings(embeddings_data, output_path)
    
    print(f"✓ Pré-processamento concluído!")
    print(f"  Total de amostras: {len(embeddings_data['e_study'])}")
    
    return embeddings_data, output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pré-processa embeddings do dataset MIMIC"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/configs.yaml",
        help="Caminho para arquivo de configuração"
    )
    
    args = parser.parse_args()
    main(args.config)