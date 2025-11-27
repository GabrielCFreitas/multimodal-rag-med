import pandas as pd
import numpy as np
import os
import torch
import re
from PIL import Image

def load_embeddings(embeddings_file: str) -> np.ndarray:
    """
    Carrega embeddings salvos de acordo com caminho do arquivo.
    Args:
        - embeddings_file: caminho do arquivo.
    Returns:
        - embeddings: numpy.ndarray
    """
    
    study_ids_file = embeddings_file
    
    if not os.path.exists(study_ids_file):
        print(f"❌ Arquivo não encontrado: {study_ids_file}")
        return None
    
    try:
        # Carregar o arquivo numpy
        embeddings = np.load(study_ids_file, allow_pickle=True)
        
        print(f"✅ Embeddings carregados com sucesso!")
        print(f"📊 Formato dos dados: {type(embeddings)}")
        print(f"📊 Shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}")
        
        return embeddings
        
    except Exception as e:
        print(f"❌ Erro ao carregar embeddings: {e}")
        return None
    

def extract_embeddings_from_text(text,model,tokenizer):
    """
    Extrai embeddings de um texto usando um modelo de linguagem pré-treinado.
    Args:
        - text: texto a ser tranformado em embedding (assume que o pré-processamento já foi feito).
        - tokenizer: tokenizador do modelo.
        - model: modelo de linguagem pré-treinado.
    Returns:
        - embedding: embedding do texto.
    """

    # Tokenizar o texto
    tokens = tokenizer(text, truncation=True, padding="max_length", return_tensors="pt")

    # Passar os tokens pelo modelo
    with torch.no_grad():
        emb = model.get_text_features(**tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    
    return emb


def extract_embeddings_from_img(imgs,model,processor):
    """
    Extrai embeddings de uma imagem usando um modelo de linguagem pré-treinado.
    Args:
        - imgs: lista de caminhos de imagens .jpg a ser tranformada em embedding
        - model: modelo de linguagem pré-treinado
        - processor: processador do modelo
    Returns:
        - embeddings: lista embeddings das imagens
    """

    emb_imgs = []
    for img in imgs:
        # Carregar a imagem
        img = Image.open(img).convert("RGB")

        # processar a imagem
        inputs = processor(images=img, return_tensors="pt")

        # Passar os tokens pelo modelo
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        
        emb_imgs.append(emb)

    return emb_imgs


def extract_embedding_single_study(texto, imagens, model, tokenizer, processor, alpha=0.5):
    """
    Extrai embeddings de texto e imagens a partir de um de estudo.
    Args:
        - texto: texto do estudo (laudo - partindo do pre suposto que já passou por extract_findings)
        - imagens: lista de imagens do estudo
        - alpha: peso de ponderacao dos embeddings de texto e imagem  
    """

    # === 1) tokens e embeddings do texto ===
    emb_text = extract_embeddings_from_text(texto, tokenizer=tokenizer, model=model)#.detach().numpy()  # embedding do texto

    # === 2) tokens e embeddings das imagens ===
    emb_images = extract_embeddings_from_img(imagens, model=model, processor=processor)

    # === 3) faz pooling com imagens de entrada
    # Stacking para [N, D]
    emb_images = torch.stack(emb_images)  # [num_imagens, embedding_dim]
    
    # Pooling (média) ao longo das imagens
    emb_pool = emb_images.mean(dim=0)  # [embedding_dim]
    emb_pool = emb_pool / emb_pool.norm(dim=-1, keepdim=True)
    #print(emb_pool.shape)

    # === 4) fez media dos embeddings para embedding final
    e_study = alpha * emb_text + (1 - alpha) * emb_pool
    e_study = e_study / e_study.norm(dim=-1, keepdim=True)

    return e_study


def _extract_findings(report_text):
        """
        Extrai o texto entre 'FINDINGS:' e a próxima seção.
        Caso não encontre, retorna o texto completo truncado.
        """

        findings = ""
        try:
            match = re.search(r"FINDINGS:(.*?)(?:IMPRESSION:|CONCLUSION:|$)", report_text, flags=re.S | re.I)
            if match:
                findings = match.group(1).strip()
            else:
                # Normaliza quebras de linha para \n
                text = report_text.replace("\r\n", "\n").replace("\r", "\n").strip()

                # Divide em blocos por linhas em branco (um ou mais \n com espaços possivelmente)
                blocks = re.split(r"\n\s*\n+", text)

                best_block = ""
                best_score = -1

                for block in blocks:
                    b = block.strip()

                    # Remove um possível cabeçalho "TÍTULO:" no início do bloco
                    # (títulos normalmente em maiúsculas, números e símbolos comuns)
                    b_clean = re.sub(r"^\s*[A-Z0-9 ,./()\-]+:\s*", "", b)

                    # Se ficar vazio, volta ao bloco original
                    if not b_clean:
                        b_clean = b

                    # Calcula um "score" para decidir o maior bloco:
                    # - comprimento do texto sem colapsar as quebras (para preservar formato)
                    score = len(b_clean)

                    if score > best_score:
                        best_score = score
                        best_block = b_clean

                findings = best_block.strip()
        except:
            findings = report_text.strip()
        return findings
    