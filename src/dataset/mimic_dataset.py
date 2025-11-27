import os
import glob
import re
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MIMICDataset(Dataset):
    """
    PyTorch Dataset para MIMIC-CXR (versão JPG, nível de estudo).
    Cada item é um estudo (study_id), contendo:
        - Lista de imagens JPG do estudo
        - Texto da seção "Findings"
        - Metadados (patient_id, study_id, image_paths)
    """

    def __init__(self, root_dir, tokenizer=None, transform=None, max_tokens=64):
        """
        Args:
            root_dir (str): Caminho raiz (ex: 'data/mimic-cxr-jpg/files/')
            tokenizer (callable): Tokenizer opcional (ex: MedSigLIP.tokenizer)
            transform (callable): Transformações de imagem (ex: Resize, ToTensor)
            max_tokens (int): Limite de tokens do texto (ex: 64)
        """
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.transform = transform or transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])
        self.max_tokens = max_tokens
        self.studies = self._collect_studies()

    def _collect_studies(self):
        """
        Percorre a estrutura pXX/pXXXXXXX/sYYYYYYY e coleta estudos completos.
        Retorna lista de dicionários:
        [{ 'patient_id': ..., 'study_id': ..., 'txt_path': ..., 'image_paths': [...] }, ...]
        """
        studies = []
        patient_dirs = sorted(glob.glob(os.path.join(self.root_dir, "p*")))
        patient_dirs = glob.glob(os.path.join(self.root_dir, "p*"))

        for pdir in sorted(patient_dirs):
            for patient_path in sorted(glob.glob(os.path.join(pdir, "p*"))):
                study_paths = glob.glob(os.path.join(patient_path, "s*"))
                for study_path in sorted(study_paths):

                    # caminho do laudo textual
                    txt_path = os.path.join(patient_path, os.path.basename(study_path) + ".txt")
                    # imagens JPG do estudo
                    image_paths = sorted(glob.glob(os.path.join(study_path, "*.jpg")))

                    if len(image_paths) == 0 or not os.path.exists(txt_path):
                        continue  # ignora estudos incompletos

                    studies.append({
                        "patient_id": os.path.basename(patient_path),
                        "study_id": os.path.basename(study_path),
                        "txt_path": txt_path,
                        "image_paths": image_paths
                    })
        return studies

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        """
        Retorna um estudo completo.
        Output:
            {
                'patient_id': str,
                'study_id': str,
                'images': Tensor [N,3,H,W],
                'text': str,
                'tokenized_text': dict ou None,
                'image_paths': [str]
            }
        """
        study_info = self.studies[idx]
        patient_id = study_info["patient_id"]
        study_id = study_info["study_id"]
        txt_path = study_info["txt_path"]
        image_paths = study_info["image_paths"]

        # === 1) Extrai o texto (Findings)
        text_full = open(txt_path, "r").read()
        findings = self._extract_findings(text_full)

        tokenized_text = None
        if self.tokenizer is not None:
            tokenized_text = self.tokenizer(
                text=findings,
                truncation=True,
                padding="max_length",
                max_length=self.max_tokens,
                return_tensors="pt"
            )

        # === 2) Carrega e processa as imagens
        imgs = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            except Exception as e:
                print(f"[Aviso] Falha ao carregar {path}: {e}")
                continue

        if len(imgs) == 0:
            raise ValueError(f"Nenhuma imagem carregada para estudo {study_id}")

        images_tensor = torch.stack(imgs)

        # === 3) Retorna o estudo
        return {
            "patient_id": patient_id,
            "study_id": study_id,
            "images": images_tensor,          # Tensor [N,3,H,W]
            "text": findings,                 # Texto do laudo (Findings)
            "tokenized_text": tokenized_text, # Se fornecido
            "image_paths": image_paths
        }

    def _extract_findings(self, report_text):
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
    