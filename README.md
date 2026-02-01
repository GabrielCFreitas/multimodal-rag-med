# Multimodal RAG for Medical Images and Text

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)](https://github.com/facebookresearch/faiss)
[![MedSIGLIP](https://img.shields.io/badge/Model-MedSIGLIP--448-orange)](https://huggingface.co/google/medsiglip-448)

Sistema de Recuperação Aumentada por Geração (RAG) multimodal para busca e análise de casos médicos, combinando imagens radiológicas e relatórios textuais do dataset MIMIC-CXR.

**Projeto desenvolvido para a disciplina IA368 - Deep Learning - Unicamp**

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Características](#características)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Metodologia](#metodologia)
- [Avaliação](#avaliação)
- [Notebooks](#notebooks)
- [Licença](#licença)

## 🎯 Sobre o Projeto

Este projeto implementa um sistema RAG (Retrieval-Augmented Generation) multimodal para recuperação de casos médicos similares, utilizando tanto imagens radiológicas quanto relatórios textuais. O sistema é baseado no dataset MIMIC-CXR e utiliza embeddings combinados de texto e imagem para realizar buscas semânticas.

### Objetivos

- **Recuperação Multimodal**: Combinar informações de texto e imagem para melhorar a precisão da busca
- **Busca Semântica**: Utilizar embeddings densos para capturar similaridade semântica
- **Aplicação Médica**: Apoiar profissionais de saúde na busca de casos similares

## ✨ Características

- 🔍 **Busca Multimodal**: Combina embeddings de texto e imagem com peso configurável (parâmetro α)
- 🏥 **Dataset MIMIC-CXR**: Utiliza dados reais de radiografias torácicas e relatórios médicos
- 🚀 **FAISS Vector Store**: Busca eficiente em larga escala usando índices FAISS
- 🤖 **MedSIGLIP-448**: Modelo especializado em domínio médico para extração de features
- 📊 **Métricas de Avaliação**: Implementação de NDCG, Precision, Recall e Jaccard
- 🔬 **Análise Abrangente**: Notebooks detalhados para experimentação e avaliação

## 📁 Estrutura do Repositório

```
multimodal-rag-med/
├── configs/               # Arquivos de configuração
│   ├── configs.yaml       # Configurações gerais
│   └── imagens_embd.yaml  # Configurações de embeddings de imagem
├── notebooks/             # Notebooks Jupyter para experimentação
│   ├── 00_exploracao_dataset.ipynb
│   ├── 01_vector_store.ipynb
│   ├── 02_busca_rag.ipynb
│   ├── 03_gabarito.ipynb
│   ├── 04_process_all_imagens_sep.ipynb
│   ├── 07_dataset_validacao.ipynb
│   ├── 08-13_avaliacao_*.ipynb
│   └── 15_graficos_avaliacao.ipynb
├── scripts/               # Scripts de pré-processamento
│   ├── preprocess_embeddings.py
│   └── preprocess_img_embeddings.py
└── src/                   # Código fonte
    ├── dataset/           # Classes para carregamento do dataset
    ├── embeddings/        # Extração de embeddings
    └── f_utils/           # Funções utilitárias
        ├── embedding_utils.py
        ├── evaluation.py
        ├── mimic_labels.py
        └── rag_search.py
```

## Pré-requisitos

- Python 3.8+
- CUDA (recomendado para GPU)
- Acesso ao dataset MIMIC-CXR

**Acesso ao dataset MIMIC-CXR**
   - Obtenha credenciais em https://physionet.org/
   - Baixe o dataset MIMIC-CXR
   - Configure o token do HuggingFace

## 🔬 Metodologia

### Extração de Embeddings

1. **Texto**: Embeddings extraídos da seção "FINDINGS" dos relatórios radiológicos
2. **Imagem**: Múltiplas imagens por estudo são processadas e agregadas
3. **Combinação**: Embeddings de texto e imagem são combinados usando peso α:

$$
\text{embedding}_{\text{final}} = \alpha \cdot \text{embedding}_{\text{texto}} + (1-\alpha) \cdot \text{embedding}_{\text{imagem}}
$$

### Vector Store

- **Índice FAISS**: Utiliza índice Flat (L2) para busca exata
- **Três modalidades**: Texto, Imagem e Multimodal (combinado)

### Busca

```python
# Busca com k vizinhos mais próximos
distances, indices = vector_store.search(query_embedding, k)
```

## 📊 Avaliação

O sistema é avaliado usando múltiplas métricas:

- **NDCG@k**: Normalized Discounted Cumulative Gain
- **Precision@k**: Precisão nos top-k resultados
- **Recall@k**: Revocação nos top-k resultados
- **Jaccard Similarity**: Similaridade entre labels

### Resultados

Os resultados das avaliações estão disponíveis em:
- `artifacts/resultados/` - Resultados numéricos
- `notebooks/15_graficos_avaliacao.ipynb` - Visualizações

## 📓 Notebooks

| Notebook | Descrição |
|----------|-----------|
| `00_exploracao_dataset.ipynb` | Exploração inicial do dataset MIMIC-CXR |
| `01_vector_store.ipynb` | Criação dos índices FAISS |
| `02_busca_rag.ipynb` | Demonstração de buscas RAG |
| `03_gabarito.ipynb` | Preparação de labels ground truth |
| `04_process_all_imagens_sep.ipynb` | Processamento de embeddings de imagens |
| `07_dataset_validacao.ipynb` | Criação do dataset de validação |
| `08_avaliacao_val_dataset.ipynb` | Avaliação no dataset de validação |
| `09_avaliacao_txt.ipynb` | Avaliação - apenas texto |
| `10_avaliacao_img.ipynb` | Avaliação - apenas imagem |
| `11_avaliacao_completa.ipynb` | Avaliação multimodal completa |
| `12_avaliacao_completa_txt.ipynb` | Análise detalhada - texto |
| `13_avaliacao_completa_img.ipynb` | Análise detalhada - imagem |
| `15_graficos_avaliacao.ipynb` | Visualização dos resultados |

## 🛠️ Tecnologias Utilizadas

- **PyTorch**: Framework de deep learning
- **Transformers (HuggingFace)**: Modelos pré-treinados
- **FAISS**: Busca de similaridade em larga escala
- **NumPy/Pandas**: Manipulação de dados
- **Matplotlib/Seaborn**: Visualização
- **MedSIGLIP-448**: Modelo multimodal especializado em medicina

## 📚 Referências

- **MIMIC-CXR Database**: Johnson et al. (2019)
- **FAISS**: Johnson et al., Facebook AI Research
- **MedSIGLIP**: Google Research

## 👥 Autores

- Maria Fernanda Bosco - [@mfbosco](https://github.com/mfbosco)
- Gabriel Carvalho de Freitas - [@GabrielCFreitas](https://github.com/GabrielCFreitas)
  
Projeto desenvolvido como parte da disciplina IA368 - Deep Learning Avançado  
Engenharia de Computação - Unicamp

## 📄 Licença

Este projeto está sob licença acadêmica. O dataset MIMIC-CXR requer credenciais e concordância com termos de uso específicos.

---

**Nota**: Este projeto utiliza dados médicos reais. Certifique-se de seguir todas as diretrizes éticas e de privacidade ao trabalhar com dados sensíveis.
