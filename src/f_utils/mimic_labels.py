import pandas as pd
import numpy as np

def _get_gabarito(study_id, df_resp, id_col = 'study_id'):

    ref = df_resp.loc[df_resp[id_col] == study_id].copy()
    if ref.empty:
        raise ValueError(f"ID {study_id} not found in dataset.")
    
    ref.fillna(0, inplace=True)
    
    df_resp = df_resp.fillna(0)

    binary_cols = [col for col in df_resp.columns if col not in ['subject_id', 'study_id']]

    ref_vals = ref[binary_cols].to_numpy()
    df_vals = df_resp[binary_cols].to_numpy()
    mask = (df_vals == ref_vals).all(axis=1)

    matches = df_resp[mask]

    return matches

def _get_gabarito_any(study_id, df_resp, id_col='study_id'):
    """
    Retorna todos os estudos que compartilham pelo menos um label (valor=1) com o estudo de referência.
    """
    ref = df_resp.loc[df_resp[id_col] == study_id].copy()
    if ref.empty:
        raise ValueError(f"ID {study_id} not found in dataset.")
    
    df_resp = df_resp.fillna(0)
    ref = ref.fillna(0)

    # Identificar colunas binárias (excluindo IDs)
    binary_cols = [col for col in df_resp.columns if col not in ['subject_id', 'study_id']]

    # Obter labels com valor 1 do estudo de referência
    ref_row = ref[binary_cols].iloc[0]
    ref_labels = set(ref_row[ref_row == 1].index)  # Usar .index em vez de .columns
    
    # print(f"Study ID: {study_id}")
    # print(f"Reference labels (valor=1): {ref_labels}")
    
    # Para cada linha, verificar se tem intersecção com ref_labels
    def has_overlap(row):
        row_labels = set(row[row == 1].index)
        return bool(row_labels & ref_labels)
    
    mask = df_resp[binary_cols].apply(has_overlap, axis=1)
    matches = df_resp[mask]

    # print(f"Matches found: {len(matches)}")
    return matches

def binary_jaccard_overlap(study_id, id_list, df_resp, id_col='study_id'):
    """
    Calcula, para cada id em id_list, o percentual:
        (# colunas com 1 em comum) / (# colunas com 1 em pelo menos um dos dois)
    usando apenas as colunas binárias (todas exceto id_col).
    
    Retorna um DataFrame com [id_col, jaccard_overlap].
    """
    ref = df_resp.loc[df_resp[id_col] == study_id].copy()
    if ref.empty:
        raise ValueError(f"ID {study_id} not found in dataset.")
    
    ref.fillna(0, inplace=True)
    
    binary_cols = [col for col in df_resp.columns if col not in ['subject_id', 'study_id']]

    # Converte para vetores numpy de 0/1
    ref_vals = ref[binary_cols].to_numpy().astype(int)
    ref_ones = (ref_vals == 1)

    results = []
    for _id in id_list:
        # Pega a linha do id atual
        row = df_resp.loc[df_resp[id_col] == _id].copy()
        if row.empty:
            results.append((_id, np.nan))
            continue
        row.fillna(0, inplace=True)

        row_vals = row[binary_cols].to_numpy().astype(int)
        row_ones = (row_vals == 1)

        # Interseção: 1 nas duas
        inter = (ref_ones & row_ones).sum()
        # União: 1 em pelo menos uma
        union = (ref_ones | row_ones).sum()

        if union == 0:
            # Nenhuma coluna com 1 em nenhum dos dois
            jaccard = np.nan  # ou 1.0, se você quiser considerar "idênticos vazios"
        else:
            jaccard = inter / union

        results.append((_id, jaccard))

    return pd.DataFrame(results, columns=[id_col, "jaccard_overlap"])
