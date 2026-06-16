"""
ADIM 2 — ÖZELLİK ÇIKARIMI (FEATURE EXTRACTION)
================================================
Her molekülden 3 farklı temsil oluşturur:
  1. RDKit Molecular Descriptors (200 sayısal özellik)
  2. SMILES Token dizisi (Transformer için)
  3. Moleküler Graf (GNN için)

Çalıştırma: python step2_feature_extraction.py
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs
import os, time

os.makedirs("features", exist_ok=True)
N_DESC  = 200    # Kaç descriptor kullanılacak
MAX_LEN = 100    # SMILES Transformer için max uzunluk

# ─── Veriyi yükle ─────────────────────────────────────
print("Veri yükleniyor...")
df = pd.read_csv("data/tox21.csv")
smiles_list = df['smiles'].tolist()
print(f"  {len(smiles_list)} bileşik")

# ─── 1. MOLECULAR DESCRIPTORS ─────────────────────────
# RDKit'in hesapladığı 200 fizikokimyasal özellik
# MolLogP (yağda çözünürlük), MolWt (molekül ağırlığı),
# NumHDonors (hidrojen verici sayısı), TPSA (polar yüzey alanı) vb.
print(f"\nDescriptors hesaplanıyor ({N_DESC} özellik × {len(smiles_list)} mol)...")
t0 = time.time()
desc_names = [n for n, _ in Descriptors.descList[:N_DESC]]
rows = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        vals = []
        for _, fn in Descriptors.descList[:N_DESC]:
            try:
                v = float(fn(mol))
                vals.append(0.0 if (np.isnan(v) or np.isinf(v)) else v)
            except:
                vals.append(0.0)
    else:
        vals = [0.0] * N_DESC
    rows.append(vals)

X_desc = np.clip(np.array(rows, dtype=np.float32), -1e6, 1e6)
np.save("features/X_descriptors.npy", X_desc)
pd.DataFrame({"descriptor": desc_names}).to_csv("features/descriptor_names.csv", index=False)
print(f"  ✅ Descriptor matrisi: {X_desc.shape}  ({time.time()-t0:.0f}s)")
print(f"  Örnek özellikler: {desc_names[:5]}")

# ─── 2. SMILES TOKEN DİZİSİ (Transformer için) ────────
# Her SMILES karakteri bir "token" — "C", "c", "O", "N", "(", ")", "=" vb.
# Tıpkı NLP'de kelimelerin token'a çevrilmesi gibi
print("\nSMILES tokenizasyonu...")
vocab  = ['<PAD>', '<UNK>'] + sorted(set(''.join(smiles_list)))
c2i    = {c: i for i, c in enumerate(vocab)}

def encode_smiles(s, max_len=MAX_LEN):
    ids = [c2i.get(c, 1) for c in s[:max_len]]
    return ids + [0] * (max_len - len(ids))  # padding

X_tokens = np.array([encode_smiles(s) for s in smiles_list], dtype=np.int64)
np.save("features/X_tokens.npy", X_tokens)
pd.DataFrame({"token": list(c2i.keys()), "index": list(c2i.values())}).to_csv(
    "features/vocab.csv", index=False)
print(f"  ✅ Token matrisi: {X_tokens.shape}")
print(f"  Vocab boyutu: {len(vocab)} karakter")
print(f"  SMILES örneği: '{smiles_list[0][:20]}...'")
print(f"  → Token: {X_tokens[0][:10]}...")

# ─── 3. MORGAN FINGERPRINT (alternatif özellik) ────────
# Dairesel parmak izi: atom çevresini 1024 bitlik binary vektöre çevirir
print("\nMorgan Fingerprint hesaplanıyor...")
fps = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        arr = np.zeros(1024, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
    else:
        arr = np.zeros(1024, dtype=np.float32)
    fps.append(arr)
X_fp = np.array(fps)
np.save("features/X_morgan_fp.npy", X_fp)
print(f"  ✅ Morgan FP matrisi: {X_fp.shape}")

print("\n" + "="*50)
print("ÖZET — Kaydedilen dosyalar (features/):")
print("  X_descriptors.npy  → MLP ve 1D-CNN için")
print("  X_tokens.npy       → SMILES Transformer için")
print("  X_morgan_fp.npy    → alternatif özellik seti")
print("  descriptor_names.csv  → özellik isimleri")
print("  vocab.csv          → SMILES token sözlüğü")
