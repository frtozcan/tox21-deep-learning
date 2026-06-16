"""
ADIM 1 — VERİ İNDİRME
======================
Tox21 ve diğer toksisite veri setlerini indirir.
Çalıştırma: python step1_download_data.py
"""

import requests
import pandas as pd
import io
import os

os.makedirs("data", exist_ok=True)

DATASETS = {
    "tox21":   "https://raw.githubusercontent.com/SY575/CMPNN/master/data/tox21.csv",
    "clintox": "https://raw.githubusercontent.com/tencent-ailab/grover/master/exampledata/finetune/clintox.csv",
    "bbbp":    "https://raw.githubusercontent.com/tencent-ailab/grover/master/exampledata/finetune/bbbp.csv",
    "bace":    "https://raw.githubusercontent.com/tencent-ailab/grover/master/exampledata/finetune/bace.csv",
    "sider":   "https://raw.githubusercontent.com/tencent-ailab/grover/master/exampledata/finetune/sider.csv",
}

print("Veri setleri indiriliyor...\n")
for name, url in DATASETS.items():
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 1000:
            df = pd.read_csv(io.StringIO(r.text))
            df.to_csv(f"data/{name}.csv", index=False)
            print(f"✅ {name:<10}: {len(df):5d} bileşik")
        else:
            print(f"❌ {name}: HTTP {r.status_code}")
    except Exception as e:
        print(f"❌ {name}: {e}")

# Tox21 detay
print("\nTox21 Görev İstatistikleri:")
df = pd.read_csv("data/tox21.csv")
tasks = [c for c in df.columns if c != 'smiles']
for t in tasks:
    v = df[t].dropna()
    pos = int((v==1).sum()); neg = int((v==0).sum())
    print(f"  {t:<20}: {pos:4d}+ {neg:5d}- ({pos/(pos+neg)*100:.1f}% pozitif)")
print(f"\nToplam: {len(df)} bileşik, {len(tasks)} görev")
