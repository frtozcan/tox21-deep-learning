
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
import torch.nn.functional as F

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ─── Çıktı klasörü ────────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)

# ─── Renk paleti (makale tutarlılığı için sabit) ──────────────────────────────
PALETTE = {
    "MLP":         "#2E75B6",   # koyu mavi
    "1D-CNN":      "#7030A0",   # mor
    "GNN":         "#C55A11",   # turuncu
    "Transformer": "#375623",   # koyu yeşil
}
MODELS = list(PALETTE.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI: Model sınıfları (step3_models_v2.py'den kopyalandı)
#  Kaydedilmiş .pt ağırlıklarını yüklemek için gerekli
# ══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, in_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64,  7, padding=3), nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x): return self.fc(self.conv(x)).squeeze(-1)


class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(9,   128, heads=4, concat=False, dropout=0.2)
        self.conv2 = GATConv(128, 128, heads=4, concat=False, dropout=0.2)
        self.conv3 = GATConv(128, 128, heads=2, concat=False, dropout=0.2)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(128), nn.LayerNorm(128), nn.LayerNorm(128)
        self.fc = nn.Sequential(nn.Linear(256,256), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(256,64),  nn.ReLU(), nn.Linear(64,1))
    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        x = F.elu(self.ln1(self.conv1(x, ei)))
        x = F.elu(self.ln2(self.conv2(x, ei)))
        x = F.elu(self.ln3(self.conv3(x, ei)))
        return self.fc(torch.cat([global_mean_pool(x, b), global_max_pool(x, b)], dim=-1)).squeeze(-1)


class SMILESTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(100, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                         dropout=0.1, batch_first=True, norm_first=True)
        self.transformer  = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.classifier   = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model,32),
                                          nn.GELU(), nn.Dropout(0.2), nn.Linear(32,1))
    def forward(self, x):
        pos  = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h    = self.token_emb(x) + self.pos_emb(pos)
        mask = (x == 0)
        h    = self.transformer(h, src_key_padding_mask=mask)
        return self.classifier(h[:, 0, :]).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
#  VERİ YÜKLEME
# ══════════════════════════════════════════════════════════════════════════════

def load_results():
    """results/model_comparison.xlsx dosyasını yükle."""
    path = "results/model_comparison.xlsx"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} bulunamadı.\n"
            "Önce step3_models_v2.py --task all çalıştırın."
        )
    df = pd.read_excel(path, sheet_name="Tüm Metrikler")
    # Türkçe sütun adlarını İngilizce'ye çevir
    df = df.rename(columns={
        'Görev':   'Task',
        'Süre (s)':'time_s',
        'ACC (%)': 'ACC',
    })
    print(f"  Yüklendi: {path}  ({len(df)} satır)")
    print(f"  Sütunlar: {df.columns.tolist()}")
    return df


def load_test_predictions(df_full, X_desc, X_tok, vocab_size):
    """
    Her görev ve model için test seti tahminlerini yeniden üretir.
    ROC eğrisi ve confusion matrix için y_true ve y_prob gerekli.
    Kaydedilmiş model ağırlıklarını (models/*.pt) yükler.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import Dataset, DataLoader
    from torch_geometric.loader import DataLoader as GeoLoader
    from torch_geometric.data import Data
    from rdkit import Chem

    TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
             'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']
    DEVICE = torch.device('cpu')

    class TabDS(Dataset):
        def __init__(self, X, y, dtype=torch.float32):
            self.X = torch.tensor(X, dtype=dtype)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i]

    def mol_to_graph(smiles, label):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        feats = [[a.GetAtomicNum()/118, a.GetDegree()/10, int(a.GetIsAromatic()),
                  a.GetTotalNumHs()/4, a.GetMass()/200, int(a.IsInRing()),
                  a.GetTotalValence()/8, a.GetFormalCharge()/5,
                  a.GetNumRadicalElectrons()/4] for a in mol.GetAtoms()]
        x = torch.tensor(feats, dtype=torch.float)
        el = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            el += [[i,j],[j,i]]
        ei = torch.tensor(el, dtype=torch.long).t().contiguous() if el else torch.zeros((2,0), dtype=torch.long)
        return Data(x=x, edge_index=ei, y=torch.tensor([float(label)]))

    predictions = {}
    all_smiles  = df_full['smiles'].tolist()

    for task in TASKS:
        # Model dosyaları var mı kontrol et
        missing = [m for m in MODELS
                   if not os.path.exists(f"models/{m.lower().replace('-','').replace('1d','1d')}_{task}.pt") and
                   not os.path.exists(f"models/{'mlp' if m=='MLP' else 'cnn' if m=='1D-CNN' else 'gnn' if m=='GNN' else 'tfm'}_{task}.pt")]

        df_t = df_full[['smiles', task]].dropna()
        df_t = df_t[df_t[task].isin([0.0, 1.0])].drop_duplicates('smiles').reset_index(drop=True)
        df_t = df_t[df_t['smiles'].apply(lambda s: Chem.MolFromSmiles(s) is not None)].reset_index(drop=True)

        smiles_list = df_t['smiles'].tolist()
        labels      = df_t[task].values
        fi          = [all_smiles.index(s) if s in all_smiles else 0 for s in smiles_list]
        fi          = np.array(fi)

        idx = np.arange(len(df_t))
        idx_tv, idx_te = train_test_split(idx, test_size=0.15, stratify=labels, random_state=42)
        idx_tr, _      = train_test_split(idx_tv, test_size=0.176, stratify=labels[idx_tv], random_state=42)
        yte = labels[idx_te]

        # Descriptor normalize
        sc  = StandardScaler()
        Xtr = np.nan_to_num(sc.fit_transform(X_desc[fi][idx_tr]))
        Xte = np.nan_to_num(sc.transform(X_desc[fi][idx_te]))
        Xt  = X_tok[fi]

        task_preds = {}

        # ── MLP ──────────────────────────────────────────────────────────────
        pt = f"models/mlp_{task}.pt"
        if os.path.exists(pt):
            model = MLP().to(DEVICE); model.load_state_dict(torch.load(pt, map_location=DEVICE))
            model.eval()
            dl = DataLoader(TabDS(Xte, yte), 64)
            probs = []
            with torch.no_grad():
                for X, _ in dl:
                    probs.extend(torch.sigmoid(model(X)).numpy())
            task_preds['MLP'] = (yte, np.array(probs))

        # ── 1D-CNN ───────────────────────────────────────────────────────────
        pt = f"models/cnn_{task}.pt"
        if os.path.exists(pt):
            class CDS(Dataset):
                def __init__(self, X, y):
                    self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
                    self.y = torch.tensor(y, dtype=torch.float32)
                def __len__(self): return len(self.y)
                def __getitem__(self, i): return self.X[i], self.y[i]
            model = CNN1D().to(DEVICE); model.load_state_dict(torch.load(pt, map_location=DEVICE))
            model.eval()
            dl = DataLoader(CDS(Xte, yte), 64)
            probs = []
            with torch.no_grad():
                for X, _ in dl:
                    probs.extend(torch.sigmoid(model(X)).numpy())
            task_preds['1D-CNN'] = (yte, np.array(probs))

        # ── GNN ──────────────────────────────────────────────────────────────
        pt = f"models/gnn_{task}.pt"
        if os.path.exists(pt):
            graphs = [mol_to_graph(smiles_list[i], yte[j]) for j, i in enumerate(idx_te)]
            graphs = [g for g in graphs if g]
            model  = GNN().to(DEVICE); model.load_state_dict(torch.load(pt, map_location=DEVICE))
            model.eval()
            dl = GeoLoader(graphs, 64)
            probs, trues = [], []
            with torch.no_grad():
                for batch in dl:
                    batch = batch.to(DEVICE)
                    probs.extend(torch.sigmoid(model(batch)).numpy())
                    trues.extend(batch.y.numpy().flatten())
            task_preds['GNN'] = (np.array(trues), np.array(probs))

        # ── Transformer ───────────────────────────────────────────────────────
        pt = f"models/tfm_{task}.pt"
        if os.path.exists(pt):
            Xte_t = Xt[idx_te]
            model  = SMILESTransformer(vocab_size).to(DEVICE)
            model.load_state_dict(torch.load(pt, map_location=DEVICE))
            model.eval()
            dl = DataLoader(TabDS(Xte_t, yte, dtype=torch.long), 64)
            probs = []
            with torch.no_grad():
                for X, _ in dl:
                    probs.extend(torch.sigmoid(model(X)).numpy())
            task_preds['Transformer'] = (yte, np.array(probs))

        predictions[task] = task_preds
        print(f"    {task}: {list(task_preds.keys())} ✓")

    return predictions


# ══════════════════════════════════════════════════════════════════════════════
#  FİGÜR 1 — ROC EĞRİLERİ (12 görev × 4 model)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_roc_curves(predictions):
    """
    Makaledeki Şekil 2 benzeri — her görev için 4 modelin ROC eğrisi.
    3×4 subplot düzeni.
    """
    TASKS = list(predictions.keys())
    n     = len(TASKS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4.5))
    fig.suptitle("ROC Curves — All Tasks and Models", fontsize=16, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for ti, task in enumerate(TASKS):
        ax = axes[ti]
        task_preds = predictions.get(task, {})

        for mname in MODELS:
            if mname not in task_preds:
                continue
            y_true, y_prob = task_preds[mname]
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=PALETTE[mname], lw=2,
                    label=f"{mname} (AUC={roc_auc:.3f})")

        ax.plot([0,1],[0,1],"k--", lw=1, alpha=0.5)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate",  fontsize=9)
        ax.set_title(task, fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=7.5)
        ax.grid(True, alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)

    # Fazla subplot'ları gizle
    for j in range(ti+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = "figures/fig1_roc_curves.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FİGÜR 2 — CONFUSION MATRIX ISIL HARİTASI
# ══════════════════════════════════════════════════════════════════════════════

def fig2_confusion_matrices(predictions):
    """
    4 modelin confusion matrix'lerini ızgara düzeninde gösterir.
    Her satır bir model, her sütun bir görev.
    """
    TASKS = list(predictions.keys())
    n_tasks  = len(TASKS)
    n_models = len(MODELS)

    fig, axes = plt.subplots(n_models, n_tasks,
                             figsize=(n_tasks * 2.2, n_models * 2.2))
    fig.suptitle("Confusion Matrices — All Models × All Tasks",
                 fontsize=14, fontweight="bold", y=1.01)

    for mi, mname in enumerate(MODELS):
        for ti, task in enumerate(TASKS):
            ax = axes[mi][ti]
            task_preds = predictions.get(task, {})

            if mname in task_preds:
                y_true, y_prob = task_preds[mname]
                y_pred = (y_prob > 0.5).astype(int)
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

                sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                            cmap=sns.light_palette(PALETTE[mname], as_cmap=True),
                            cbar=False, linewidths=0.5,
                            annot_kws={"size": 8})
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                ax.set_xticks([]); ax.set_yticks([])

            # Etiketler
            if mi == 0:
                ax.set_title(task, fontsize=8, fontweight="bold", rotation=25,
                             ha="left", pad=2)
            if ti == 0:
                ax.set_ylabel(mname, fontsize=9, fontweight="bold",
                              color=PALETTE[mname])
            else:
                ax.set_ylabel("")

            ax.set_xlabel("")
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = "figures/fig2_confusion_matrices.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FİGÜR 3 — MODEL KARŞILAŞTIRMA ÇUBUK GRAFİĞİ
# ══════════════════════════════════════════════════════════════════════════════

def fig3_bar_comparison(df):
    """
    Makaledeki Tablo 3/4 karşılığı görsel.
    AUC, F1 ve MCC için 12 görevde ortalama + std çubuk grafiği.
    """
    metrics = ["AUC", "F1", "MCC"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
    fig.suptitle("Model Performance Comparison Across 12 Tox21 Tasks",
                 fontsize=14, fontweight="bold")

    for ai, metric in enumerate(metrics):
        ax = axes[ai]
        model_stats = []
        for mname in MODELS:
            vals = df[df["Model"] == mname][metric].dropna().values
            model_stats.append({
                "model": mname,
                "mean":  vals.mean() if len(vals) else 0,
                "std":   vals.std()  if len(vals) else 0,
            })

        means = [s["mean"] for s in model_stats]
        stds  = [s["std"]  for s in model_stats]
        colors = [PALETTE[s["model"]] for s in model_stats]
        x = np.arange(len(MODELS))

        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=colors, width=0.55, edgecolor="white",
                      linewidth=0.8, error_kw=dict(lw=1.5, capsize=4))

        # Değer etiketleri
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + std + 0.005,
                    f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=10)
        ax.set_title(metric, fontsize=13, fontweight="bold", pad=8)
        ax.set_ylabel(f"Mean {metric} ± SD", fontsize=10)
        ax.set_ylim(0, min(1.15, max(means) + max(stds) + 0.12))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    path = "figures/fig3_bar_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FİGÜR 4 — AUC ISIL HARİTASI (görev × model matrisi)
# ══════════════════════════════════════════════════════════════════════════════

def fig4_auc_heatmap(df):
    """
    Makalede olmayan ama güçlü bir figür:
    Satırlar = görevler, Sütunlar = modeller, Değerler = AUC.
    Her hücreyi renklendirerek güçlü/zayıf görev-model kombinasyonları görünür.
    """
    TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
             'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

    matrix = []
    for task in TASKS:
        row = []
        for mname in MODELS:
            val = df[(df["Task"] == task) & (df["Model"] == mname)]["AUC"]
            row.append(float(val.values[0]) if len(val) else np.nan)
        matrix.append(row)

    mat = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.65, vmax=1.0, aspect="auto")

    # Eksen etiketleri
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels(TASKS, fontsize=11)

    # Hücre değerleri
    for i in range(len(TASKS)):
        for j in range(len(MODELS)):
            val = mat[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.78 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=9, color="gray")

    # Renk çubuğu
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("AUC", fontsize=11)

    # En iyi modeli her satırda çerçevele
    for i in range(len(TASKS)):
        row = mat[i]
        if not np.all(np.isnan(row)):
            best_j = np.nanargmax(row)
            rect = plt.Rectangle((best_j - 0.47, i - 0.47), 0.94, 0.94,
                                  fill=False, edgecolor="#1F3864", lw=2.5)
            ax.add_patch(rect)

    ax.set_title("AUC Heatmap — Model × Task\n(box = best model per task)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    plt.tight_layout()
    path = "figures/fig4_auc_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FİGÜR 5 — RADAR CHART (çok metrikli model karşılaştırması)
# ══════════════════════════════════════════════════════════════════════════════

def fig5_radar_chart(df):
    """
    5 metrikte (AUC, F1, MCC, SE, SP) 4 modelin radar grafiği.
    12 görev üzerinden ortalama değerler kullanılır.
    """
    metrics = ["AUC", "F1", "MCC", "SE", "SP"]
    N = len(metrics)

    # Her metriği [0,1] arasında normalize et (MCC -1/+1 → 0/1)
    def normalize(vals, metric):
        if metric == "MCC":
            return [(v + 1) / 2 for v in vals]  # -1..+1 → 0..1
        return vals

    # Açılar
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # kapat

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for mname in MODELS:
        mdf  = df[df["Model"] == mname]
        vals = []
        for m in metrics:
            v = mdf[m].dropna().mean()
            vals.append(v if not np.isnan(v) else 0)
        vals_n = [normalize([v], m)[0] for v, m in zip(vals, metrics)]
        vals_n += vals_n[:1]

        ax.plot(angles, vals_n, "o-", lw=2, color=PALETTE[mname], label=mname)
        ax.fill(angles, vals_n, alpha=0.08, color=PALETTE[mname])

    # Eksen etiketleri
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["AUC", "F1", "MCC\n(norm.)", "SE", "SP"],
                       fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8, color="gray")
    ax.grid(color="gray", alpha=0.3)
    ax.spines["polar"].set_alpha(0.3)

    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.15),
                       fontsize=11, framealpha=0.9)
    for lh in legend.legend_handles:
        lh.set_linewidth(3)

    ax.set_title("Multi-metric Radar Chart\n(12-task averages)",
                 fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    path = "figures/fig5_radar_chart.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FİGÜR 6 — MAKALETABLOsu (Tablo 3/4 benzeri, görsel tablo)
# ══════════════════════════════════════════════════════════════════════════════

def fig6_result_table(df):
    """
    Erturan et al. Tablo 3 ve 4 stilinde — her model için tüm metriklerin
    özet tablosu. Matplotlib ile çizilir, makaleye direkt yapıştırılabilir.
    """
    TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
             'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']
    cols_display = ["ACC", "AUC", "F1", "MCC", "SE", "SP"]

    for mname in MODELS:
        mdf = df[df["Model"] == mname].set_index("Task")

        # Tablo verisini topla
        rows = []
        for task in TASKS:
            if task in mdf.index:
                r = mdf.loc[task]
                rows.append([
                    task,
                    f"{r.get('ACC', float('nan')):.2f}",
                    f"{r.get('AUC', float('nan')):.4f}",
                    f"{r.get('F1',  float('nan')):.4f}",
                    f"{r.get('MCC', float('nan')):.4f}",
                    f"{r.get('SE',  float('nan')):.4f}",
                    f"{r.get('SP',  float('nan')):.4f}",
                ])
            else:
                rows.append([task] + ["N/A"] * 6)

        # Ortalama satırı
        means = []
        for col in cols_display:
            vals = mdf[col].dropna().values if col in mdf.columns else []
            means.append(f"{np.mean(vals):.4f}" if len(vals) else "N/A")
        rows.append(["Mean"] + means)

        fig, ax = plt.subplots(figsize=(13, len(rows) * 0.48 + 1.5))
        ax.axis("off")

        header = ["Task", "ACC (%)", "AUC", "F1", "MCC", "SE", "SP"]
        col_widths = [0.22, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11]

        # Başlık satırı
        for ci, (h, w) in enumerate(zip(header, col_widths)):
            x = sum(col_widths[:ci]) + w / 2
            ax.add_patch(FancyBboxPatch((sum(col_widths[:ci]), len(rows) / (len(rows)+1)),
                                        w, 1/(len(rows)+1),
                                        boxstyle="square,pad=0",
                                        facecolor=PALETTE[mname], edgecolor="white", lw=0.5,
                                        transform=ax.transAxes))
            ax.text(x, (len(rows) + 0.5) / (len(rows)+1), h,
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white", transform=ax.transAxes)

        # Veri satırları
        for ri, row in enumerate(rows):
            is_mean = (row[0] == "Mean")
            bg = "#F0F0F0" if ri % 2 == 0 else "white"
            if is_mean: bg = "#E2EFDA"

            for ci, (val, w) in enumerate(zip(row, col_widths)):
                x_pos = sum(col_widths[:ci])
                y_pos = (len(rows) - ri - 1) / (len(rows)+1)
                h_pos = 1 / (len(rows)+1)

                ax.add_patch(FancyBboxPatch((x_pos, y_pos), w, h_pos,
                                            boxstyle="square,pad=0",
                                            facecolor=bg, edgecolor="#CCCCCC", lw=0.3,
                                            transform=ax.transAxes))

                bold = is_mean or ci == 0
                ax.text(x_pos + w/2, y_pos + h_pos/2, val,
                        ha="center", va="center",
                        fontsize=8.5 if not is_mean else 9,
                        fontweight="bold" if bold else "normal",
                        transform=ax.transAxes)

        ax.set_title(f"Results — {mname}  (Test Set, 12 Tox21 Tasks)",
                     fontsize=12, fontweight="bold", pad=10, color=PALETTE[mname])

        path = f"figures/fig6_table_{mname.replace('-','').replace(' ','_').lower()}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Kaydedildi: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FİGÜR 7 — SE / SP SCATTER (Duyarlılık vs Özgüllük)
# ══════════════════════════════════════════════════════════════════════════════

def fig7_se_sp_scatter(df):
    """
    Her görev için dört modelin SE (Sensitivity) ve SP (Specificity) değerlerini
    scatter plot olarak gösterir. İdeal nokta sağ üst köşe (SE=1, SP=1).
    """
    TASKS = df["Task"].unique().tolist()

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle("Sensitivity vs Specificity per Task",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for ti, task in enumerate(TASKS):
        ax = axes[ti]
        tdf = df[df["Task"] == task]

        for mname in MODELS:
            row = tdf[tdf["Model"] == mname]
            if len(row) == 0: continue
            se = float(row["SE"].values[0]) if not pd.isna(row["SE"].values[0]) else None
            sp = float(row["SP"].values[0]) if not pd.isna(row["SP"].values[0]) else None
            if se is None or sp is None: continue

            ax.scatter(sp, se, color=PALETTE[mname], s=120, zorder=5,
                       edgecolors="white", linewidths=1.5,
                       label=mname)
            ax.annotate(mname[:3], (sp, se),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color=PALETTE[mname])

        # İdeal nokta
        ax.scatter(1.0, 1.0, marker="*", s=150, color="gold",
                   edgecolors="gray", lw=0.8, zorder=6, label="Ideal")
        ax.set_xlim(0.4, 1.05); ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Specificity (SP)", fontsize=9)
        ax.set_ylabel("Sensitivity (SE)", fontsize=9)
        ax.set_title(task, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)

    # Ortak legend
    handles = [mpatches.Patch(color=PALETTE[m], label=m) for m in MODELS]
    handles.append(plt.scatter([], [], marker="*", s=100,
                               color="gold", edgecolors="gray", label="Ideal"))
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    for j in range(ti+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = "figures/fig7_se_sp_scatter.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  ANA AKIŞ
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  ADIM 5 — MAKALEYİ KALİTESİNDE FİGÜRLER")
    print("=" * 60)

    # ── Sonuç tablosunu yükle ─────────────────────────────────────────────────
    print("\n[1/3] Sonuç tablosu yükleniyor...")
    df = load_results()

    # ── Tahminleri yeniden üret (ROC + confusion matrix için) ─────────────────
    print("\n[2/3] Test seti tahminleri yeniden üretiliyor...")
    try:
        import pandas as pd
        df_full = pd.read_csv("data/tox21.csv")
        X_desc  = np.load("features/X_descriptors.npy")
        X_tok   = np.load("features/X_tokens.npy")
        all_chars = sorted(set(''.join(df_full['smiles'].tolist())))
        vocab     = ['<PAD>', '<UNK>'] + all_chars
        predictions = load_test_predictions(df_full, X_desc, X_tok, len(vocab))
        has_predictions = True
    except Exception as e:
        print(f"  ⚠ Tahminler yüklenemedi: {e}")
        print("  ROC ve confusion matrix figürleri atlanacak.")
        has_predictions = False

    # ── Figürleri üret ────────────────────────────────────────────────────────
    print("\n[3/3] Figürler oluşturuluyor...")

    if has_predictions:
        print("  → Fig 1: ROC eğrileri")
        fig1_roc_curves(predictions)

        print("  → Fig 2: Confusion matrix ısıl haritası")
        fig2_confusion_matrices(predictions)

    print("  → Fig 3: Model karşılaştırma çubuk grafiği")
    fig3_bar_comparison(df)

    print("  → Fig 4: AUC ısıl harita matrisi")
    fig4_auc_heatmap(df)

    print("  → Fig 5: Radar chart")
    fig5_radar_chart(df)

    print("  → Fig 6: Makale tarzı sonuç tabloları")
    fig6_result_table(df)

    print("  → Fig 7: Sensitivity vs Specificity scatter")
    fig7_se_sp_scatter(df)

    # ── Özet ──────────────────────────────────────────────────────────────────
    saved = [f for f in os.listdir("figures") if f.endswith(".png")]
    print(f"\n{'='*60}")
    print(f"  ✅ {len(saved)} figür 'figures/' klasörüne kaydedildi:")
    for f in sorted(saved):
        print(f"     {f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
