import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# --- Konfiguráció ---
DATA_DIR = 'data'
BETEG_DIR = os.path.join(DATA_DIR, 'beteg')
EGESZSEGES_DIR = os.path.join(DATA_DIR, 'egeszseges')
SFREQ = 256          # mintavételezési frekvencia
EPOCH_SEC = 2        # epoch hossza másodpercben
MAX_EPOCHS = 5       # maximum epoch száma felvételenként
EPOCH_LEN = EPOCH_SEC * SFREQ  # 512 sor

EEG_COLS = ['alpha1', 'alpha2', 'beta1', 'beta2', 'theta', 'gamma1', 'gamma2']


def load_epochs_from_dir(directory):
    """
    Beolvassa az összes CSV fájlt egy mappából,
    epochokra bontja őket, és visszaadja a csatornánkénti epochokat.
    Visszatérési érték: lista of dict {csatorna: epochok tömbje}
    """
    all_epochs = []

    for file in os.listdir(directory):
        if not file.endswith('.csv'):
            continue

        df = pd.read_csv(os.path.join(directory, file), sep=';')
        cols = [c for c in EEG_COLS if c in df.columns]
        data = df[cols].values
        n_samples = data.shape[0]
        n_epochs = min(n_samples // EPOCH_LEN, MAX_EPOCHS)

        # epochok kinyerése
        file_epochs = {col: [] for col in cols}
        for i in range(n_epochs):
            start = i * EPOCH_LEN
            end = start + EPOCH_LEN
            for j, col in enumerate(cols):
                file_epochs[col].append(data[start:end, j])

        all_epochs.append(file_epochs)

    return all_epochs


def compute_contrast(epochs_list):
    """
    Epochonként kiszámítja a min és max értékek közötti különbséget
    minden csatornára, minden felvételre.
    Visszatérési érték: dict {csatorna: lista of kontrast értékek epochonként}
    """
    contrast = {col: [[] for _ in range(MAX_EPOCHS)] for col in EEG_COLS}

    for file_epochs in epochs_list:
        for col in file_epochs:
            for epoch_idx, epoch in enumerate(file_epochs[col]):
                c = np.max(epoch) - np.min(epoch)
                contrast[col][epoch_idx].append(c)

    return contrast


def compute_normality(epochs_list):
    """
    Shapiro-Wilk normalitás tesztet végez minden csatornára és epochra.
    Visszatérési érték: dict {csatorna: lista of (statisztika, p-érték) epochonként}
    """
    normality = {col: [[] for _ in range(MAX_EPOCHS)] for col in EEG_COLS}

    for file_epochs in epochs_list:
        for col in file_epochs:
            for epoch_idx, epoch in enumerate(file_epochs[col]):
                if len(epoch) >= 3:  # Shapiro-Wilk legalább 3 mintát igényel
                    stat, p = stats.shapiro(epoch)
                    normality[col][epoch_idx].append((stat, p))

    return normality


def plot_all(beteg_contrast, egeszseges_contrast, beteg_normality, egeszseges_normality):
    """
    Egy figurában jeleníti meg a kontraszt és normalitás eredményeket.
    Felső sor: min-max kontraszt, alsó sor: normalitás p-értékek.
    """
    n_cols = len(EEG_COLS)
    fig, axes = plt.subplots(2, n_cols, figsize=(20, 10))
    fig.suptitle('Epoch analízis – beteg vs egészséges', fontsize=14)

    for idx, col in enumerate(EEG_COLS):
        epoch_labels = [f'E{i+1}' for i in range(MAX_EPOCHS)]
        x = np.arange(MAX_EPOCHS)
        width = 0.35

        # --- Felső sor: kontraszt ---
        ax_top = axes[0][idx]

        beteg_means = [np.mean(beteg_contrast[col][i]) if beteg_contrast[col][i] else 0
                       for i in range(MAX_EPOCHS)]
        egesz_means = [np.mean(egeszseges_contrast[col][i]) if egeszseges_contrast[col][i] else 0
                       for i in range(MAX_EPOCHS)]

        ax_top.bar(x - width/2, beteg_means, width, label='Beteg', color='red', alpha=0.7)
        ax_top.bar(x + width/2, egesz_means, width, label='Egészséges', color='green', alpha=0.7)
        ax_top.set_title(col)
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(epoch_labels)
        ax_top.set_ylabel('Átlagos kontraszt')
        ax_top.legend(fontsize=7)

        # --- Alsó sor: normalitás p-értékek ---
        ax_bot = axes[1][idx]

        beteg_pvals = [
            np.mean([p for _, p in beteg_normality[col][i]]) if beteg_normality[col][i] else 0
            for i in range(MAX_EPOCHS)
        ]
        egesz_pvals = [
            np.mean([p for _, p in egeszseges_normality[col][i]]) if egeszseges_normality[col][i] else 0
            for i in range(MAX_EPOCHS)
        ]

        ax_bot.bar(x - width/2, beteg_pvals, width, label='Beteg', color='red', alpha=0.7)
        ax_bot.bar(x + width/2, egesz_pvals, width, label='Egészséges', color='green', alpha=0.7)

        # 0.05-ös határvonal – ez alatt nem normális az eloszlás
        ax_bot.axhline(y=0.05, color='black', linestyle='--', linewidth=0.8, label='p=0.05')
        ax_bot.set_title(col)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(epoch_labels)
        ax_bot.set_ylabel('Átlagos p-érték')
        ax_bot.legend(fontsize=7)

    # felső és alsó sor feliratai
    axes[0][0].annotate('Min-Max kontraszt', xy=(0, 0.5), xytext=(-60, 0),
                        xycoords='axes fraction', textcoords='offset points',
                        fontsize=11, ha='right', va='center', rotation=90)
    axes[1][0].annotate('Normalitás (p-érték)', xy=(0, 0.5), xytext=(-60, 0),
                        xycoords='axes fraction', textcoords='offset points',
                        fontsize=11, ha='right', va='center', rotation=90)

    plt.tight_layout()
    plt.show()


def print_summary(beteg_contrast, egeszseges_contrast, beteg_normality, egeszseges_normality):
    """
    Szöveges összefoglaló a kontrast és normalitás eredményekről.
    """
    print("=" * 60)
    print("EPOCH ANALÍZIS ÖSSZEFOGLALÓ")
    print("=" * 60)

    for col in EEG_COLS:
        print(f"\n--- {col} ---")
        for i in range(MAX_EPOCHS):
            b_contrast = np.mean(beteg_contrast[col][i]) if beteg_contrast[col][i] else 0
            e_contrast = np.mean(egeszseges_contrast[col][i]) if egeszseges_contrast[col][i] else 0

            b_pval = np.mean([p for _, p in beteg_normality[col][i]]) if beteg_normality[col][i] else 0
            e_pval = np.mean([p for _, p in egeszseges_normality[col][i]]) if egeszseges_normality[col][i] else 0

            print(f"  Epoch {i+1}:")
            print(f"    Kontraszt  – beteg: {b_contrast:.2f}, egészséges: {e_contrast:.2f}, különbség: {abs(b_contrast - e_contrast):.2f}")
            print(f"    Normalitás – beteg p: {b_pval:.4f}, egészséges p: {e_pval:.4f}")


if __name__ == "__main__":
    print("Adatok betöltése...")
    beteg_epochs = load_epochs_from_dir(BETEG_DIR)
    egeszseges_epochs = load_epochs_from_dir(EGESZSEGES_DIR)

    print("Kontraszt számítása...")
    beteg_contrast = compute_contrast(beteg_epochs)
    egeszseges_contrast = compute_contrast(egeszseges_epochs)

    print("Normalitás teszt futtatása...")
    beteg_normality = compute_normality(beteg_epochs)
    egeszseges_normality = compute_normality(egeszseges_epochs)

    print("Eredmények kiírása...")
    print_summary(beteg_contrast, egeszseges_contrast, beteg_normality, egeszseges_normality)

    print("Grafikonok megjelenítése...")
    plot_all(beteg_contrast, egeszseges_contrast, beteg_normality, egeszseges_normality)