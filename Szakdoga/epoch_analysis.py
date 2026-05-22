import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

#Konfiguráció
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
    epochokra bontja őket, és visszaadja a csatornánkénti epochokat
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
    minden csatornára, minden felvételre
    Visszatérési érték: dict {csatorna: lista of kontrast értékek epochonként}
    """
    contrast = {col: [[] for _ in range(MAX_EPOCHS)] for col in EEG_COLS}

    for file_epochs in epochs_list:
        for col in file_epochs:
            for epoch_idx, epoch in enumerate(file_epochs[col]):
                c = np.max(epoch) - np.min(epoch)
                contrast[col][epoch_idx].append(c)

    return contrast


def compute_ttest(beteg_contrast, egeszseges_contrast):
    """
    Epochonként és csatornánként elvégzi a kétmintás t-tesztet
    a beteg és egészséges csoport kontrasztértékein
    Visszatérési érték: dict {csatorna: lista of (t-statisztika, p-érték) epochonként}
    """
    ttest_results = {col: [] for col in EEG_COLS}

    for col in EEG_COLS:
        for i in range(MAX_EPOCHS):
            b = beteg_contrast[col][i]
            e = egeszseges_contrast[col][i]
            if len(b) >= 2 and len(e) >= 2:  # t-teszt legalább 2 minta kell mindkét csoportból
                t_stat, p_val = stats.ttest_ind(b, e)
                ttest_results[col].append((t_stat, p_val))
            else:
                ttest_results[col].append((0, 1.0))  # ha nincs elég adat, p=1 (nem szignifikáns)

    return ttest_results


def plot_all(beteg_contrast, egeszseges_contrast, ttest_results):
    """
    Egy figurában jeleníti meg a kontraszt és t-teszt eredményeket
    Felső sor: min-max kontraszt, alsó sor: t-teszt p-értékek
    """
    n_cols = len(EEG_COLS)
    fig, axes = plt.subplots(2, n_cols, figsize=(20, 10))
    fig.suptitle('Epoch analízis – beteg vs egészséges', fontsize=14)

    for idx, col in enumerate(EEG_COLS):
        epoch_labels = [f'E{i+1}' for i in range(MAX_EPOCHS)]
        x = np.arange(MAX_EPOCHS)
        width = 0.35

        # Kontraszt
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

        # t-teszt p érték
        ax_bot = axes[1][idx]

        pvals = [ttest_results[col][i][1] for i in range(MAX_EPOCHS)]

        # oszlopok színezése: szignifikáns (p<0.05) = sötét, nem szignifikáns = világos
        colors = ['#111111' if p < 0.05 else '#aaaaaa' for p in pvals]
        ax_bot.bar(x, pvals, width=0.5, color=colors, alpha=0.8)

        # 0.05-ös határvonal
        ax_bot.axhline(y=0.05, color='red', linestyle='--', linewidth=0.8, label='p=0.05')
        ax_bot.set_title(col)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(epoch_labels)
        ax_bot.set_ylabel('p-érték')
        ax_bot.legend(fontsize=7)

    axes[0][0].annotate('Min-Max kontraszt', xy=(0, 0.5), xytext=(-60, 0),
                        xycoords='axes fraction', textcoords='offset points',
                        fontsize=11, ha='right', va='center', rotation=90)
    axes[1][0].annotate('T-teszt p-érték', xy=(0, 0.5), xytext=(-60, 0),
                        xycoords='axes fraction', textcoords='offset points',
                        fontsize=11, ha='right', va='center', rotation=90)

    plt.tight_layout()
    plt.show()


def print_summary(beteg_contrast, egeszseges_contrast, ttest_results):
    """
    Szöveges összefoglaló a kontraszt és t-teszt eredményekről.
    """
    print("=" * 60)
    print("EPOCH ANALÍZIS ÖSSZEFOGLALÓ")
    print("=" * 60)

    for col in EEG_COLS:
        print(f"\n--- {col} ---")
        for i in range(MAX_EPOCHS):
            b_contrast = np.mean(beteg_contrast[col][i]) if beteg_contrast[col][i] else 0
            e_contrast = np.mean(egeszseges_contrast[col][i]) if egeszseges_contrast[col][i] else 0
            t_stat, p_val = ttest_results[col][i]
            szignifikans = "SZIGNIFIKÁNS" if p_val < 0.05 else "nem szignifikáns"

            print(f"  Epoch {i+1}:")
            print(f"    Kontraszt  – beteg: {b_contrast:.2f}, egészséges: {e_contrast:.2f}, különbség: {abs(b_contrast - e_contrast):.2f}")
            print(f"    T-teszt    – t={t_stat:.3f}, p={p_val:.4f} → {szignifikans}")


if __name__ == "__main__":
    print("Adatok betöltése...")
    beteg_epochs = load_epochs_from_dir(BETEG_DIR)
    egeszseges_epochs = load_epochs_from_dir(EGESZSEGES_DIR)

    print("Kontraszt számítása...")
    beteg_contrast = compute_contrast(beteg_epochs)
    egeszseges_contrast = compute_contrast(egeszseges_epochs)

    print("T-teszt futtatása...")
    ttest_results = compute_ttest(beteg_contrast, egeszseges_contrast)

    print("Eredmények kiírása...")
    print_summary(beteg_contrast, egeszseges_contrast, ttest_results)

    print("Grafikonok megjelenítése...")
    plot_all(beteg_contrast, egeszseges_contrast, ttest_results)