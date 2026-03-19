import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch


FS = 256  # mintavételezési frekvencia (Hz)

EEG_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

MENTAL_COLS = ['attention', 'meditation']



def bandpower_welch(signal, fs, fmin, fmax):
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])


def load_group_bandpower(folder):
    eeg_features = []
    mental_means = []

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file), sep=';')

            band_features = []

            for band, (fmin, fmax) in EEG_BANDS.items():
                all_band_power = []

                for col in df.columns:
                    if col.startswith(("alpha", "beta", "theta", "gamma", "raw")):
                        signal = df[col].values
                        bp = bandpower_welch(signal, FS, fmin, fmax)
                        all_band_power.append(bp)

                band_features.append(np.mean(all_band_power))

            eeg_features.append(band_features)

            mental_cols = [c for c in MENTAL_COLS if c in df.columns]
            mental_means.append(df[mental_cols].mean().values)

    return np.array(eeg_features), np.array(mental_means), list(EEG_BANDS.keys()), mental_cols


def plot_eeg_comparison(cols, beteg_avg, kontroll_avg):
    x = np.arange(len(cols))
    width = 0.35

    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, beteg_avg, width, label="Beteg")
    plt.bar(x + width/2, kontroll_avg, width, label="Egészséges")

    plt.xticks(x, cols)
    plt.ylabel("Band Power (Welch)")
    plt.title("EEG Sáv Teljesítmények – Beteg vs Kontroll")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_mental_comparison(cols, beteg_avg, kontroll_avg):
    x = np.arange(len(cols))
    width = 0.35

    plt.figure(figsize=(6,5))
    plt.bar(x - width/2, beteg_avg, width, label="Beteg")
    plt.bar(x + width/2, kontroll_avg, width, label="Egészséges")

    plt.xticks(x, cols)
    plt.ylabel("Átlagos érték")
    plt.title("Attention & Meditation – Beteg vs Kontroll")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()



def main():
    beteg_path = "data/group_by_song/beteg"
    kontroll_path = "data/group_by_song/kontroll"

    beteg_eeg, beteg_mental, eeg_cols, mental_cols = load_group_bandpower(beteg_path)
    kontroll_eeg, kontroll_mental, _, _ = load_group_bandpower(kontroll_path)

    beteg_eeg_avg = np.mean(beteg_eeg, axis=0)
    kontroll_eeg_avg = np.mean(kontroll_eeg, axis=0)

    beteg_mental_avg = np.mean(beteg_mental, axis=0)
    kontroll_mental_avg = np.mean(kontroll_mental, axis=0)

    plot_eeg_comparison(eeg_cols, beteg_eeg_avg, kontroll_eeg_avg)
    plot_mental_comparison(mental_cols, beteg_mental_avg, kontroll_mental_avg)

    print("\nEEG SÁV TELJESÍTMÉNY KÜLÖNBSÉGEK (WELCH):\n")
    for c, b, k in zip(eeg_cols, beteg_eeg_avg, kontroll_eeg_avg):
        print(f"{c:8s} | Beteg: {b:.6f} | Kontroll: {k:.6f} | Diff: {abs(b-k):.6f}")

    print("\nATTENTION & MEDITATION KÜLÖNBSÉGEK:\n")
    for c, b, k in zip(mental_cols, beteg_mental_avg, kontroll_mental_avg):
        print(f"{c:10s} | Beteg: {b:.4f} | Kontroll: {k:.4f} | Diff: {abs(b-k):.4f}")

    plt.show()


if __name__ == "__main__":
    main()
