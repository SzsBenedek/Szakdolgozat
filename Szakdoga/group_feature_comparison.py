import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EEG_COLS = [
    'alpha1', 'alpha2', 'beta1', 'beta2',
    'theta', 'gamma1', 'gamma2'
]

MENTAL_COLS = ['attention', 'meditation']


def load_group_means(folder):
    eeg_means = []
    mental_means = []

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file), sep=';')

            eeg_cols = [c for c in EEG_COLS if c in df.columns]
            mental_cols = [c for c in MENTAL_COLS if c in df.columns]

            eeg_means.append(df[eeg_cols].mean().values)
            mental_means.append(df[mental_cols].mean().values)

    return np.array(eeg_means), np.array(mental_means), eeg_cols, mental_cols


def plot_eeg_comparison(cols, beteg_avg, kontroll_avg):
    x = np.arange(len(cols))
    width = 0.35

    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, beteg_avg, width, label="Beteg", color="red")
    plt.bar(x + width/2, kontroll_avg, width, label="Egészséges", color="green")

    plt.xticks(x, cols, rotation=45)
    plt.ylabel("Átlagos EEG aktivitás")
    plt.title("EEG hullámok – Beteg vs Kontroll")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_mental_comparison(cols, beteg_avg, kontroll_avg):
    x = np.arange(len(cols))
    width = 0.35

    plt.figure(figsize=(6,5))
    plt.bar(x - width/2, beteg_avg, width, label="Beteg", color="orange")
    plt.bar(x + width/2, kontroll_avg, width, label="Egészséges", color="blue")

    plt.xticks(x, cols)
    plt.ylabel("Átlagos érték")
    plt.title("Attention & Meditation – Beteg vs Kontroll")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def main():
    beteg_path = "data/beteg"
    kontroll_path = "data/egeszseges"

    beteg_eeg, beteg_mental, eeg_cols, mental_cols = load_group_means(beteg_path)
    kontroll_eeg, kontroll_mental, _, _ = load_group_means(kontroll_path)

    beteg_eeg_avg = np.mean(beteg_eeg, axis=0)
    kontroll_eeg_avg = np.mean(kontroll_eeg, axis=0)

    beteg_mental_avg = np.mean(beteg_mental, axis=0)
    kontroll_mental_avg = np.mean(kontroll_mental, axis=0)

    plot_eeg_comparison(eeg_cols, beteg_eeg_avg, kontroll_eeg_avg)
    plot_mental_comparison(mental_cols, beteg_mental_avg, kontroll_mental_avg)


    print("\nEEG HULLÁM KÜLÖNBSÉGEK:\n")
    for c, b, k in zip(eeg_cols, beteg_eeg_avg, kontroll_eeg_avg):
        print(f"{c:10s} | Beteg: {b:.4f} | Kontroll: {k:.4f} | Diff: {abs(b-k):.4f}")

    print("\nATTENTION & MEDITATION KÜLÖNBSÉGEK:\n")
    for c, b, k in zip(mental_cols, beteg_mental_avg, kontroll_mental_avg):
        print(f"{c:10s} | Beteg: {b:.4f} | Kontroll: {k:.4f} | Diff: {abs(b-k):.4f}")
    plt.show()

if __name__ == "__main__":
    main()
