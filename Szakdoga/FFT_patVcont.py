import numpy as np
import matplotlib.pyplot as plt
import csv

def load_csv(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        data = list(reader)

    data = np.array(data, dtype=float)

    if 'rawEEG' in header:
        eeg_index = header.index('rawEEG')
        signal = data[:, eeg_index]
    else:
        signal = data[:, 0]

    return signal

def compute_fft(signal, fs=256):
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(n, d=1/fs)
    fft_power = np.abs(fft_vals[:n // 2]) ** 2
    fft_freqs = fft_freqs[:n // 2]
    return fft_freqs, fft_power

def plot_individual(patient_freqs, patient_power, control_freqs, control_power):
    x_min, x_max = 0, 1.75
    y_min, y_max = 0, 0.1e12  # figyelembe veszi a nagyságrendet

    # Beteg FFT
    plt.figure(figsize=(10, 5))
    plt.plot(patient_freqs, patient_power, color='red')
    plt.title("Beteg EEG FFT spektruma")
    plt.xlabel("Frekvencia (Hz)")
    plt.ylabel("Teljesítmény")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.tight_layout()

    # Egészséges FFT
    plt.figure(figsize=(10, 5))
    plt.plot(control_freqs, control_power, color='green')
    plt.title("Egészséges EEG FFT spektruma")
    plt.xlabel("Frekvencia (Hz)")
    plt.ylabel("Teljesítmény")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    patient_file = "data/beteg/Emese_Csaj.csv"
    control_file = "data/egeszseges/Kontroll_Csaj.csv"

    patient_data = load_csv(patient_file)
    control_data = load_csv(control_file)

    patient_freqs, patient_power = compute_fft(patient_data)
    control_freqs, control_power = compute_fft(control_data)

    plot_individual(patient_freqs, patient_power, control_freqs, control_power)

if __name__ == "__main__":
    main()
