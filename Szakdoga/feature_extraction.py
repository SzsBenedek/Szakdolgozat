import pandas as pd
import numpy as np

def extract_statistical(filepath):
    df = pd.read_csv(filepath, sep=';')

    eeg_cols = [
        'alpha1', 'alpha2', 'beta1', 'beta2',
        'theta', 'gamma1', 'gamma2', 'attention', 'meditation'
    ]
    eeg_cols = [col for col in eeg_cols if col in df.columns]
    data = df[eeg_cols]

    features = []

    for col in eeg_cols:
        x = data[col].values
        features.extend([
            np.mean(x),
            np.std(x),
            np.mean(x ** 2)
        ])

    try:
        # minden sáv külön kezelve, nem összevonva
        alpha1_power = np.mean(df['alpha1'])
        alpha2_power = np.mean(df['alpha2'])
        beta1_power = np.mean(df['beta1'])
        beta2_power = np.mean(df['beta2'])
        gamma1_power = np.mean(df['gamma1'])
        gamma2_power = np.mean(df['gamma2'])
        theta_power = np.mean(df['theta'])

        total_power = alpha1_power + alpha2_power + beta1_power + beta2_power + gamma1_power + gamma2_power + theta_power + 1e-9

        # arányok külön sávonként
        features.extend([
            alpha1_power / (beta1_power + 1e-9),   # alpha1 / beta1
            alpha2_power / (beta2_power + 1e-9),   # alpha2 / beta2
            theta_power / (alpha1_power + 1e-9),   # theta / alpha1
            theta_power / (alpha2_power + 1e-9),   # theta / alpha2
            gamma1_power / total_power,             # gamma1 / összes
            gamma2_power / total_power,             # gamma2 / összes
        ])
    except KeyError:
        pass

    return np.array(features, dtype=float)


def extract_phase_epoch(filepath, sfreq=256, epoch_sec=2, max_epochs=5):
    df = pd.read_csv(filepath, sep=';')

    eeg_cols = ['alpha1', 'alpha2', 'beta1', 'beta2', 'theta', 'gamma1', 'gamma2']
    stats_cols = ['attention', 'meditation']
    eeg_cols = [col for col in eeg_cols if col in df.columns]
    stats_cols = [col for col in stats_cols if col in df.columns]

    data = df[eeg_cols].values
    features = []

    # minden csatorna külön statisztikái
    for i in range(len(eeg_cols)):
        x = data[:, i]
        features.extend([np.mean(x), np.std(x), np.mean(x**2)])

    # epoch-ok letrehozasa
    n_samples = data.shape[0]
    epoch_len = epoch_sec * sfreq
    n_epochs = min(n_samples // epoch_len, max_epochs)

    for i in range(n_epochs):
        start = int(i * epoch_len)
        end = int(start + epoch_len)
        epoch = data[start:end, :]

        # minden csatorna külön FFT és fázisszög
        for j in range(epoch.shape[1]):
            fft_vals = np.fft.rfft(epoch[:, j])
            phase = np.angle(fft_vals)
            features.extend([
                np.mean(phase),
                np.std(phase),
                np.max(np.abs(phase))
            ])

    # hianyzo epoch-ok potlasa
    missing_epochs = max_epochs - n_epochs
    if missing_epochs > 0:
        features.extend([0] * missing_epochs * len(eeg_cols) * 3)

    # attention es meditation
    for col in stats_cols:
        x = df[col].values
        features.extend([np.mean(x), np.std(x)])

    try:
        alpha1_power = np.mean(df['alpha1'])
        alpha2_power = np.mean(df['alpha2'])
        beta1_power = np.mean(df['beta1'])
        beta2_power = np.mean(df['beta2'])
        gamma1_power = np.mean(df['gamma1'])
        gamma2_power = np.mean(df['gamma2'])
        theta_power = np.mean(df['theta'])
        total_power = alpha1_power + alpha2_power + beta1_power + beta2_power + gamma1_power + gamma2_power + theta_power + 1e-9

        # savaranyok
        features.extend([
            alpha1_power / (beta1_power + 1e-9),
            alpha2_power / (beta2_power + 1e-9),
            theta_power / (alpha1_power + 1e-9),
            theta_power / (alpha2_power + 1e-9),
            gamma1_power / total_power,
            gamma2_power / total_power,
        ])
    except KeyError:
        pass

    return np.array(features, dtype=float)


def extract_phase_band(filepath, sfreq=256, epoch_sec=2, max_epochs=5):
    df = pd.read_csv(filepath, sep=';')

    eeg_cols = ['alpha1', 'alpha2', 'beta1', 'beta2', 'theta', 'gamma1', 'gamma2']
    stats_cols = ['attention', 'meditation']
    eeg_cols = [col for col in eeg_cols if col in df.columns]
    stats_cols = [col for col in stats_cols if col in df.columns]

    data = df[eeg_cols].values
    features = []

    # minden csatorna külön statisztikái
    for i in range(len(eeg_cols)):
        x = data[:, i]
        features.extend([np.mean(x), np.std(x), np.mean(x**2)])

    n_samples = data.shape[0]
    epoch_len = epoch_sec * sfreq
    n_epochs = min(n_samples // epoch_len, max_epochs)

    freqs = np.fft.rfftfreq(epoch_len, 1/sfreq)

    # minden sáv külön frekvenciahatárral
    alpha1_idx = (freqs >= 8)  & (freqs <= 10)
    alpha2_idx = (freqs >= 10) & (freqs <= 12)
    beta1_idx  = (freqs >= 13) & (freqs <= 20)
    beta2_idx  = (freqs >= 20) & (freqs <= 30)
    theta_idx  = (freqs >= 4)  & (freqs <= 7)
    gamma1_idx = (freqs >= 30) & (freqs <= 40)
    gamma2_idx = (freqs >= 40) & (freqs <= 45)

    band_indices = [alpha1_idx, alpha2_idx, beta1_idx, beta2_idx,
                    theta_idx, gamma1_idx, gamma2_idx]

    for i in range(n_epochs):
        start = int(i * epoch_len)
        end = int(start + epoch_len)
        epoch = data[start:end, :]

        for j in range(epoch.shape[1]):
            fft_vals = np.fft.rfft(epoch[:, j])
            phase = np.angle(fft_vals)
            # minden sávhoz külön fázis átlag
            for band_idx in band_indices:
                features.append(np.mean(phase[band_idx]) if np.any(band_idx) else 0.0)

    n_features_per_epoch_per_chan = len(band_indices)
    missing_epochs = max_epochs - n_epochs
    if missing_epochs > 0:
        features.extend([0] * missing_epochs * len(eeg_cols) * n_features_per_epoch_per_chan)

    for col in stats_cols:
        for i in range(n_epochs):
            start = int(i * epoch_len)
            end = int(start + epoch_len)
            epoch_vals = df[col].values[start:end]
            features.extend([np.mean(epoch_vals), np.std(epoch_vals)])

    missing_epochs_stats = max_epochs - n_epochs
    if missing_epochs_stats > 0:
        features.extend([0] * missing_epochs_stats * len(stats_cols) * 2)

    try:
        alpha1_power = np.mean(df['alpha1'])
        alpha2_power = np.mean(df['alpha2'])
        beta1_power = np.mean(df['beta1'])
        beta2_power = np.mean(df['beta2'])
        gamma1_power = np.mean(df['gamma1'])
        gamma2_power = np.mean(df['gamma2'])
        theta_power = np.mean(df['theta'])
        total_power = alpha1_power + alpha2_power + beta1_power + beta2_power + gamma1_power + gamma2_power + theta_power + 1e-9

        features.extend([
            alpha1_power / (beta1_power + 1e-9),
            alpha2_power / (beta2_power + 1e-9),
            theta_power / (alpha1_power + 1e-9),
            theta_power / (alpha2_power + 1e-9),
            gamma1_power / total_power,
            gamma2_power / total_power,
        ])
    except KeyError:
        pass

    return np.array(features, dtype=float)