import pandas as pd
import numpy as np
def extract_features(filepath):   #egyszeru statisztiaki featureok .9166

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
        alpha_power = np.mean(df['alpha1'] + df['alpha2'])
        beta_power = np.mean(df['beta1'] + df['beta2'])
        gamma_power = np.mean(df['gamma1'] + df['gamma2'])
        theta_power = np.mean(df['theta'])

        ratio_alpha_beta = alpha_power / (beta_power + 1e-9)
        ratio_theta_alpha = theta_power / (alpha_power + 1e-9)
        ratio_gamma_total = gamma_power / (alpha_power + beta_power + theta_power + 1e-9)
        features.extend([ratio_alpha_beta, ratio_theta_alpha, ratio_gamma_total])
    except KeyError:
        pass

    return np.array(features, dtype=float)


def extract_featuresFFT(filepath, sfreq=256, epoch_sec=2, max_epochs=5): #FFT alapu featureok .75
    df = pd.read_csv(filepath, sep=';')

    eeg_cols = ['alpha1','alpha2','beta1','beta2','theta','gamma1','gamma2']
    stats_cols = ['attention','meditation']
    eeg_cols = [col for col in eeg_cols if col in df.columns]
    stats_cols = [col for col in stats_cols if col in df.columns]

    data = df[eeg_cols].values
    features = []

    for i, col in enumerate(eeg_cols):
        x = data[:, i]
        features.extend([np.mean(x), np.std(x), np.mean(x**2)])
    n_samples = data.shape[0]
    epoch_len = epoch_sec * sfreq
    n_epochs = n_samples // epoch_len
    n_epochs = min(n_epochs, max_epochs)

    for i in range(n_epochs):
        start = int(i * epoch_len)
        end = int(start + epoch_len)
        epoch = data[start:end, :]

        for j in range(epoch.shape[1]):
            fft_vals = np.fft.rfft(epoch[:, j])
            fft_power = np.abs(fft_vals) ** 2
            features.extend([
                np.mean(fft_power),
                np.std(fft_power),
                np.max(fft_power)
            ])
    n_features_per_epoch_per_chan = 3  # mean, std, max
    missing_epochs = max_epochs - n_epochs
    if missing_epochs > 0:
        features.extend([0] * missing_epochs * len(eeg_cols) * n_features_per_epoch_per_chan)
    for col in stats_cols:
        x = df[col].values
        features.extend([np.mean(x), np.std(x)])
    try:
        alpha_power = np.mean(df['alpha1'] + df['alpha2'])
        beta_power = np.mean(df['beta1'] + df['beta2'])
        gamma_power = np.mean(df['gamma1'] + df['gamma2'])
        theta_power = np.mean(df['theta'])
        ratio_alpha_beta = alpha_power / (beta_power + 1e-9)
        ratio_theta_alpha = theta_power / (alpha_power + 1e-9)
        ratio_gamma_total = gamma_power / (alpha_power + beta_power + theta_power + 1e-9)
        features.extend([ratio_alpha_beta, ratio_theta_alpha, ratio_gamma_total])
    except KeyError:
        pass

    return np.array(features, dtype=float)


def extract_featuresFFT_epoch(filepath, sfreq=256, epoch_sec=2, max_epochs=5): #FFT alapu epochonkenti featureok .8333
    df = pd.read_csv(filepath, sep=';')

    eeg_cols = ['alpha1','alpha2','beta1','beta2','theta','gamma1','gamma2']
    stats_cols = ['attention','meditation']
    eeg_cols = [col for col in eeg_cols if col in df.columns]
    stats_cols = [col for col in stats_cols if col in df.columns]

    data = df[eeg_cols].values
    features = []
    for i, col in enumerate(eeg_cols):
        x = data[:, i]
        features.extend([np.mean(x), np.std(x), np.mean(x**2)])
    n_samples = data.shape[0]
    epoch_len = epoch_sec * sfreq
    n_epochs = n_samples // epoch_len
    n_epochs = min(n_epochs, max_epochs)
    freqs = np.fft.rfftfreq(epoch_len, 1/sfreq)
    alpha_idx = (freqs >= 8) & (freqs <= 12)
    beta_idx  = (freqs >= 13) & (freqs <= 30)
    theta_idx = (freqs >= 4) & (freqs <= 7)
    gamma_idx = (freqs >= 30) & (freqs <= 45)

    for i in range(n_epochs):
        start = int(i * epoch_len)
        end = int(start + epoch_len)
        epoch = data[start:end, :]

        for j in range(epoch.shape[1]):
            fft_vals = np.fft.rfft(epoch[:, j])
            fft_power = np.abs(fft_vals)**2
            features.extend([
                np.mean(fft_power[alpha_idx]),
                np.mean(fft_power[beta_idx]),
                np.mean(fft_power[theta_idx]),
                np.mean(fft_power[gamma_idx])
            ])

    n_features_per_epoch_per_chan = 4  # alpha, beta, theta, gamma
    missing_epochs = max_epochs - n_epochs
    if missing_epochs > 0:
        features.extend([0] * missing_epochs * len(eeg_cols) * n_features_per_epoch_per_chan)

    for col in stats_cols:
        for i in range(n_epochs):
            start = int(i*epoch_len)
            end = int(start+epoch_len)
            epoch_vals = df[col].values[start:end]
            features.extend([np.mean(epoch_vals), np.std(epoch_vals)])
    missing_epochs_stats = max_epochs - n_epochs
    if missing_epochs_stats > 0:
        features.extend([0] * missing_epochs_stats * len(stats_cols) * 2)

    try:
        alpha_power = np.mean(df['alpha1'] + df['alpha2'])
        beta_power = np.mean(df['beta1'] + df['beta2'])
        gamma_power = np.mean(df['gamma1'] + df['gamma2'])
        theta_power = np.mean(df['theta'])
        ratio_alpha_beta = alpha_power / (beta_power + 1e-9)
        ratio_theta_alpha = theta_power / (alpha_power + 1e-9)
        ratio_gamma_total = gamma_power / (alpha_power + beta_power + theta_power + 1e-9)
        features.extend([ratio_alpha_beta, ratio_theta_alpha, ratio_gamma_total])
    except KeyError:
        pass

    return np.array(features, dtype=float)

