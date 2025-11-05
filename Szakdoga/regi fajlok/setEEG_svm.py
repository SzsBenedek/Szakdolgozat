import mne
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

raw = mne.io.read_raw_eeglab('src/sub-001_eeg.set', preload=True)
raw.filter(1., 40., fir_design='firwin')
if raw.annotations:
    events, event_id = mne.events_from_annotations(raw)
    print("Események betöltve annotációkból.")
else:
    print("Nem találhatók annotációk, generálunk eseményeket fix hosszra.")
    events = mne.make_fixed_length_events(raw, id=1, duration=2.0)
    event_id = {'Fix': 1}
raw.crop(tmax=90)
events = events[events[:, 0] <= raw.last_samp]
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=2.0, baseline=None, preload=True)
def extract_band_power(epoch_data, sfreq):
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta':  (12, 30),
        'gamma': (30, 40)
    }
    features = []
    for epoch in epoch_data:
        band_powers = []
        psd, freqs = mne.time_frequency.psd_array_welch(epoch, sfreq=sfreq, fmin=1, fmax=40, n_fft=256)
        for band, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            power = psd[:, idx].mean(axis=1)  # átlagos sávteljesítmény minden csatornára
            band_powers.extend(power)
        features.append(np.array(band_powers).flatten())
    return np.array(features)

X = extract_band_power(epochs.get_data(), epochs.info['sfreq'])
y = np.array([0 if i < len(X)/2 else 1 for i in range(len(X))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=1))
raw.plot(n_channels=32, duration=10, scalings='auto')
epochs.plot(n_epochs=10, n_channels=32)
plt.show(block=True)