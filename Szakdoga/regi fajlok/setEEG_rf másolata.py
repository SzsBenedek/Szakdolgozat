# eeg felvetelek .set kiterjesztessel valo feldolgozasa majd Random Foresttel modell betanitasa

import mne
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. EEG .set fájl betöltése
raw = mne.io.read_raw_eeglab('src/sub-001_eeg.set', preload=True)

# 2. Szűrés / előfeldogozás
raw.filter(1., 40., fir_design='firwin')

# 3. Epoch-olás (a folyamatos jelet szegmensekre bontjuk)
#    Ehhez esemény annotációkra van szükség – ha nincs, szimulálunk például 2 másodperces szegmenseket
events = mne.make_fixed_length_events(raw, id=1, duration=2.0)  # egy esemény 2 másodpercenként
epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, preload=True)

# 4. Jellemzők kinyerése – sávszélességre eső teljesítmény átlagolása
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
            power = psd[:, idx].mean(axis=1)  # csatornák és sáv mentén vett átlag
            band_powers.extend(power)
        features.append(np.array(band_powers).flatten())
    return np.array(features)

# Jellemzők kinyerése az epoch-olt adatokból
X = extract_band_power(epochs.get_data(), epochs.info['sfreq'])

# 5. Dummy (ál) címkék létrehozása – csak példa kedvéért, valódi címkéket kellene használni!
y = np.array([0 if i < len(X)/2 else 1 for i in range(len(X))])  # bináris címkék

# 6. Tanító és teszt adathalmaz szétválasztása
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 7. Random Forest tanítása
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 8. Kiértékelés
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
