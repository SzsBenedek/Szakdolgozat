import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1. EEG adatok betöltése és előfeldolgozása
nyers = mne.io.read_raw_eeglab('src/sub-001_eeg.set', preload=True)
nyers.filter(1., 40., fir_design='firwin')

if nyers.annotations:
    események, esemény_azonosító = mne.events_from_annotations(nyers)
    print("Események betöltve annotációkból.")
else:
    print("Nem találhatók annotációk, generálunk eseményeket fix hosszra.")
    események = mne.make_fixed_length_events(nyers, id=1, duration=2.0)
    esemény_azonosító = {'Fix': 1}

nyers.crop(tmax=90)
események = események[események[:, 0] <= nyers.last_samp]

# 2. Epokszámítás (adatok szeletelése események mentén)
epokkok = mne.Epochs(nyers, események, event_id=esemény_azonosító, tmin=0, tmax=2.0, baseline=None, preload=True)

# 3. Jellemzők kinyerése: theta + alpha sávok teljesítménye
def sáv_powerszámítás(epok_adat, mintavételi_freq):
    savok = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta':  (12, 30),
        'gamma': (30, 40)
    }
    jellemzok = []
    theta_alpha_osszeg = []
    for epok in epok_adat:
        sav_powers = {}
        psd, frekvenciak = mne.time_frequency.psd_array_welch(epok, sfreq=mintavételi_freq, fmin=1, fmax=40, n_fft=256)
        for sav, (also, felso) in savok.items():
            idx = np.logical_and(frekvenciak >= also, frekvenciak <= felso)
            teljesitmeny = psd[:, idx].mean(axis=1)
            sav_powers[sav] = teljesitmeny
        # Minden sáv teljesítményének összefűzése
        osszes_sav_powers = np.concatenate([sav_powers[sav] for sav in savok])
        jellemzok.append(osszes_sav_powers)
        # Theta és alpha sávok összesített teljesítménye csatornánként
        theta_alpha_osszeg.append(sav_powers['theta'].sum() + sav_powers['alpha'].sum())
    return np.array(jellemzok), np.array(theta_alpha_osszeg)

X, theta_alpha_ertekek = sáv_powerszámítás(epokkok.get_data(), epokkok.info['sfreq'])

# 4. Címkézés: a theta+alpha értékek felső 30%-a "élvezet" (1), többi "nem élvezet" (0)
határérték = np.percentile(theta_alpha_ertekek, 70)
y = np.array([1 if ertek >= határérték else 0 for ertek in theta_alpha_ertekek])

# 5. Jellemzők normalizálása
normalizalo = StandardScaler()
X_normalizalt = normalizalo.fit_transform(X)

# 6. Adat szétválasztása tanító és teszt halmazra, stratifikálva (az arányokat megtartva)
X_tanito, X_teszt, y_tanito, y_teszt = train_test_split(
    X_normalizalt, y, test_size=0.3, random_state=42, stratify=y
)

# 7. Paraméter keresés az SVM-hez, GridSearch
param_ter = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

svc = SVC(random_state=42)
racs_kereses = GridSearchCV(svc, param_ter, cv=5, scoring='accuracy', n_jobs=-1)
racs_kereses.fit(X_tanito, y_tanito)

print("Legjobb paraméterek:", racs_kereses.best_params_)

# 8. Legjobb modell kiértékelése
modell = racs_kereses.best_estimator_
y_pred = modell.predict(X_teszt)

print("\nOsztályozási jelentés:")
print(classification_report(y_teszt, y_pred, zero_division=1))

print("Konfúziós mátrix:")
print(confusion_matrix(y_teszt, y_pred))

# 9. Opcionális: EEG és epokkok megjelenítése
nyers.plot(n_channels=32, duration=10, scalings='auto')
epokkok.plot(n_epochs=10, n_channels=32)
plt.show(block=True)

# 10. Az "élvezet" események megjelenítése a nyers EEG adatban

# Esemény időpontok kigyűjtése az élvezet (1) címkékkel rendelkező epokkokból
elvezet_onsetek = []
elvezet_idotartamok = []

for i, (esemeny, cimke) in enumerate(zip(események, y)):
    if cimke == 1:
        kezdő_mp = esemeny[0] / nyers.info['sfreq']
        elvezet_onsetek.append(kezdő_mp)
        elvezet_idotartamok.append(2.0)  # epokkhossz

# Élvezet annotációk létrehozása
elvezet_annotaciok = mne.Annotations(
    onset=elvezet_onsetek,
    duration=elvezet_idotartamok,
    description=['Élevezi'] * len(elvezet_onsetek)
)

# Annotációk hozzáadása a nyers adathoz
if nyers.annotations is None:
    nyers.set_annotations(elvezet_annotaciok)
else:
    nyers.set_annotations(nyers.annotations + elvezet_annotaciok)

# EEG megjelenítése az élvezet annotációkkal kiemelve
nyers.plot(n_channels=32, duration=10, scalings='auto', title='Elvezet kiemelve', block=True)
