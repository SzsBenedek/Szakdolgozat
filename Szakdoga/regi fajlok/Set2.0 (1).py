import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. EEG adatok betöltése és előfeldolgozása
nyers = mne.io.read_raw_eeglab('sub-001.set', preload=True)
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
epokkok = mne.Epochs(
    nyers,
    események,
    event_id=esemény_azonosító,
    tmin=0,
    tmax=2.0,
    baseline=None,
    preload=True
)

# 3. Alpha és Theta sávteljesítmény számítása minden epokra és csatornára
def alpha_theta_powers(epok_adat, mintavételi_freq):
    """
    Epokkok (epokok x csatornák x minták) alapján
    kiszámolja az alpha (8-12 Hz) és theta (4-8 Hz) sáv átlagos teljesítményét.
    """
    n_epokok, n_csatornak, n_mintak = epok_adat.shape
    powers = np.zeros((n_epokok, n_csatornak, 2))  # utolsó dim: [theta, alpha]

    for i in range(n_epokok):
        for j in range(n_csatornak):
            fft_ertek = np.fft.rfft(epok_adat[i, j, :])
            amplitudo = np.abs(fft_ertek)
            frekvenciak = np.fft.rfftfreq(n_mintak, 1/mintavételi_freq)

            # Theta 4-8 Hz
            theta_idx = np.logical_and(frekvenciak >= 4, frekvenciak <= 8)
            powers[i, j, 0] = amplitudo[theta_idx].mean()

            # Alpha 8-12 Hz
            alpha_idx = np.logical_and(frekvenciak >= 8, frekvenciak <= 12)
            powers[i, j, 1] = amplitudo[alpha_idx].mean()

    return powers

# FFT és csatornaspecifikus alpha-theta jellemzők kiszámítása
epok_adat = epokkok.get_data()
sfreq = epokkok.info['sfreq']
alpha_theta_tomb = alpha_theta_powers(epok_adat, sfreq)

# Konzolra kiírás áttekinthetően
for i in range(alpha_theta_tomb.shape[0]):
    print(f"\nEpok {i+1}:")
    for j, ch_name in enumerate(epokkok.ch_names):
        print(f"{ch_name} - Theta: {alpha_theta_tomb[i,j,0]:.3f}, Alpha: {alpha_theta_tomb[i,j,1]:.3f}")

# 4. Jellemzők normalizálása (epok x (csatornák*2) formára lapítva)
n_epokok, n_csatornak, _ = alpha_theta_tomb.shape
X = alpha_theta_tomb.reshape(n_epokok, n_csatornak*2)
normalizalo = StandardScaler()
X_normalizalt = normalizalo.fit_transform(X)

# 5. Opcionális: EEG és epokkok megjelenítése
nyers.plot(n_channels=32, duration=10, scalings='auto')
epokkok.plot(n_epochs=10, n_channels=32)
plt.show(block=True)


# Ez a kód betölti és szűri az EEG jelet, 2 mp-es epokokra vágja,
# majd Fourier-transzformációval kiszámítja a theta (4–8 Hz) és alpha (8–12 Hz) sáv átlagos teljesítményét minden csatornára és epokra.
# Az eredményeket normalizálja, és előkészíti további elemzéshez/osztályozáshoz.