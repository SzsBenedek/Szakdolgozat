import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

#Csv betoltese
df = pd.read_csv('ZsuzsiKab.csv', sep=';')
csatornak = ['rawEEG']
data = df[csatornak].values.T  # shape: (csatornák, minták)
sfreq = 256  # mintavételi frekvencia
info = mne.create_info(ch_names=csatornak, sfreq=sfreq, ch_types='eeg')
nyers = mne.io.RawArray(data, info)
#Szűrés és epocholas
nyers.filter(1., 40., fir_design='firwin')
események = mne.make_fixed_length_events(nyers, id=1, duration=2.0)
epokkok = mne.Epochs(
    nyers,
    események,
    event_id={'Epoch': 1},
    tmin=0,
    tmax=2.0,
    baseline=None,
    preload=True
)
#Vizualizacio
epokkok.plot(
    n_epochs=10,       # egyszerre hány epokot mutat
    n_channels=len(csatornak),  # csatornák száma
    scalings='auto',   # automatikus skálázás
    title='EEG epokok',
    block=True         # interaktív GUI
)

