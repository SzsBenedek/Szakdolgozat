import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import plot_tree
import joblib
from feature_extraction import extract_statistical

DATA_DIR = 'data'
BETEG_DIR = os.path.join(DATA_DIR, 'beteg')
EGESZSEGES_DIR = os.path.join(DATA_DIR, 'egeszseges')

# feature nevek az extract_statistical-hoz
EEG_COLS = ['alpha1', 'alpha2', 'beta1', 'beta2', 'theta', 'gamma1', 'gamma2', 'attention', 'meditation']
RATIO_NAMES = ['alpha1/beta1', 'alpha2/beta2', 'theta/alpha1', 'theta/alpha2', 'gamma1/total', 'gamma2/total']


def get_feature_names():
    """Visszaadja az extract_statistical jellemzőinek neveit."""
    names = []
    for col in EEG_COLS:
        names += [f'{col}_mean', f'{col}_std', f'{col}_rms']
    names += RATIO_NAMES
    return names


def load_data():
    """Betölti az adatokat és visszaadja X és y tömböket."""
    X, y = [], []

    for file in os.listdir(BETEG_DIR):
        if file.endswith('.csv'):
            X.append(extract_statistical(os.path.join(BETEG_DIR, file)))
            y.append(1)

    for file in os.listdir(EGESZSEGES_DIR):
        if file.endswith('.csv'):
            X.append(extract_statistical(os.path.join(EGESZSEGES_DIR, file)))
            y.append(0)

    return np.array(X), np.array(y)


def plot_scatter(X, y, feature_names):
    """
    Két szórásdiagram egymás mellett:
    - Bal: SVM top 2 jellemző (beta2_mean, alpha2_std)
    - Jobb: RF+MLP top 2 jellemző (beta1_std, alpha2_std)
    """
    svm_feat1 = feature_names.index('beta2_mean')
    svm_feat2 = feature_names.index('alpha2_std')
    rf_feat1 = feature_names.index('beta1_std')
    rf_feat2 = feature_names.index('alpha2_std')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('2D szórásdiagram – beteg vs egészséges', fontsize=14)

    beteg_patch = mpatches.Patch(color='red', label='Beteg', alpha=0.7)
    egesz_patch = mpatches.Patch(color='green', label='Egészséges', alpha=0.7)

    # Bal oldal: SVM top 2
    ax1 = axes[0]
    ax1.scatter(X[y==1, svm_feat1], X[y==1, svm_feat2], c='red', alpha=0.7, s=60, label='Beteg')
    ax1.scatter(X[y==0, svm_feat1], X[y==0, svm_feat2], c='green', alpha=0.7, s=60, label='Egészséges')
    ax1.set_xlabel('beta2_mean')
    ax1.set_ylabel('alpha2_std')
    ax1.set_title('SVM – top 2 jellemző')
    ax1.legend(handles=[beteg_patch, egesz_patch])
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Jobb oldal: RF+MLP top 2
    ax2 = axes[1]
    ax2.scatter(X[y==1, rf_feat1], X[y==1, rf_feat2], c='red', alpha=0.7, s=60, label='Beteg')
    ax2.scatter(X[y==0, rf_feat1], X[y==0, rf_feat2], c='green', alpha=0.7, s=60, label='Egészséges')
    ax2.set_xlabel('beta1_std')
    ax2.set_ylabel('alpha2_std')
    ax2.set_title('RF+MLP – top 2 jellemző')
    ax2.legend(handles=[beteg_patch, egesz_patch])
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_radar(X, y, feature_names):
    """
    Pókháló diagram – 33 tengellyel, minden tengelyen
    az átlagos beteg és egészséges értékek normalizálva.
    """
    beteg_mean = np.mean(X[y==1], axis=0)
    egesz_mean = np.mean(X[y==0], axis=0)

    # normalizálás 0-1 közé hogy összehasonlítható legyen
    max_vals = np.maximum(beteg_mean, egesz_mean)
    max_vals[max_vals == 0] = 1
    beteg_norm = beteg_mean / max_vals
    egesz_norm = egesz_mean / max_vals

    N = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # zárt alakzathoz visszafűzzük az első értéket
    beteg_vals = beteg_norm.tolist() + [beteg_norm[0]]
    egesz_vals = egesz_norm.tolist() + [egesz_norm[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))
    fig.suptitle('Pókháló diagram – átlagos beteg vs egészséges\n(normalizált értékek)', fontsize=14)

    ax.plot(angles, beteg_vals, color='red', linewidth=1.5, label='Beteg')
    ax.fill(angles, beteg_vals, color='red', alpha=0.15)

    ax.plot(angles, egesz_vals, color='green', linewidth=1.5, label='Egészséges')
    ax.fill(angles, egesz_vals, color='green', alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, size=7)
    ax.set_yticklabels([])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_decision_tree(feature_names):
    """
    Kirajzolja a Random Forest egyik döntési fáját,
    megmutatva melyik jellemző alapján dönt a modell.
    Max 3 mélységig rajzolja ki, különben olvashatatlan lenne.
    """
    # betöltjük a már elmentett RF modellt
    model = joblib.load('rf_model.pkl')

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model.estimators_[0],  # az első fa a Random Forest-ből
        feature_names=feature_names,
        class_names=['Egészséges', 'Beteg'],
        filled=True,           # színekkel jelzi az osztályokat
        max_depth=3,           # max 3 szint, különben olvashatatlan
        ax=ax,
        fontsize=8
    )
    plt.title('Random Forest – egyik döntési fa (max 3 szint)', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Adatok betöltése...")
    X, y = load_data()
    feature_names = get_feature_names()

    print("Szórásdiagram megjelenítése...")
    plot_scatter(X, y, feature_names)

    print("Pókháló diagram megjelenítése...")
    plot_radar(X, y, feature_names)

    print("Döntési fa megjelenítése...")
    plot_decision_tree(feature_names)

    print("Kész!")