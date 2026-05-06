import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from feature_extraction import extract_statistical, extract_phase_epoch, extract_phase_band
from train_model import train_and_evaluate_svm, train_and_evaluate_rf, train_and_evaluate_xgb, train_and_evaluate_rf_mlp
import warnings

warnings.filterwarnings("ignore")

FEATURE_FUNCS = {
    'Statistical': extract_statistical,
    'Phase epoch': extract_phase_epoch,
    'Phase band': extract_phase_band
}

MODELS = {
    'SVM': train_and_evaluate_svm,
    'Random Forest': train_and_evaluate_rf,
    'Random Forest MLP': train_and_evaluate_rf_mlp,
    'XGBoost': train_and_evaluate_xgb
}

# Globális változók
last_run_info = ""
last_used_feature = None
last_feature_importances = None
last_feature_names = None

def get_feature_names(feature_name, n_features):
    eeg_cols = ['alpha1', 'alpha2', 'beta1', 'beta2', 'theta', 'gamma1', 'gamma2']
    stats_cols = ['attention', 'meditation']
    ratio_names = [
        'alpha1/beta1', 'alpha2/beta2',
        'theta/alpha1', 'theta/alpha2',
        'gamma1/total', 'gamma2/total'
    ]

    names = []

    if feature_name == 'Statistical':
        for col in eeg_cols + stats_cols:
            names += [f'{col}_mean', f'{col}_std', f'{col}_rms']
        names += ratio_names

    elif feature_name == 'Phase epoch':
        for col in eeg_cols:
            names += [f'{col}_mean', f'{col}_std', f'{col}_rms']
        for epoch in range(5):
            for col in eeg_cols:
                names += [
                    f'e{epoch+1}_{col}_phase_mean',
                    f'e{epoch+1}_{col}_phase_std',
                    f'e{epoch+1}_{col}_phase_max'
                ]
        for col in stats_cols:
            names += [f'{col}_mean', f'{col}_std']
        names += ratio_names

    elif feature_name == 'Phase band':
        for col in eeg_cols:
            names += [f'{col}_mean', f'{col}_std', f'{col}_rms']
        bands = ['alpha1', 'alpha2', 'beta1', 'beta2', 'theta', 'gamma1', 'gamma2']
        for epoch in range(5):
            for col in eeg_cols:
                for band in bands:
                    names.append(f'e{epoch+1}_{col}_{band}_phase')
        for epoch in range(5):
            for col in stats_cols:
                names += [f'e{epoch+1}_{col}_mean', f'e{epoch+1}_{col}_std']
        names += ratio_names

    # ha valamiért nem egyezik a hossz, fallback
    if len(names) != n_features:
        names = [f'feature_{i}' for i in range(n_features)]

    return names


def load_data(feature_func, base_dir='data'):
    X, y = [], []

    beteg_dir = os.path.join(base_dir, 'beteg')
    kontroll_dir = os.path.join(base_dir, 'egeszseges')

    for file in os.listdir(beteg_dir):
        if file.endswith('.csv'):
            X.append(feature_func(os.path.join(beteg_dir, file)))
            y.append(1)

    for file in os.listdir(kontroll_dir):
        if file.endswith('.csv'):
            X.append(feature_func(os.path.join(kontroll_dir, file)))
            y.append(0)

    X = np.array(X)
    print(f"Betöltött adatok alakja: {X.shape}")
    return X, np.array(y)


def run_training():
    global last_run_info, last_used_feature, last_feature_importances, last_feature_names
    feature_name = feature_var.get()
    model_name = model_var.get()

    if not feature_name or not model_name:
        messagebox.showwarning("Hiányzó választás", "Válassz modellt és feature extraction-t!")
        return

    feature_func = FEATURE_FUNCS[feature_name]
    model_func = MODELS[model_name]

    X, y = load_data(feature_func)
    if len(X) < 2:
        messagebox.showerror("Hiba", "Túl kevés adat a betanításhoz.")
        return

    acc, bal_acc, cm, feature_importances = model_func(X, y)

    last_run_info = (
        f"Összes minta: {len(X)}, jellemzők száma: {X.shape[1]}\n"
        f"Osztályok aránya – beteg: {sum(y)}, egészséges: {len(y)-sum(y)}\n\n"
        f"Pontosság: {acc*100:.2f}%\n"
        f"Balanced accuracy: {bal_acc*100:.2f}%\n"
        f"Confusion matrix:\n{cm}"
    )

    last_used_feature = feature_name
    last_feature_importances = feature_importances
    last_feature_names = get_feature_names(feature_name, X.shape[1])

    messagebox.showinfo("Kész", f"A {model_name} modell betanítva a {feature_name} alapján.")


def show_visuals():
    if last_used_feature is None:
        messagebox.showerror("Hiba", "Először tanítsd meg a modellt!")
        return

    X, y = load_data(FEATURE_FUNCS[last_used_feature])
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    X_2d_shifted = X_2d - X_2d.min(axis=0) + 1e-6

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d_shifted[y==0, 0], X_2d_shifted[y==0, 1], c='green', label='Kontroll', alpha=0.6)
    plt.scatter(X_2d_shifted[y==1, 0], X_2d_shifted[y==1, 1], c='red', label='Beteg', alpha=0.6)
    plt.xlabel('PCA 1 (log skála)')
    plt.ylabel('PCA 2 (log skála)')
    plt.title('EEG feature tér – PCA redukció (log skálán)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()


def show_last_run_info():
    global last_run_info
    if not last_run_info:
        messagebox.showinfo("Info", "Még nem futott tanítás.")
        return

    win = tk.Toplevel(root)
    win.title("Tanítás eredménye")
    st = scrolledtext.ScrolledText(win, width=60, height=20)
    st.pack(padx=10, pady=10)
    st.insert(tk.END, last_run_info)
    st.configure(state='disabled')


def show_feature_importance():
    if last_feature_importances is None:
        messagebox.showerror("Hiba", "Először tanítsd meg a modellt!")
        return

    importances = last_feature_importances
    names = last_feature_names

    # top 20 jellemző csökkenő sorrendben
    top_n = min(20, len(importances))
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_names = [names[i] for i in indices]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_importances[::-1], color='steelblue')
    plt.yticks(range(top_n), top_names[::-1])
    plt.xlabel('Fontossági érték')
    plt.title(f'Top {top_n} legmérvadóbb jellemző')
    plt.tight_layout()
    plt.show()


def update_feature_options(event=None):
    model_name = model_var.get()
    if model_name == "Random Forest":
        feature_menu['values'] = ['Statistical']
        feature_var.set('Statistical')
        feature_menu.config(state='disabled')
    else:
        feature_menu['values'] = list(FEATURE_FUNCS.keys())
        feature_menu.config(state='readonly')


def predict_file():
    global last_used_feature
    model_name = model_var.get()

    if last_used_feature is None:
        messagebox.showerror("Hiba", "Először tanítsd meg a modellt!")
        return

    filepath = filedialog.askopenfilename(
        title="Válassz EEG CSV fájlt",
        filetypes=[("CSV fájlok", "*.csv")]
    )
    if not filepath:
        return

    feature_func = FEATURE_FUNCS[last_used_feature]
    features = feature_func(filepath).reshape(1, -1)

    if model_name == "SVM":
        model = joblib.load("svm_model.pkl")
    elif model_name == "XGBoost":
        model = joblib.load("xgb_model.pkl")
    elif model_name == "Random Forest":
        model = joblib.load("rf_model.pkl")
    elif model_name == "Random Forest MLP":
        model = joblib.load("rf_mlp_model.pkl")

    pred = model.predict(features)[0]
    result = "BETEG" if pred == 1 else "EGÉSZSÉGES"
    messagebox.showinfo("Predikció", f"A fájl alapján: {result}")


# GUI elemek
root = tk.Tk()
root.title("EEG Classification GUI")

tk.Label(root, text="Válassz modellt:").grid(row=0, column=0, padx=10, pady=10)
model_var = tk.StringVar()
model_menu = ttk.Combobox(root, textvariable=model_var, values=list(MODELS.keys()), state="readonly")
model_menu.grid(row=0, column=1, padx=10, pady=10)
model_menu.bind("<<ComboboxSelected>>", update_feature_options)

tk.Label(root, text="Válassz feature extraction-t:").grid(row=1, column=0, padx=10, pady=10)
feature_var = tk.StringVar()
feature_menu = ttk.Combobox(root, textvariable=feature_var, values=list(FEATURE_FUNCS.keys()), state="readonly")
feature_menu.grid(row=1, column=1, padx=10, pady=10)

train_btn = tk.Button(root, text="Tanítás", command=run_training, bg="lightblue")
train_btn.grid(row=2, column=0, columnspan=2, pady=10)

visual_btn = tk.Button(root, text="Vizualizáció", command=show_visuals, bg="lightgreen")
visual_btn.grid(row=3, column=0, columnspan=2, pady=10)

info_btn = tk.Button(root, text="Tanítás adatai", command=show_last_run_info, bg="lightyellow")
info_btn.grid(row=4, column=0, columnspan=2, pady=10)

importance_btn = tk.Button(root, text="Jellemzők fontossága", command=show_feature_importance, bg="lightyellow")
importance_btn.grid(row=5, column=0, columnspan=2, pady=10)

predict_btn = tk.Button(root, text="Fájl feltöltése és predikció", command=predict_file, bg="lightpink")
predict_btn.grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()