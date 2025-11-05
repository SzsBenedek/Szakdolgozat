import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from feature_extraction import extract_features, extract_featuresFFT, extract_featuresFFT_epoch
from train_model import train_and_evaluate_SVM, train_and_evaluate_rf
import warnings

warnings.filterwarnings("ignore")

FEATURE_FUNCS = {
    'Simple features': extract_features,
    'FFT features': extract_featuresFFT,
    'FFT epoch features': extract_featuresFFT_epoch
}

MODELS = {
    'SVM': train_and_evaluate_SVM,
    'Random Forest': train_and_evaluate_rf
}

# Globális változók
last_run_info = ""
last_used_feature = None

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

    return np.array(X), np.array(y)

def run_training():
    global last_run_info, last_used_feature
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

    acc, bal_acc, cm = model_func(X, y)
    last_run_info = (
        f"Összes minta: {len(X)}, jellemzők száma: {X.shape[1]}\n"
        f"Osztályok aránya – beteg: {sum(y)}, egészséges: {len(y)-sum(y)}\n\n"
        f"Pontosság: {acc*100:.2f}%\n"
        f"Balanced accuracy: {bal_acc*100:.2f}%\n"
        f"Confusion matrix:\n{cm}"
    )

    # Eltároljuk a legutóbbi futtatásnál használt feature extraction-t
    last_used_feature = feature_name

    messagebox.showinfo("Kész", f"A {model_name} modell betanítva a {feature_name} alapján.")

def show_visuals():
    if last_used_feature is None:
        messagebox.showerror("Hiba", "Először tanítsd meg a modellt!")
        return

    X, y = load_data(FEATURE_FUNCS[last_used_feature])
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    plt.scatter(X_2d[y==0,0], X_2d[y==0,1], c='green', label='Kontroll', alpha=0.6)
    plt.scatter(X_2d[y==1,0], X_2d[y==1,1], c='red', label='Beteg', alpha=0.6)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('EEG feature tér – PCA redukció')
    plt.legend()
    plt.grid(True)
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

def predict_file():
    global last_used_feature
    feature_name = feature_var.get()
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
    else:
        model = joblib.load("rf_model.pkl")

    pred = model.predict(features)[0]
    result = "BETEG" if pred == 1 else "EGÉSZSÉGES"

    messagebox.showinfo("Predikció", f"A fájl alapján: {result}")

root = tk.Tk()
root.title("EEG Classification GUI")

tk.Label(root, text="Válassz modellt:").grid(row=0, column=0, padx=10, pady=10)
model_var = tk.StringVar()
model_menu = ttk.Combobox(root, textvariable=model_var, values=list(MODELS.keys()), state="readonly")
model_menu.grid(row=0, column=1, padx=10, pady=10)

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

predict_btn = tk.Button(root, text="Fájl feltöltése és predikció", command=predict_file, bg="lightpink")
predict_btn.grid(row=5, column=0, columnspan=2, pady=10)

root.mainloop()
