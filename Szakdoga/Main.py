import os
import numpy as np
from feature_extraction import extract_features
from feature_extraction import extract_featuresFFT_epoch
from train_model import train_and_evaluate
import warnings
import joblib


warnings.filterwarnings("ignore")

def load_data(base_dir='data'):
    X, y = [], []

    beteg_dir = os.path.join(base_dir, 'beteg')
    kontroll_dir = os.path.join(base_dir, 'egeszseges')

    for file in os.listdir(beteg_dir):
        if file.endswith('.csv'):
            X.append(extract_features(os.path.join(beteg_dir, file)))
            y.append(1)

    for file in os.listdir(kontroll_dir):
        if file.endswith('.csv'):
            X.append(extract_features(os.path.join(kontroll_dir, file)))
            y.append(0)

    return np.array(X), np.array(y)
def predict_csv(file_path):
    model = joblib.load('svm_model.pkl')
    features = extract_features(file_path).reshape(1, -1)
    pred = model.predict(features)[0]
    return 'BETEG' if pred == 1 else 'EGÉSZSÉGES'


def main():
    print("EEG fájlok beolvasása és jellemzőképzés...")
    X, y = load_data('data')

    print(f"Összes minta: {len(X)}, jellemzők száma: {X.shape[1]}")
    print(f"Osztályok aránya – beteg: {sum(y)}, egészséges: {len(y) - sum(y)}")

    # Tanítás és kiértékelés
    train_and_evaluate(X, y)

    # Példa új fájl predikciójára
    test_file = 'data/beteg/Zsuzsi_Bartok5.csv'
    result = predict_csv(test_file)
    print(f"\n{test_file} → {result}")

if __name__ == "__main__":
    main()

