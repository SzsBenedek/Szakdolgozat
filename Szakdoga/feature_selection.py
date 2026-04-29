import os
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from feature_extraction import extract_statistical, extract_phase_epoch, extract_phase_band

N_SPLITS = 5
TOP_N_OPTIONS = [10, 15, 20]  # kipróbált top N jellemző értékek

FEATURE_FUNCS = {
    'Statistical': extract_statistical,
    'Phase epoch': extract_phase_epoch,
    'Phase band': extract_phase_band
}


def load_data(feature_func, base_dir='data'):
    """Betölti az adatokat a megadott feature extraction függvénnyel."""
    X, y = [], []
    beteg_dir = os.path.join(base_dir, 'beteg')
    egeszseges_dir = os.path.join(base_dir, 'egeszseges')

    for file in os.listdir(beteg_dir):
        if file.endswith('.csv'):
            X.append(feature_func(os.path.join(beteg_dir, file)))
            y.append(1)

    for file in os.listdir(egeszseges_dir):
        if file.endswith('.csv'):
            X.append(feature_func(os.path.join(egeszseges_dir, file)))
            y.append(0)

    return np.array(X), np.array(y)


def kfold_evaluate(model, X, y):
    """5-szörös rétegzett keresztvalidáció."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    accs, bal_accs = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        bal_accs.append(balanced_accuracy_score(y_test, y_pred))

    return np.mean(accs), np.mean(bal_accs)


def select_top_features(X, feature_importances, top_n):
    """Kiválasztja a top N legfontosabb jellemző indexeit."""
    top_n = min(top_n, X.shape[1])
    top_indices = np.argsort(feature_importances)[::-1][:top_n]
    return X[:, top_indices], top_indices


def analyze_svm(X, y):
    """SVM feature selection elemzés."""
    print("\n=== SVM ===")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    # 1. tanítás – összes jellemzővel
    gs = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    gs.fit(X, y)
    base_model = gs.best_estimator_
    base_model.fit(X, y)
    base_acc, base_bal_acc = kfold_evaluate(base_model, X, y)
    print(f"  Összes jellemző ({X.shape[1]}): acc={base_acc:.3f}, bal_acc={base_bal_acc:.3f}")

    # permutation importance
    perm = permutation_importance(base_model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    feature_importances = perm.importances_mean

    # 2. tanítás – top N jellemzőkkel
    for top_n in TOP_N_OPTIONS:
        X_sel, _ = select_top_features(X, feature_importances, top_n)
        gs_sel = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        gs_sel.fit(X_sel, y)
        sel_model = gs_sel.best_estimator_
        sel_acc, sel_bal_acc = kfold_evaluate(sel_model, X_sel, y)
        diff = sel_bal_acc - base_bal_acc
        print(f"  Top {top_n} jellemző: acc={sel_acc:.3f}, bal_acc={sel_bal_acc:.3f} (változás: {diff:+.3f})")


def analyze_rf(X, y):
    """Random Forest feature selection elemzés."""
    print("\n=== Random Forest ===")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # 1. tanítás – összes jellemzővel
    gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    gs.fit(X, y)
    base_model = gs.best_estimator_
    base_model.fit(X, y)
    base_acc, base_bal_acc = kfold_evaluate(base_model, X, y)
    print(f"  Összes jellemző ({X.shape[1]}): acc={base_acc:.3f}, bal_acc={base_bal_acc:.3f}")

    # beépített feature importance
    feature_importances = base_model.feature_importances_

    # 2. tanítás – top N jellemzőkkel
    for top_n in TOP_N_OPTIONS:
        X_sel, _ = select_top_features(X, feature_importances, top_n)
        gs_sel = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        gs_sel.fit(X_sel, y)
        sel_model = gs_sel.best_estimator_
        sel_acc, sel_bal_acc = kfold_evaluate(sel_model, X_sel, y)
        diff = sel_bal_acc - base_bal_acc
        print(f"  Top {top_n} jellemző: acc={sel_acc:.3f}, bal_acc={sel_bal_acc:.3f} (változás: {diff:+.3f})")


class RF_MLP_Combined(BaseEstimator, ClassifierMixin):
    def __init__(self, rf_model, mlp_model):
        self.rf_model = rf_model
        self.mlp_model = mlp_model

    def fit(self, X, y):
        self.rf_model.fit(X, y)
        rf_proba = self.rf_model.predict_proba(X)
        self.mlp_model.fit(rf_proba, y)
        return self

    def predict(self, X):
        rf_proba = self.rf_model.predict_proba(X)
        return self.mlp_model.predict(rf_proba)


def analyze_rf_mlp(X, y):
    """RF+MLP feature selection elemzés."""
    print("\n=== RF+MLP ===")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # 1. tanítás – összes jellemzővel
    gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    gs.fit(X, y)
    rf_best = gs.best_estimator_
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',
                                                         solver='adam', max_iter=1000, random_state=42))
    combined = RF_MLP_Combined(rf_best, mlp)
    base_acc, base_bal_acc = kfold_evaluate(combined, X, y)
    combined.fit(X, y)
    print(f"  Összes jellemző ({X.shape[1]}): acc={base_acc:.3f}, bal_acc={base_bal_acc:.3f}")

    # RF rész feature importance-a
    feature_importances = rf_best.feature_importances_

    # 2. tanítás – top N jellemzőkkel
    for top_n in TOP_N_OPTIONS:
        X_sel, _ = select_top_features(X, feature_importances, top_n)
        gs_sel = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, scoring='accuracy')
        gs_sel.fit(X_sel, y)
        rf_sel = gs_sel.best_estimator_
        mlp_sel = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',
                                                                  solver='adam', max_iter=1000, random_state=42))
        combined_sel = RF_MLP_Combined(rf_sel, mlp_sel)
        sel_acc, sel_bal_acc = kfold_evaluate(combined_sel, X_sel, y)
        diff = sel_bal_acc - base_bal_acc
        print(f"  Top {top_n} jellemző: acc={sel_acc:.3f}, bal_acc={sel_bal_acc:.3f} (változás: {diff:+.3f})")


def analyze_xgb(X, y):
    """XGBoost feature selection elemzés."""
    print("\n=== XGBoost ===")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # 1. tanítás – összes jellemzővel
    gs = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    gs.fit(X, y)
    base_model = gs.best_estimator_
    base_model.fit(X, y)
    base_acc, base_bal_acc = kfold_evaluate(base_model, X, y)
    print(f"  Összes jellemző ({X.shape[1]}): acc={base_acc:.3f}, bal_acc={base_bal_acc:.3f}")

    # beépített feature importance
    feature_importances = base_model.feature_importances_

    # 2. tanítás – top N jellemzőkkel
    for top_n in TOP_N_OPTIONS:
        X_sel, _ = select_top_features(X, feature_importances, top_n)
        gs_sel = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        gs_sel.fit(X_sel, y)
        sel_model = gs_sel.best_estimator_
        sel_acc, sel_bal_acc = kfold_evaluate(sel_model, X_sel, y)
        diff = sel_bal_acc - base_bal_acc
        print(f"  Top {top_n} jellemző: acc={sel_acc:.3f}, bal_acc={sel_bal_acc:.3f} (változás: {diff:+.3f})")


if __name__ == "__main__":
    for feature_name, feature_func in FEATURE_FUNCS.items():
        print(f"\n{'='*60}")
        print(f"FEATURE EXTRACTION: {feature_name}")
        print(f"{'='*60}")

        X, y = load_data(feature_func)
        print(f"Adatok betöltve: {len(X)} minta, {X.shape[1]} jellemző")

        analyze_svm(X, y)
        analyze_rf(X, y)
        analyze_rf_mlp(X, y)
        analyze_xgb(X, y)

    print("\nElemzés kész!")