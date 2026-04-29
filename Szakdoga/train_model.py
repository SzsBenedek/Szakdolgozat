from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import joblib

N_SPLITS = 5

def kfold_evaluate(model, X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    accs, bal_accs, cms = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        bal_accs.append(balanced_accuracy_score(y_test, y_pred))
        cms.append(confusion_matrix(y_test, y_pred))

    avg_cm = np.mean(cms, axis=0).astype(int)
    return np.mean(accs), np.mean(bal_accs), avg_cm


def train_and_evaluate_svm(X, y):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    grid_search = GridSearchCV(
        SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    acc, bal_acc, cm = kfold_evaluate(best_model, X, y)

    # teljes adaton tanítjuk be mentés előtt
    best_model.fit(X, y)
    joblib.dump(best_model, 'svm_model.pkl')

    return acc, bal_acc, cm


def train_and_evaluate_rf(X, y):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, n_jobs=-1, scoring='accuracy'
    )
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    acc, bal_acc, cm = kfold_evaluate(best_model, X, y)

    best_model.fit(X, y)
    joblib.dump(best_model, 'rf_model.pkl')

    return acc, bal_acc, cm


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

    def predict_proba(self, X):
        rf_proba = self.rf_model.predict_proba(X)
        return self.mlp_model.predict_proba(rf_proba)


def train_and_evaluate_rf_mlp(X, y):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=3, n_jobs=-1, scoring='accuracy'
    )
    grid_search.fit(X, y)
    rf_best = grid_search.best_estimator_

    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',
                      solver='adam', max_iter=1000, random_state=42)
    )

    combined_model = RF_MLP_Combined(rf_best, mlp)
    acc, bal_acc, cm = kfold_evaluate(combined_model, X, y)

    # teljes adaton tanítjuk be mentés előtt
    combined_model.fit(X, y)
    joblib.dump(combined_model, 'rf_mlp_model.pkl')

    return acc, bal_acc, cm


def train_and_evaluate_xgb(X, y):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        param_grid, cv=5, n_jobs=-1, scoring='accuracy'
    )
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    acc, bal_acc, cm = kfold_evaluate(best_model, X, y)

    best_model.fit(X, y)
    joblib.dump(best_model, 'xgb_model.pkl')

    return acc, bal_acc, cm