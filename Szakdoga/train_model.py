from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib

def train_and_evaluate_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    grid_search = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    joblib.dump(best_model, 'svm_model.pkl')

    return acc, bal_acc, cm

def train_and_evaluate_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    joblib.dump(best_model, 'rf_model.pkl')

    return acc, bal_acc, cm

class RF_MLP_Combined(BaseEstimator, ClassifierMixin):
    def __init__(self, rf_model, mlp_model):
        self.rf_model = rf_model
        self.mlp_model = mlp_model

    def predict(self, X):
        # 1️⃣ először a Random Forest probability-jeit számítjuk
        rf_proba = self.rf_model.predict_proba(X)
        # 2️⃣ azt adjuk az MLP-nek bemenetként
        return self.mlp_model.predict(rf_proba)

    def predict_proba(self, X):
        rf_proba = self.rf_model.predict_proba(X)
        return self.mlp_model.predict_proba(rf_proba)


def train_and_evaluate_rf_mlp(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)
    rf_best = grid_search.best_estimator_

    rf_train_proba = rf_best.predict_proba(X_train)
    rf_test_proba = rf_best.predict_proba(X_test)

    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(32, 16),
                      activation='relu',
                      solver='adam',
                      max_iter=1000,
                      random_state=42)
    )

    mlp.fit(rf_train_proba, y_train)

    y_pred = mlp.predict(rf_test_proba)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 🔥 Itt hozzuk létre a kombinált modellt, ami mindkettőt tartalmazza
    combined_model = RF_MLP_Combined(rf_best, mlp)
    joblib.dump(combined_model, 'rf_mlp_model.pkl')

    return acc, bal_acc, cm
def train_and_evaluate_xgb(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        XGBClassifier(
            random_state=42,
            eval_metric='logloss',
        ),
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    joblib.dump(best_model, 'xgb_model.pkl')

    return acc, bal_acc, cm