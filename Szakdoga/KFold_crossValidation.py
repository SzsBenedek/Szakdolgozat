from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from xgboost import XGBClassifier
def evaluate_random_forest(X, y, n_estimators=200, random_state=42, n_splits=5):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    balanced_scores = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy')

    print(f"Pontosság fold-önként:           {np.round(scores, 3)}")
    print(f"Átlagos pontosság:              {np.mean(scores):.3f}")
    print(f"Balanced accuracy fold-önként:  {np.round(balanced_scores, 3)}")
    print(f"Átlagos balanced accuracy:      {np.mean(balanced_scores):.3f}")

    return np.mean(scores), np.mean(balanced_scores)
def evaluate_xgboost(X, y, random_state=42, n_splits=10):
    clf = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric='logloss'
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    balanced_scores = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy')

    print("=== XGBoost Cross-Validation ===")
    print(f"Pontosság fold-önként:           {np.round(scores, 3)}")
    print(f"Átlagos pontosság:              {np.mean(scores):.3f}")
    print(f"Balanced accuracy fold-önként:  {np.round(balanced_scores, 3)}")
    print(f"Átlagos balanced accuracy:      {np.mean(balanced_scores):.3f}")

    return np.mean(scores), np.mean(balanced_scores)


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Tesztadat generálása
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    print("=== Random Forest Cross-Validation Példa ===")
    rf_acc, rf_bal = evaluate_random_forest(X, y)

    print("=== XGBoost Cross-Validation Példa ===")
    xgb_acc, xgb_bal = evaluate_xgboost(X, y)

    print("=== Összegzés ===")
    print(f"Random Forest - Pontosság: {rf_acc:.3f}, Balanced acc: {rf_bal:.3f}")
    print(f"XGBoost - Pontosság:       {xgb_acc:.3f}, Balanced acc: {xgb_bal:.3f}")