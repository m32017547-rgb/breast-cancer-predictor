"""Train and compare several models on the Breast Cancer Wisconsin dataset.

Run:  python train_model.py
Outputs: model.joblib, metrics.json
"""
import json

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42

# 1) Data: keep the 10 "mean" features so the app stays simple to use
data = load_breast_cancer(as_frame=True)
X = data.data[[c for c in data.data.columns if c.startswith("mean ")]]
y = data.target  # 0 = malignant, 1 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# 2) Candidate models
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results, fitted = {}, {}

for name, clf in candidates.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
    cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    results[name] = {
        "cv_accuracy": round(float(cv_acc.mean()), 4),
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    fitted[name] = pipe
    print(f"{name:22s} CV={results[name]['cv_accuracy']:.3f}  "
          f"Test={results[name]['test_accuracy']:.3f}  AUC={results[name]['roc_auc']:.3f}")

# 3) Pick the best model by cross-validated accuracy (no peeking at the test set)
best_name = max(results, key=lambda n: results[n]["cv_accuracy"])
best = fitted[best_name]
print(f"\nBest model: {best_name}")

# 4) Feature importance (permutation-free: use coefficients / importances when available)
inner = best.named_steps["model"]
if hasattr(inner, "feature_importances_"):
    imp = inner.feature_importances_
elif hasattr(inner, "coef_"):
    imp = np.abs(inner.coef_[0])
else:
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(best, X_test, y_test, n_repeats=20,
                                 random_state=RANDOM_STATE).importances_mean
importance = {f: round(float(v), 4) for f, v in zip(X.columns, imp)}

joblib.dump({"model": best, "features": list(X.columns),
             "ranges": {c: [float(X[c].min()), float(X[c].max()), float(X[c].median())]
                        for c in X.columns}}, "model.joblib")

with open("metrics.json", "w") as f:
    json.dump({"best_model": best_name, "results": results,
               "feature_importance": importance,
               "n_samples": int(len(X)), "n_features": int(X.shape[1])}, f, indent=2)
print("Saved model.joblib and metrics.json")
