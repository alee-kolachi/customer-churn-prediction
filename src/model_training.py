# src/train.py

import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

def train_and_evaluate(X_train, y_train, X_dev, y_dev, model_name='logistic', do_cv=True, show_feature_importance=True, tune_hyperparams=False):
    # === Model Selection ===
    if model_name == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
        model_path = "models/logistic_model.pkl"
    elif model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model_path = "models/random_forest_model.pkl"
    elif model_name == 'xgboost':
        if tune_hyperparams:
            print("🔍 Running hyperparameter tuning for XGBoost...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.01, 0.1],
                'reg_lambda': [1, 1.5, 2]
            }
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            random_search = RandomizedSearchCV(
                estimator=xgb,
                param_distributions=param_grid,
                n_iter=25,
                scoring='f1',
                cv=3,
                verbose=1,
                n_jobs=-1,
                random_state=42
            )
            random_search.fit(X_train, y_train)
            model = random_search.best_estimator_
            print("✅ Best hyperparameters:", random_search.best_params_)
        else:
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model_path = "models/xgboost_model.pkl"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # === Train Model ===
    model.fit(X_train, y_train)

    # === Feature Importance ===
    if show_feature_importance:
        if model_name == 'logistic':
            coeffs = pd.Series(model.coef_[0], index=X_train.columns)
            coeffs.sort_values().plot(kind='barh', figsize=(10, 8), title='Logistic Coefficients')
            plt.tight_layout()
            plt.savefig("reports/logistic_coeffs.png")
            plt.clf()
        elif model_name in ['random_forest', 'xgboost']:
            importances = pd.Series(model.feature_importances_, index=X_train.columns)
            importances.sort_values().plot(kind='barh', figsize=(10, 8), title=f'{model_name.title()} Feature Importance')
            plt.tight_layout()
            plt.savefig(f"reports/{model_name}_feature_importance.png")
            plt.clf()

    # === Predict on Dev Set ===
    y_pred = model.predict(X_dev)
    y_proba = model.predict_proba(X_dev)[:, 1]

    # === Cross-Validation ===
    if do_cv:
        print("\n📊 Running cross-validation for bias-variance estimation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        print(f"Average CV ROC AUC: {mean_score:.4f}")
        print(f"Variance (std): {std_score:.4f}")
    else:
        mean_score = std_score = None

    # === Evaluation ===
    report = classification_report(y_dev, y_pred)
    cm = confusion_matrix(y_dev, y_pred)
    auc = roc_auc_score(y_dev, y_proba)

    # === Save Outputs ===
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    joblib.dump(model, model_path)

    with open("reports/metrics.txt", "w") as f:
        f.write(f"=== Model Used: {model_name} ===\n")
        f.write("=== Classification Report ===\n")
        f.write(report)
        f.write("\n=== Confusion Matrix ===\n")
        f.write(str(cm))
        f.write("\n=== ROC AUC ===\n")
        f.write(str(auc))
        if mean_score:
            f.write("\n=== Cross-Validation ROC AUC ===\n")
            f.write(f"Mean: {mean_score:.4f}\n")
            f.write(f"Std (Variance): {std_score:.4f}\n")

    print(f"✅ {model_name} model trained and saved as {model_path}")
    print("📄 Metrics saved in reports/metrics.txt")
    if show_feature_importance:
        print("📈 Feature importance saved as plot in reports/")
