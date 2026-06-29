import dagshub
import matplotlib.pyplot as plt
import mlflow
import optuna
import os
import pandas as pd

from dotenv import load_dotenv
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score
)
from xgboost import XGBClassifier

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "hotel-booking_preprocessing.csv"
)

# ==========================================
# SPLIT DATA
# ==========================================

X = df.drop(
    columns=["is_canceled"]
)

y = df["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# HYPERPARAMETER TUNING  
# ==========================================

def objective(trial):

    params = {

        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            300
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.3
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        "random_state": 42,

        "eval_metric": "logloss"
    }

    model = XGBClassifier(
        **params
    )

    score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=3,
        scoring="accuracy"
    ).mean()

    return score

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=10
)

best_params = study.best_params

# ==========================================
# MLFLOW (MANUAL LOGGING)
# ==========================================

load_dotenv()

DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

if DAGSHUB_TOKEN is None:
    raise Exception("DAGSHUB_TOKEN belum ditemukan.")

dagshub.auth.add_app_token(DAGSHUB_TOKEN)

dagshub.init(
    repo_owner="feryanuar24",
    repo_name="Membangun_Model",
    mlflow=True
)

mlflow.set_experiment("Workflow CI")

# ======================================
# PARAMETER
# ======================================

n_estimators = best_params["n_estimators"]
max_depth = best_params["max_depth"]
learning_rate = best_params["learning_rate"]

print(f"n_estimators: {n_estimators}")
print(f"max_depth: {max_depth}")
print(f"learning_rate: {learning_rate}")

mlflow.log_param(
    "n_estimators",
    n_estimators
)

mlflow.log_param(
    "max_depth",
    max_depth
)

mlflow.log_param(
    "learning_rate",
    learning_rate
)

# ======================================
# CREATE ARTIFACTS FOLDER
# ======================================

MODEL_DIR = Path("artifacts/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# MODEL TRAINING
# ==========================================

model = XGBClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    random_state=42,
    eval_metric="logloss"
)

model.fit(
    X_train,
    y_train
)

mlflow.sklearn.save_model(
    sk_model=model,
    path=str(MODEL_DIR),
    serialization_format="cloudpickle",
    pip_requirements=[
        "mlflow==2.19.0",
        "numpy==2.2.6",
        "pandas==2.2.3",
        "scikit-learn==1.6.1",
        "xgboost==2.1.4"
    ]
)

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    serialization_format="cloudpickle",
    pip_requirements=[
        "mlflow==2.19.0",
        "numpy==2.2.6",
        "pandas==2.2.3",
        "scikit-learn==1.6.1",
        "xgboost==2.1.4"
    ]
)

predictions = model.predict(
    X_test
)

# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Skor Akurasi: {accuracy}")

mlflow.log_metric(
    "accuracy",
    accuracy
)

# ======================================
# CONFUSION MATRIX
# ======================================

cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(6,4))

plt.imshow(cm)

plt.title(
    "Confusion Matrix"
)

plt.colorbar()

plt.savefig(
    "artifacts/confusion_matrix.png"
)

plt.close()

print("Confusion matrix berhasil disimpan ke artifacts/confusion_matrix.png")

mlflow.log_artifact(
    "artifacts/confusion_matrix.png"
)

# ======================================
# CLASSIFICATION REPORT
# ======================================

report = classification_report(
    y_test,
    predictions
)

with open(
    "artifacts/classification_report.txt",
    "w"
) as f:

    f.write(report)

print("Classification report berhasil disimpan ke artifacts/classification_report.txt")

mlflow.log_artifact(
    "artifacts/classification_report.txt"
)

# ======================================
# FEATURE IMPORTANCE
# ======================================

importance = model.feature_importances_

plt.figure(
    figsize=(12,6)
)

plt.bar(
    X.columns,
    importance
)

plt.xticks(
    rotation=90
)

plt.tight_layout()

plt.savefig(
    "artifacts/feature_importance.png"
)

plt.close()

print("Feature importance berhasil disimpan ke artifacts/feature_importance.png")

mlflow.log_artifact(
    "artifacts/feature_importance.png"
)