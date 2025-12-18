import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    classification_report,
    f1_score,
    recall_score,
    precision_score
)

from config import (
    EXPERIMENT_NAME,
    TRACKING_URI
)

RUN_ID = os.environ.get("RUN_ID", "")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# =========================
# Load model
# =========================
model_uri = f"runs:/{RUN_ID}/model"
pipeline = mlflow.sklearn.load_model(model_uri)

# =========================
# Load val data
# =========================
val_df = pd.read_csv("data/split/val_enhanced.csv")

TEXT_FEATURE = "EventTemplate"
CATEGORICAL_FEATURES = ["Component", "EventId", "time_of_day"]

X_val = val_df[[TEXT_FEATURE] + CATEGORICAL_FEATURES].copy()

y_val = val_df["Label"]

# =========================
# Predict
# =========================
y_pred = pipeline.predict(X_val)

# =========================
# Metrics (ANOMALY-CENTRIC)
# =========================
f1_macro = f1_score(y_val, y_pred, average="macro")
f1_weighted = f1_score(y_val, y_pred, average="weighted")

recall_anomaly = recall_score(y_val, y_pred, pos_label=1)
precision_anomaly = precision_score(y_val, y_pred, pos_label=1)

# =========================
# Log to MLflow
# =========================
with mlflow.start_run(run_name="val_linearSVC", nested=True):
    mlflow.log_metric("val_f1_macro", f1_macro)
    mlflow.log_metric("val_f1_weighted", f1_weighted)
    mlflow.log_metric("val_recall_anomaly", recall_anomaly)
    mlflow.log_metric("val_precision_anomaly", precision_anomaly)

    mlflow.log_text(
        classification_report(y_val, y_pred),
        "val_classification_report.txt"
    )
