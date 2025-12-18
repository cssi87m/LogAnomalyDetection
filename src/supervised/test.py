import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    classification_report
)

from config import TRACKING_URI

# =========================
# Config
# =========================
RUN_ID = os.environ.get("RUN_ID", "")

TEXT_FEATURE = "EventTemplate"
CATEGORICAL_FEATURES = ["Component", "EventId", "time_of_day"]

# =========================
# Load test data
# =========================
test_df = pd.read_csv("data/split/test_enhanced.csv")

X_test = test_df[[TEXT_FEATURE] + CATEGORICAL_FEATURES].copy()
X_test[TEXT_FEATURE] = X_test[TEXT_FEATURE].astype(str)

y_test = test_df["Label"]

# =========================
# Load model from MLflow
# =========================
mlflow.set_tracking_uri(TRACKING_URI)

pipeline = mlflow.sklearn.load_model(
    f"runs:/{RUN_ID}/model"
)

# =========================
# Predict
# =========================
y_pred = pipeline.predict(X_test)

# =========================
# Metrics (ANOMALY-CENTRIC)
# =========================
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

recall_anomaly = recall_score(y_test, y_pred, pos_label=1)
precision_anomaly = precision_score(y_test, y_pred, pos_label=1)

# =========================
# Output
# =========================
print("TEST f1_macro:", f1_macro)
print("TEST f1_weighted:", f1_weighted)
print("TEST recall_anomaly:", recall_anomaly)
print("TEST precision_anomaly:", precision_anomaly)

print("\nClassification report:")
print(classification_report(y_test, y_pred))
