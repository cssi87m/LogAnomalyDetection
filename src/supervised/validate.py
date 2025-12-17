import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score, f1_score
from config import TRACKING_URI

val_df = pd.read_csv("../data/split/val.csv")

X_val = val_df["EventTemplate"].astype(str)
y_val = val_df["Label"]

RUN_ID = "<TRAIN_RUN_ID>"

model = mlflow.sklearn.load_model(
    f"runs:/{RUN_ID}/model"
)

mlflow.set_tracking_uri(TRACKING_URI)

with mlflow.start_run(run_id=RUN_ID):
    y_pred = model.predict(X_val)

    mlflow.log_metric("val_accuracy", accuracy_score(y_val, y_pred))
    mlflow.log_metric("val_f1", f1_score(y_val, y_pred))
