import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score, f1_score
from config import TRACKING_URI

test_df = pd.read_csv("../data/split/test.csv")

X_test = test_df["EventTemplate"].astype(str)
y_test = test_df["Label"]

BEST_RUN_ID = "<BEST_RUN_ID>"

model = mlflow.sklearn.load_model(
    f"runs:/{BEST_RUN_ID}/model"
)

y_pred = model.predict(X_test)

print("TEST accuracy:", accuracy_score(y_test, y_pred))
print("TEST f1:", f1_score(y_test, y_pred))
