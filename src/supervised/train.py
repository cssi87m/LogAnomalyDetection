# src/train.py

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import TFIDF_CONFIG, MODEL_CONFIG, EXPERIMENT_NAME, TRACKING_URI

train_df = pd.read_csv("../data/split/train.csv")

X_train = train_df["EventTemplate"].astype(str)
y_train = train_df["Label"]

pipeline = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
        ("clf", LogisticRegression(**MODEL_CONFIG)),
    ]
)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="train"):
    pipeline.fit(X_train, y_train)

    mlflow.log_params(TFIDF_CONFIG)
    mlflow.log_params(MODEL_CONFIG)

    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        input_example=X_train.iloc[:5],
        signature=mlflow.models.infer_signature(
            X_train.iloc[:5],
            pipeline.predict(X_train.iloc[:5])
        )
    )
