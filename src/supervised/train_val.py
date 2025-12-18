import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

from sklearn.metrics import (
    classification_report,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score
)

from config import (
    TFIDF_CONFIG,
    MODEL_CONFIG,
    EXPERIMENT_NAME,
    TRACKING_URI
)

# =========================
# Load data
# =========================
train_df = pd.read_csv("data/split/train_enhanced.csv")
val_df = pd.read_csv("data/split/val_enhanced.csv")

TEXT_FEATURE = "EventTemplate"
CATEGORICAL_FEATURES = ["Component", "EventId", "time_of_day"]

X_train = train_df[[TEXT_FEATURE] + CATEGORICAL_FEATURES].copy()
y_train = train_df["Label"]

X_val = val_df[[TEXT_FEATURE] + CATEGORICAL_FEATURES].copy()
y_val = val_df["Label"]

# =========================
# Preprocessor
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(**TFIDF_CONFIG), TEXT_FEATURE),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            CATEGORICAL_FEATURES
        ),
    ]
)

# =========================
# Pipeline
# =========================
pipeline = Pipeline(
    steps=[
        ("features", preprocessor),
        ("clf", SVC(**MODEL_CONFIG)),
    ]
)

# =========================
# MLflow setup
# =========================
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="train_and_validate_SVC"):
    
    # ---------- Train ----------
    pipeline.fit(X_train, y_train)

    # Save TF-IDF vocab
    vectorizer = pipeline.named_steps["features"].named_transformers_["text"]
    vocab = vectorizer.vocabulary_
    vocab_df = pd.DataFrame(list(vocab.items()), columns=["token", "index"]).sort_values("index")
    vocab_df.to_csv("tfidf_vocab.csv", index=False)
    
    # Log model type and configs
    model_name = pipeline.named_steps["clf"].__class__.__name__
    mlflow.set_tag("model_type", model_name)
    mlflow.log_params(TFIDF_CONFIG)
    mlflow.log_params(MODEL_CONFIG)
    mlflow.log_param("tfidf_vocab_size", len(vocab))
    
    # Log vocab artifact
    mlflow.log_artifact("tfidf_vocab.csv", artifact_path="tfidf")
    
    # Log pipeline model
    mlflow.sklearn.log_model(
        pipeline,
        "model",
        input_example=X_train.iloc[:5],
        signature=mlflow.models.infer_signature(
            X_train.iloc[:5],
            pipeline.predict(X_train.iloc[:5])
        )
    )
    
    # ---------- Validate ----------
    y_pred = pipeline.predict(X_val)

    # Metrics
    f1_macro = f1_score(y_val, y_pred, average="macro")
    f1_weighted = f1_score(y_val, y_pred, average="weighted")
    recall_anomaly = recall_score(y_val, y_pred, pos_label=1)
    precision_anomaly = precision_score(y_val, y_pred, pos_label=1)
    accuracy_anomaly = accuracy_score(y_val, y_pred)

    # Log metrics
    mlflow.log_metric("val_f1_macro", f1_macro)
    mlflow.log_metric("val_f1_weighted", f1_weighted)
    mlflow.log_metric("val_recall_anomaly", recall_anomaly)
    mlflow.log_metric("val_precision_anomaly", precision_anomaly)
    mlflow.log_metric("val_accuracy_anomaly", accuracy_anomaly)
    
    # Log classification report
    mlflow.log_text(
        classification_report(y_val, y_pred),
        "val_classification_report.txt"
    )

print("Training and validation complete. All artifacts and metrics logged to MLflow.")
