import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC, SVC

from config import (
    TFIDF_CONFIG,
    MODEL_CONFIG,
    EXPERIMENT_NAME,
    TRACKING_URI
)

# Load data
train_df = pd.read_csv("data/split/train_enhanced.csv")

TEXT_FEATURE = "EventTemplate"
CATEGORICAL_FEATURES = ["Component", "EventId", "time_of_day"]

X_train = train_df[[TEXT_FEATURE] + CATEGORICAL_FEATURES].copy()
# X_train['is_weekend'] = X_train['is_weekend'].astype(float)
y_train = train_df["Label"]

preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(**TFIDF_CONFIG), TEXT_FEATURE),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
            CATEGORICAL_FEATURES
        ),
    ]
)


# Pipeline LinearSVC
pipeline = Pipeline(
    steps=[
        ("features", preprocessor),
        ("clf", SVC(**MODEL_CONFIG)),
    ]
)

# MLflow config
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="train_SVC"):
    pipeline.fit(X_train, y_train)

    vectorizer = pipeline.named_steps["features"].named_transformers_["text"]
    vocab = vectorizer.vocabulary_
    vocab_df = pd.DataFrame(list(vocab.items()), columns=["token", "index"]).sort_values("index")
    vocab_df.to_csv("tfidf_vocab.csv", index=False)

    # Get model type
    model_name = pipeline.named_steps["clf"].__class__.__name__

    # Tag
    mlflow.set_tag("model_type", model_name)

    # Log TF-IDF config + model config
    mlflow.log_params(TFIDF_CONFIG)
    mlflow.log_params(MODEL_CONFIG)
    mlflow.log_param("tfidf_vocab_size", len(vocab))

    # ===== Log vocab artifact =====  

    # mlflow.log_artifact("tfidf_vocab.csv", artifact_path="tfidf")

    # Log model
    mlflow.sklearn.log_model(
        pipeline,
        name="model",
        input_example=X_train.iloc[:5],
        signature=mlflow.models.infer_signature(
            X_train.iloc[:5],
            pipeline.predict(X_train.iloc[:5])
        )
    )