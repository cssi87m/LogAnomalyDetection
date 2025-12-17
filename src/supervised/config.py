# src/config.py

TFIDF_CONFIG = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 2,
}

MODEL_CONFIG = {
    "max_iter": 1000,
    "n_jobs": -1,
}

EXPERIMENT_NAME = "log_tfidf_pipeline"
TRACKING_URI = "http://127.0.0.1:5000"
