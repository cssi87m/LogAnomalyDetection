TFIDF_CONFIG = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 2,
}

MODEL_CONFIG = {
    "C": 1.0,
    "loss": "squared_hinge",
    "class_weight": "balanced",
    "max_iter": 5000
}

EXPERIMENT_NAME = "log_tfidf_pipeline"
TRACKING_URI = "http://127.0.0.1:5000"
