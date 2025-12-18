TFIDF_CONFIG = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 2,
}

MODEL_CONFIG = {
    "C": 1.0,          # Regularization
    "kernel": "rbf",   # Kernel type
    "class_weight": "balanced",    
}

EXPERIMENT_NAME = "Anomaly Detection Pipeline"
TRACKING_URI = "http://127.0.0.1:5000"
