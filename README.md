# Pipeline Overview
## Supervised Pipeline
1. Text feature (EventTemplate): Vectorization using TF-IDF
2. One-Hot Encoding
3. Combine features
4. Classification: SVM
5. Evaluation: Precision, Recall, F1-score

## Unsupervised Pipeline
1. Text feature (EventTemplate): Vectorization using TF-IDF
2. Dimension Reduction: SVD
2. One-Hot Encoding
3. Combine feature
4. Anomaly Detection: Isolation Forest
5. Evaluation: 

# Running MLflow
```sh
mlflow models serve \
  -m runs:/{runs_id}/model \
  -p 1234 \
  --host 0.0.0.0 \
  --env-manager local
```