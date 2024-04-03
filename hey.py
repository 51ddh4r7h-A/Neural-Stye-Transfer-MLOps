import mlflow

# Define the run_id of the MLflow run containing the model artifacts
run_id = "d3b22973d11e4765a60d82a68edca4d7"

# Download model artifacts locally
local_model_path = "local_model"
mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=local_model_path)
