import dagshub
import os

from mlflow.tracking import MlflowClient

dagshub.auth.add_app_token(
    os.getenv("DAGSHUB_TOKEN")
)

dagshub.init(
    repo_owner="feryanuar24",
    repo_name="Membangun_Model",
    mlflow=True
)

client = MlflowClient()

experiment = client.get_experiment_by_name(
    "Hotel Booking - Workflow CI"
)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    max_results=1,
    order_by=["attributes.start_time DESC"]
)

print(runs[0].info.run_id)