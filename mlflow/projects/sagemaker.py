import logging
import docker
from mlflow import tracking

import boto3

from mlflow.exceptions import ExecutionException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.entities import RunStatus

_logger = logging.getLogger(__name__)


def train_sagemaker():
    import os

    print("Running train sagemaker...")
    for root, dirs, files in os.walk("/opt/ml/"):
        path = root.split(os.sep)
        print((len(path) - 1) * "---", os.path.basename(root))
        for file in files:
            print(len(path) * "---", file)
    return "Success"


def _upload_project_to_s3(project_dir, experiment_id, s3_uri):
    import os
    from urllib.parse import urlparse

    print(os.listdir(project_dir))
    print(s3_uri)
    print(experiment_id)
    s3_parsed_uri = urlparse(s3_uri)

    s3_client = boto3.client("s3")

    def upload_objects(bucket_name, project_dir, prefix):
        try:
            s3_resource = boto3.resource("s3", region_name="us-east-1")
            bucket_name = bucket_name  # s3 bucket name
            root_path = project_dir  # local folder for upload

            print(f"Rootpath: {root_path}")

            bucket = s3_resource.Bucket(bucket_name)

            for path, subdirs, files in os.walk(root_path):
                path = path.replace("\\", "/")
                directory_name = path.replace(root_path, "")
                if "mlruns" not in directory_name:
                    for file in files:
                        bucket.upload_file(
                            os.path.join(path, file), prefix + "/" + file
                        )

        except Exception as err:
            print(err)

    upload_objects(s3_parsed_uri.netloc, project_dir, s3_parsed_uri.path[1:])
    response = s3_client.list_objects(
        Bucket=s3_parsed_uri.netloc, Prefix=s3_parsed_uri.path[1:]
    )

    code_channel_config = {
        "ChannelName": "code",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": s3_uri,
                "S3DataDistributionType": "FullyReplicated",
            }
        },
        "CompressionType": "None",
        "InputMode": "File",
    }
    return code_channel_config


def run_sagemaker_training_job(uri, work_dir, experiment_id, run_id, sagemaker_config):
    s3_artifacts_uri = sagemaker_config["s3_artifact_uri"]
    code_channel_config = _upload_project_to_s3(
        work_dir, experiment_id, s3_artifacts_uri
    )
    # env_vars = {
    #     tracking._TRACKING_URI_ENV_VAR: tracking_uri,
    #     tracking._EXPERIMENT_ID_ENV_VAR: experiment_id,
    # }
    del sagemaker_config["s3_artifact_uri"]
    sagemaker_config["InputDataConfig"].append(code_channel_config)
    print(f"Sagemaker config:{sagemaker_config}")

    client = boto3.client("sagemaker")
    response = client.create_training_job(**sagemaker_config)

    # _logger.info(
    #     "=== Running entry point %s of project %s on Sagemaker ===", entry_point, uri
    # )

    return


class SagemakerSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Kubernetes Job run launched to run an MLflow
    project.
    :param mlflow_run_id: ID of the MLflow project run.
    :param job_name: Sagemaker job name.
    :param job_namespace: Sagemaker job namespace.
    """

    def __init__(self, mlflow_run_id, job_name):
        super(SagemakerSubmittedRun, self).__init__()

    def run_id(self):
        return self._mlflow_run_id
