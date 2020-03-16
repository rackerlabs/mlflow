import json
import logging
import os
import urllib
import docker
import mlflow.tracking as tracking
from mlflow.tracking import _get_store
from mlflow.projects import run
import shutil
import tempfile
import tarfile
import gzip
import boto3
import sagemaker as aws_sm
from mlflow.exceptions import ExecutionException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.entities import RunStatus
from mlflow.utils.codebuild_tags import S3_BUCKET_NAME, CODEBUILD_STAGE, CODEBUILD_NO, MODEL_COMMIT, \
    CANONICAL_MODEL_NAME, \
    TRAINING_JOB_NAME, CODEBUILD_INHERENT_NAMES
from mlflow.utils.mlflow_tags import (
    MLFLOW_PROJECT_ENV,
    MLFLOW_PROJECT_BACKEND,
)

_logger = logging.getLogger(__name__)


def train_sagemaker():
    import os

    print("Running train sagemaker...")
    for root, dirs, files in os.walk("/opt/ml/"):
        path = root.split(os.sep)
        print((len(path) - 1) * "---", os.path.basename(root))
        for file in files:
            print(len(path) * "---", file)
    print('-------')
    print('tree output')
    print('-------')
    os.cmd('tree -a /opt/ml')
    print('-------')
    print('Running mlflow')
    print('-------')
    with open(os.path.join('/opt/ml/input/config/', 'hyperparameters.json')) as json_file:
        hyperparameters = json.load(json_file)
    run(uri='/opt/ml/code', parameters=hyperparameters, run_id=hyperparameters['run_id'],
        experiment_id=hyperparameters['experiment_id'])
    return "Success"


def _compress_upload_project_to_s3(bucket_name, prefix):
    client = boto3.client('s3')
    transfer = boto3.s3.transfer.S3Transfer(client=client)
    s3_source_code_key = '{}//sourcedir.tar.gz'.format(prefix)
    target_archive_path = os.path.join(os.getcwd(), 'sourcedir.tar.gz')
    _create_code_archive(os.path.realpath('.'), target_archive_path, 'model')
    transfer.upload_file(filename=target_archive_path,
                         bucket=bucket_name, key=s3_source_code_key,
                         extra_args={'ServerSideEncryption': 'AES256'})
    code_channel_config = {
        "ChannelName": "code",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": 's3://{}/{}'.format(bucket_name, s3_source_code_key),
                "S3DataDistributionType": "FullyReplicated"
            }
        }
    }
    return code_channel_config


def _compress_upload_project_to_s3_codebuild(pipeline_bucket, canonical_name, codebuild_stage,
                                             codebuild_no, commit_id_short):
    client = boto3.client('s3')
    transfer = boto3.s3.transfer.S3Transfer(client=client)
    s3_source_code_key = '{}/{}/{}-{}/sourcedir.tar.gz'.format(canonical_name, codebuild_stage,
                                                               codebuild_no, commit_id_short)
    target_archive_path = os.path.join(os.getcwd(), 'sourcedir.tar.gz')
    _create_code_archive(os.path.realpath('.'), target_archive_path, canonical_name)
    transfer.upload_file(filename=target_archive_path,
                         bucket=pipeline_bucket, key=s3_source_code_key,
                         extra_args={'ServerSideEncryption': 'AES256'})
    code_channel_config = {
        "ChannelName": "code",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": 's3://{}/{}'.format(pipeline_bucket, s3_source_code_key),
                "S3DataDistributionType": "FullyReplicated"
            }
        }
    }
    return code_channel_config


def _create_code_archive(work_dir, result_path, model_name):
    """
    Creates build context tarfile containing Dockerfile and project code, returning path to tarfile
    """
    directory = tempfile.mkdtemp()
    try:
        contents = "mlflow-project-contents"
        archive = "{}-archive".format(model_name)
        dst_path = os.path.join(directory, contents)
        _logger.info('Copying files %s', dst_path)
        shutil.copytree(src=work_dir, dst=dst_path)

        _logger.info('Making compressed source file at %s', result_path)
        _make_tarfile(
            output_filename=result_path,
            source_dir=dst_path, archive_name=archive)
        _logger.info('Successfully generated compressed source file at %s', result_path)
    except Exception as e:
        _logger.fatal('Error occurred during creation of the compressed source file. Error: %s', str(e))
    finally:
        shutil.rmtree(directory)
    return result_path


def _make_tarfile(output_filename, source_dir, archive_name, custom_filter=None):
    # Helper for filtering out modification timestamps
    def _filter_timestamps(tar_info):
        tar_info.mtime = 0
        return tar_info if custom_filter is None else custom_filter(tar_info)

    unzipped_filename = tempfile.mktemp()
    try:
        with tarfile.open(unzipped_filename, "w") as tar:
            tar.add(source_dir, arcname=archive_name, filter=_filter_timestamps)
        # When gzipping the tar, don't include the tar's filename or modification time in the
        # zipped archive (see https://docs.python.org/3/library/gzip.html#gzip.GzipFile)
        with gzip.GzipFile(filename="", fileobj=open(output_filename, 'wb'), mode='wb', mtime=0) \
                as gzipped_tar, open(unzipped_filename, 'rb') as tar:
            gzipped_tar.write(tar.read())
    finally:
        os.remove(unzipped_filename)


# def run_sagemaker_training_job(uri, work_dir, experiment_id, run_id, sagemaker_config):
def run_sagemaker_training_job(sagemaker_config, uri, experiment_id, run_id, work_dir, project, synchronous):
    job_runner = SagemakerCodeBuildJobRunner(sagemaker_config, experiment_id, run_id, uri, work_dir, project)
    job_runner.set_up_tags()
    job_runner.prep_code()
    job_runner.run()

    return SagemakerSubmittedRun(sagemaker_config, experiment_id, run_id, uri, work_dir, project, synchronous)
    # env_vars = {
    #     tracking._TRACKING_URI_ENV_VAR: tracking_uri,
    #     tracking._EXPERIMENT_ID_ENV_VAR: experiment_id,
    # }


def _parse_s3_uri(uri):
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3":
        raise Exception("Not an S3 URI: %s" % uri)
    path = parsed.path
    if path.startswith('/'):
        path = path[1:]
    return parsed.netloc, path


class SagemakerRunner(object):
    def __init__(self, mlflow_experiment_id, mlflow_run_id, sagemaker_config, uri, work_dir, project):
        self._mlflow_run_id = mlflow_run_id
        self._mlflow_experiment_id = mlflow_experiment_id
        self._uri = uri
        self._work_dir = work_dir
        self._project = project
        self.sagemaker_config = sagemaker_config
        self.training_job_name = sagemaker_config[TRAINING_JOB_NAME]

    def set_up_tags(self):
        self._setup_mlflow_tags()
        self._set_up_training_job_tags()

    def _set_up_training_job_tags(self):
        self.sagemaker_config['Tags'].append({'Key': 'Uri', 'Value': self.uri})
        self.sagemaker_config['Tags'].append({'RunId': 'Uri', 'Value': self.run_id})
        self.sagemaker_config['Tags'].append({'ExperimentId': 'Uri', 'Value': self.experiment_id})
        self.sagemaker_config['Tags'].append({'WorkDir': 'Uri', 'Value': self.work_dir})
        self.sagemaker_config['Tags'].append({'WorkDir': 'Project', 'Value': self.project})
        self.sagemaker_config['HyperParameters']['run_id'] = self.run_id
        self.sagemaker_config['HyperParameters']['experiment_id'] = self.experiment_id

    def run(self):
        client = boto3.client("sagemaker")
        response = client.create_training_job(**self.sagemaker_config)
        print(type(response))

    def _setup_mlflow_tags(self):
        tracking.MlflowClient().set_tag(
            self._mlflow_run_id, MLFLOW_PROJECT_ENV, "conda"
        )
        tracking.MlflowClient().set_tag(
            self._mlflow_run_id, MLFLOW_PROJECT_BACKEND, "sagemaker"
        )
        for tag in self.sagemaker_config['Tags']:
            key = tag['Key']
            value = tag['Value']
            tracking.MlflowClient().set_tag(
                self._mlflow_run_id, key, value
            )


class SagemakerCodeBuildJobRunner(SagemakerRunner):
    def __init__(self, mlflow_experiment_id, mlflow_run_id, sagemaker_config, uri, work_dir, project):
        super(SagemakerCodeBuildJobRunner).__init__(mlflow_experiment_id, mlflow_run_id, sagemaker_config, uri,
                                                    work_dir,
                                                    project)

    def set_up_tags(self):
        super(SagemakerCodeBuildJobRunner).set_up_tags()
        self._set_up_codebuild_tags()

    def _set_up_codebuild_tags(self):
        pipeline_bucket = self.sagemaker_config[S3_BUCKET_NAME]
        codebuild_stage = self.sagemaker_config[CODEBUILD_STAGE]
        codebuild_no = self.sagemaker_config[CODEBUILD_NO]
        commit_id_short = self.sagemaker_config[MODEL_COMMIT]
        canonical_name = self.sagemaker_config[CANONICAL_MODEL_NAME]

        for name in CODEBUILD_INHERENT_NAMES:
            value = self.sagemaker_config[name]
            tracking.MlflowClient().set_tag(
                self._mlflow_run_id, name, value
            )
            del self.sagemaker_config[name]

    def prep_code(self):
        store = _get_store()
        artifact_uri = store.get_run(self._mlflow_run_id).info.artifact_uri
        bucket_name, prefix = _parse_s3_uri(artifact_uri)
        code_channel_config = _compress_upload_project_to_s3(bucket_name, prefix)
        self.sagemaker_config["InputDataConfig"].append(code_channel_config)
        print("Sagemaker config: {sagemaker_config}")


class SagemakerSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Kubernetes Job run launched to run an MLflow
    project.
    :param mlflow_run_id: ID of the MLflow project run.
    :param job_name: Sagemaker job name.
    :param job_namespace: Sagemaker job namespace.
    """

    def __init__(self, mlflow_experiment_id, mlflow_run_id, sagemaker_config, uri, work_dir, project, synchronous):
        self._mlflow_run_id = mlflow_run_id
        self._mlflow_experiment_id = mlflow_experiment_id
        self._uri = uri
        self._work_dir = work_dir
        self._project = project
        self.sagemaker_config = sagemaker_config
        self.synchronous = synchronous
        self.training_job_name = sagemaker_config[TRAINING_JOB_NAME]

        super(SagemakerSubmittedRun, self).__init__()

    def run_id(self):
        return self._mlflow_run_id

    def wait(self):
        sagemaker_session = aws_sm.Session()
        sagemaker_session.logs_for_job(self.training_job_name, wait=self.synchronous, log_type='All')
