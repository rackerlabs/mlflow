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
from mlflow import set_tracking_uri
from mlflow.entities import SourceType
from mlflow.exceptions import ExecutionException, MlflowException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.entities import RunStatus
from mlflow import get_tracking_uri
from mlflow.utils.codebuild_tags import S3_BUCKET_NAME, CODEBUILD_STAGE, CODEBUILD_NO, MODEL_COMMIT, \
    CANONICAL_MODEL_NAME, GIT_REPO_NAME, GIT_REPO_URL, CODEBUILD_URL, \
    TRAINING_JOB_NAME, CODEBUILD_INHERENT_NAMES
from mlflow.utils.mlflow_tags import (
    MLFLOW_PROJECT_ENV,
    MLFLOW_USER,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_GIT_COMMIT,
    MLFLOW_RUN_NAME,
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_REPO_URL,
    MLFLOW_PROJECT_BACKEND,
)

TRAINING_MODE = 'Training'
INFERENCE_MODE = 'Inference'

_logger = logging.getLogger(__name__)


def train_sagemaker():
    import os

    _logger.info("Running training sagemaker...")
    _logger.info('-------')
    _logger.info('Running mlflow')
    _logger.info('-------')

    with open(os.path.join('/opt/ml/input/config/', 'hyperparameters.json')) as json_file:
        hyperparameters = json.load(json_file)
    with open(os.path.join('/opt/ml/input/config/', 'inputdataconfig.json')) as json_file:
        inputdataconfig = json.load(json_file)
    with open(os.path.join('/opt/ml/input/config/', 'trainingjobconfig.json')) as json_file:
        trainingjobconfig = json.load(json_file)
    with open(os.path.join('/opt/ml/input/config/', 'metric-definition-regex.json')) as json_file:
        metric = json.load(json_file)
    with open(os.path.join('/opt/ml/input/config/', 'upstreamoutputdataconfig.json')) as json_file:
        upstreamoutputdataconfig = json.load(json_file)

    os.makedirs('/opt/ml/code', exist_ok=True)
    shutil.unpack_archive('/opt/ml/input/data/code/sourcedir.tar.gz', '/opt/ml/code')
    _logger.info('-------')
    os.system('tree -a -L 4 /opt/ml')
    _logger.info('-------')
    os.system('tree -a /opt/ml/code')
    _logger.info('-------DONE unpacking-------')

    # for root, dirs, files in os.walk("/opt/ml/"):
    #     path = root.split(os.sep)
    #     print((len(path) - 1) * "---", os.path.basename(root))
    #     for file in files:
    #         print(len(path) * "---", file)
    tracking_server_uri = hyperparameters['tracking_uri']
    del hyperparameters['tracking_uri']
    run_id = hyperparameters['run_id']
    del hyperparameters['run_id']
    experiment_id = hyperparameters['experiment_id']
    del hyperparameters['experiment_id']
    _logger.info('Run ID: %s', run_id)
    _logger.info('Experiment ID: %s', experiment_id)
    _logger.info('Hyperparameters: %s', hyperparameters)
    _logger.info('Input Job Config: %s', inputdataconfig)
    _logger.info('Output Data Config: %s', upstreamoutputdataconfig)
    _logger.info('Metric: %s', metric)
    _logger.info('Job Config: %s', trainingjobconfig)

    try:
        with open('/opt/ml/input/data/mode/_run_mode', 'r') as f:
            current_mode = f.read()
    except (FileNotFoundError, IOError):
        current_mode = TRAINING_MODE
    # Setup environment variables
    os.environ['MLFLOW_MODE'] = current_mode
    os.environ['MLFLOW_S3_ENABLE_ENCRYPTION'] = '1'
    os.environ['MLFLOW_TRACKING_URI'] = tracking_server_uri
    os.environ[tracking._EXPERIMENT_ID_ENV_VAR] = experiment_id
    os.environ[tracking._RUN_ID_ENV_VAR] = run_id

    set_tracking_uri(tracking_server_uri)
    if current_mode == INFERENCE_MODE:
        run(uri='/opt/ml/code/archive', parameters=hyperparameters, run_id=run_id, entry_point='inference',
            experiment_id=experiment_id, ignore_duplicate_params=True)
    else:
        run(uri='/opt/ml/code/archive', parameters=hyperparameters, run_id=run_id,
            experiment_id=experiment_id, ignore_duplicate_params=True)
    _logger.info('After the run: %s', run_id)
    os.system('tree -a -L 4 /opt/ml')
    return


def _compress_upload_project_to_s3(work_dir, bucket_name, prefix):
    client = boto3.client('s3')
    transfer = boto3.s3.transfer.S3Transfer(client=client)
    s3_source_code_key = '{}//sourcedir.tar.gz'.format(prefix)
    target_archive_path = os.path.join(os.getcwd(), 'sourcedir.tar.gz')
    _create_code_archive(work_dir, target_archive_path, 'model')
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


def _compress_upload_project_to_s3_codebuild(work_dir, pipeline_bucket, canonical_name, codebuild_stage,
                                             codebuild_no, commit_id_short):
    client = boto3.client('s3')
    transfer = boto3.s3.transfer.S3Transfer(client=client)
    s3_source_code_key = '{}/{}/{}-{}/sourcedir.tar.gz'.format(canonical_name, codebuild_stage,
                                                               codebuild_no, commit_id_short)
    target_archive_path = os.path.join(os.getcwd(), 'sourcedir.tar.gz')
    _create_code_archive(work_dir, target_archive_path, canonical_name)
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
        archive = "archive".format(model_name)
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
    job_runner = SagemakerCodeBuildJobRunner(experiment_id, run_id, sagemaker_config, uri, work_dir, project)
    job_runner.setup_tags()
    job_runner.setup_code_channel()
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

    def setup_tags(self):
        self._setup_mlflow_tags()
        self._setup_training_job_tags()

    def _setup_training_job_tags(self):
        self.sagemaker_config['Tags'].append({'Key': 'RunId', 'Value': self._mlflow_run_id})
        self.sagemaker_config['Tags'].append({'Key': 'ExperimentId', 'Value': self._mlflow_experiment_id})
        self.sagemaker_config['Tags'].append({'Key': 'WorkDir', 'Value': self._work_dir})
        self.sagemaker_config['Tags'].append({'Key': 'Project', 'Value': self._project.name})
        self.sagemaker_config['HyperParameters']['run_id'] = self._mlflow_run_id
        self.sagemaker_config['HyperParameters']['experiment_id'] = self._mlflow_experiment_id
        self.sagemaker_config['HyperParameters']['tracking_uri'] = get_tracking_uri()

    def run(self):
        client = boto3.client("sagemaker")
        with open(os.path.join(self._work_dir, 'final_training_job.json'), "w") as handle:
            json.dump(self.sagemaker_config, handle)
        response = client.create_training_job(**self.sagemaker_config)
        try:
            if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                training_job_arn = response['TrainingJobArn']
                training_job_region = training_job_arn.split(':')[3]
                training_job_name = training_job_arn.split(':')[-1].split('/')[-1]
                training_job_url = 'https://console.aws.amazon.com/sagemaker/home?region={}#/jobs/{}'.format(
                    training_job_region, training_job_name)
                _logger.info('Successfully started the training job %s - %s', training_job_arn, training_job_url)
            else:
                raise MlflowException('Failed to start Sagemaker training job. Response: %s' % response)
        except KeyError as ke:
            raise MlflowException('Failed to start Sagemaker training job. Response: %s' % response)

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

    def setup_code_channel(self):
        store = _get_store()
        artifact_uri = store.get_run(self._mlflow_run_id).info.artifact_uri
        bucket_name, prefix = _parse_s3_uri(artifact_uri)
        code_channel_config = _compress_upload_project_to_s3(self._work_dir, bucket_name, prefix)
        self.sagemaker_config["InputDataConfig"].append(code_channel_config)


class SagemakerCodeBuildJobRunner(SagemakerRunner):

    def setup_tags(self):
        super(SagemakerCodeBuildJobRunner, self).setup_tags()
        self._set_up_codebuild_tags()

    def _set_up_codebuild_tags(self):
        self._pipeline_bucket = self.sagemaker_config[S3_BUCKET_NAME]
        self._codebuild_stage = self.sagemaker_config[CODEBUILD_STAGE]
        self._codebuild_no = self.sagemaker_config[CODEBUILD_NO]
        self._codebuild_url = self.sagemaker_config[CODEBUILD_URL]
        self._commit_id_short = self.sagemaker_config[MODEL_COMMIT]
        self._canonical_name = self.sagemaker_config[CANONICAL_MODEL_NAME]
        self._source_name = self.sagemaker_config[GIT_REPO_NAME]
        self._git_repo_url = self.sagemaker_config[GIT_REPO_URL]


        # Mlflow Tags
        tracking.MlflowClient().set_tag(
            self._mlflow_run_id, MLFLOW_USER, "sagemaker-training"
        )

        tags = {
            MLFLOW_USER: "codebuild",
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.JOB),
            MLFLOW_GIT_COMMIT: self._commit_id_short,
            MLFLOW_RUN_NAME: "{}-{}".format(self._codebuild_stage, self._codebuild_no),
            MLFLOW_GIT_BRANCH: "master",  # TODO: improve this hardcode later
            MLFLOW_SOURCE_NAME: self._source_name,
            MLFLOW_GIT_REPO_URL: self._git_repo_url
        }

        for key, value in tags.items():
            tracking.MlflowClient().set_tag(
                self._mlflow_run_id, key, value
            )

        for name in CODEBUILD_INHERENT_NAMES:
            value = self.sagemaker_config[name]
            tracking.MlflowClient().set_tag(
                self._mlflow_run_id, name, value
            )
            del self.sagemaker_config[name]

    def setup_code_channel(self):
        code_channel_config = _compress_upload_project_to_s3_codebuild(self._work_dir, self._pipeline_bucket,
                                                                       self._canonical_name,
                                                                       self._codebuild_stage,
                                                                       self._codebuild_no, self._commit_id_short)
        self.sagemaker_config["InputDataConfig"].append(code_channel_config)


class SagemakerSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Kubernetes Job run launched to run an MLflow
    project.
    :param mlflow_run_id: ID of the MLflow project run.
    :param job_name: Sagemaker job name.
    :param job_namespace: Sagemaker job namespace.
    """

    def __init__(self, sagemaker_config, mlflow_experiment_id, mlflow_run_id, uri, work_dir, project, synchronous):
        self._mlflow_run_id = mlflow_run_id
        self._mlflow_experiment_id = mlflow_experiment_id
        self._uri = uri
        self._work_dir = work_dir
        self._project = project
        self.sagemaker_config = sagemaker_config
        self.synchronous = synchronous
        self.training_job_name = sagemaker_config[TRAINING_JOB_NAME]

        super(SagemakerSubmittedRun, self).__init__()

    @property
    def run_id(self):
        return self._mlflow_run_id

    def wait(self):
        sagemaker_session = aws_sm.Session()
        sagemaker_session.logs_for_job(self.training_job_name, wait=self.synchronous, log_type='All')
        training_job = sagemaker_session.describe_training_job(job_name=self.training_job_name)
        job_status = training_job['TrainingJobStatus']
        if job_status != 'Completed':
            _logger.error('Job did not complete successfully. Current Status: %s', job_status)
            raise Exception('Job: %s failed with a status - %s', self.training_job_name, job_status)
        return True
