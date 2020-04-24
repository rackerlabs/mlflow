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
from mlflow import __version__
import sagemaker as aws_sm
from mlflow import set_tracking_uri
from mlflow.entities import SourceType
from mlflow.exceptions import ExecutionException, MlflowException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.entities import RunStatus
from mlflow import get_tracking_uri
from mlflow.utils.codebuild_tags import S3_BUCKET_NAME, CODEBUILD_STAGE, CODEBUILD_NO, MODEL_COMMIT, \
    CANONICAL_MODEL_NAME, GIT_REPO_NAME, GIT_REPO_URL, CODEBUILD_URL, CLUSTER_NAME, MODE, \
    TRAINING_JOB_NAME, CODEBUILD_INHERENT_NAMES
from mlflow.utils.file_utils import TempDir
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

TIMEOUT_SCRIPT = "terminate_script.sh"

SETUP_SCRIPT = "setup.sh"
RUN_SCRIPT = "run.sh"

TRAINING_MODE = 'Training'
INFERENCE_MODE = 'Inference'

_logger = logging.getLogger(__name__)

_EMR_BOOTSTRAP_SETUP_TEMPLATE = "#!/usr/bin/env bash"
"pre_reqs() {"
"    echo \"Installing Pre-requisites\""
"    sudo /usr/bin/easy_install-3.6 pip"
"    sudo /usr/local/bin/pip3 install --ignore-installed configparser==3.5.0 # ll /usr/lib/python2.7/dist-packages/backports/"
"    sudo /usr/local/bin/pip3 install mlflow python-dateutil==2.5.0 pyarrow ipython sklearn tensorflow keras"
"    sudo /usr/local/bin/pip3 install python-dateutil --upgrade"
"    sudo sed -i -e '$a\export PYSPARK_PYTHON=/usr/bin/python3' /etc/spark/conf/spark-env.sh"
""
"    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh"
"    bash ~/miniconda.sh -b -p $HOME/miniconda"
"    sudo yum install git -y"
"    echo -e \"alias python=python3\nalias ll=\\\"ls -alF\\\"\n\" >> /home/hadoop/.bashrc"
"    echo -e \"alias python=python3\nalias ll=\\\"ls -alF\\\"\nexport PATH=\\\"\$PATH:/usr/local/bin/\\\" | sudo tee -a /root/.bashrc"
"    eval \"$(/home/hadoop/miniconda/bin/conda shell.bash hook)\""
"    conda init"
"}"
"export MLFLOW_VERSION={mlflow_version}"
"export PATH=\"$PATH:/usr/local/bin/\""
""
"pre_reqs"
""
"aws s3 cp {source_location} source_dir.tgz"
"aws s3 cp {environment_config_location} var.env"
"{input_tasks}"
"source var.env"
"mkdir -p ~/mlflow-code"
"tar xvf source_dir.tgz -C ~/mlflow-code"
"cd ~/mlflow-code"
"conda create -f ~/mlflow-code/conda.yaml -n $MLFLOW_CONDA_ENV_NAME"

# Copyright 2013 Lyft
# Copyright 2014 Alex Konradi
# Copyright 2015 Yelp and Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: David Marin <dm@davidmarin.org>

# This script is part of mrjob, but can be run as a bootstrap action on
# ANY Elastic MapReduce cluster. Arguments are totally optional.

# This script runs `hadoop job -list` in a loop and considers the cluster
# idle if no jobs are currently running. If the cluster stays idle long
# enough AND we're close enough to the end of an EC2 billing hour, we
# shut down the master node, which kills the cluster.

# By default, we allow an idle time of 15 minutes, and shut down within
# the last 5 minutes of the hour.
_TIMEOUT_SCRIPT = "#!/bin/sh"\
"MAX_SECS_IDLE={max_secs_idle}"
"if [ -z \"$MAX_SECS_IDLE\" ]; then MAX_SECS_IDLE=1800; fi"
""
"MIN_SECS_TO_END_OF_HOUR={min_secs_to_end_of_hour}"
"if [ -z \"$MIN_SECS_TO_END_OF_HOUR\" ]; then MIN_SECS_TO_END_OF_HOUR=300; fi"
""
""
"("
"while true  # the only way out is to SHUT DOWN THE MACHINE"
"do"
"    # get the uptime as an integer (expr can't handle decimals)"
"    UPTIME=$(cat /proc/uptime | cut -f 1 -d .)"
"    SECS_TO_END_OF_HOUR=$(expr 3600 - $UPTIME % 3600)"
""
"    # if LAST_ACTIVE hasn't been initialized, hadoop hasn't been installed"
"    # yet (this happens on 4.x AMIs), or there are jobs running, just set"
"    # LAST_ACTIVE to UPTIME. This also checks yarn application if it"
"    # exists (see #1145)"
"    if [ -z \"$LAST_ACTIVE\" ] || \ "
"        ! which hadoop > /dev/null || \ "
"        nice hadoop job -list 2> /dev/null | grep -q '^\s*job_' || \ "
"        (which yarn > /dev/null && \ "
"            nice yarn application -list 2> /dev/null | \ "
"            grep -v 'Total number' | grep -q RUNNING)"
"    then"
"        LAST_ACTIVE=$UPTIME"
"    else"
"	# the cluster is idle! how long has this been going on?"
"        SECS_IDLE=$(expr $UPTIME - $LAST_ACTIVE)"
""
"        if expr $SECS_IDLE '>' $MAX_SECS_IDLE '&' \ "
"            $SECS_TO_END_OF_HOUR '<' $MIN_SECS_TO_END_OF_HOUR > /dev/null"
"        then"
"            sudo shutdown -h now"
"            exit"
"        fi"
"    fi"
"done"
"# close file handles to daemonize the script; otherwise bootstrapping"
"# never finishes"
") 0<&- &> /dev/null &"

_MLFLOW_RUN_SCRIPT = "#!/usr/bin/env bash"
"aws s3 cp {environment_config_location} var.env"
"mlflow run --ignore-duplicate-parameters ~/mlflow-code/ $MLFLOW_PARSED_PARAMETER"
"{output_tasks}"


def train_emr():

    _logger.info("Running training EMR...")
    _logger.info('-------')
    _logger.info('Running mlflow')
    _logger.info('-------')
    #
    # with open(os.path.join('/opt/ml/input/config/', 'hyperparameters.json')) as json_file:
    #     hyperparameters = json.load(json_file)
    # with open(os.path.join('/opt/ml/input/config/', 'inputdataconfig.json')) as json_file:
    #     inputdataconfig = json.load(json_file)
    # with open(os.path.join('/opt/ml/input/config/', 'trainingjobconfig.json')) as json_file:
    #     trainingjobconfig = json.load(json_file)
    # with open(os.path.join('/opt/ml/input/config/', 'metric-definition-regex.json')) as json_file:
    #     metric = json.load(json_file)
    # with open(os.path.join('/opt/ml/input/config/', 'upstreamoutputdataconfig.json')) as json_file:
    #     upstreamoutputdataconfig = json.load(json_file)
    #
    # os.makedirs('/opt/ml/code', exist_ok=True)
    # shutil.unpack_archive('/opt/ml/input/data/code/sourcedir.tar.gz', '/opt/ml/code')
    # _logger.info('-------')
    # os.system('tree -a -L 4 /opt/ml')
    # _logger.info('-------')
    # os.system('tree -a /opt/ml/code')
    # _logger.info('-------DONE unpacking-------')
    #
    # # for root, dirs, files in os.walk("/opt/ml/"):
    # #     path = root.split(os.sep)
    # #     print((len(path) - 1) * "---", os.path.basename(root))
    # #     for file in files:
    # #         print(len(path) * "---", file)
    # tracking_server_uri = hyperparameters['tracking_uri']
    # del hyperparameters['tracking_uri']
    # run_id = hyperparameters['run_id']
    # del hyperparameters['run_id']
    # experiment_id = hyperparameters['experiment_id']
    # del hyperparameters['experiment_id']
    # _logger.info('Run ID: %s', run_id)
    # _logger.info('Experiment ID: %s', experiment_id)
    # _logger.info('Hyperparameters: %s', hyperparameters)
    # _logger.info('Input Job Config: %s', inputdataconfig)
    # _logger.info('Output Data Config: %s', upstreamoutputdataconfig)
    # _logger.info('Metric: %s', metric)
    # _logger.info('Job Config: %s', trainingjobconfig)
    #
    # try:
    #     with open('/opt/ml/input/data/mode/_run_mode', 'r') as f:
    #         current_mode = f.read()
    # except (FileNotFoundError, IOError):
    #     current_mode = TRAINING_MODE
    # # Setup environment variables
    # os.environ['MLFLOW_MODE'] = current_mode
    # os.environ['MLFLOW_S3_ENABLE_ENCRYPTION'] = '1'
    # os.environ['MLFLOW_TRACKING_URI'] = tracking_server_uri
    # os.environ[tracking._EXPERIMENT_ID_ENV_VAR] = experiment_id
    # os.environ[tracking._RUN_ID_ENV_VAR] = run_id
    #
    # set_tracking_uri(tracking_server_uri)
    # if current_mode == INFERENCE_MODE:
    #     run(uri='/opt/ml/code/src', parameters=hyperparameters, run_id=run_id, entry_point='inference',
    #         experiment_id=experiment_id, ignore_duplicate_params=True)
    # else:
    #     run(uri='/opt/ml/code/src', parameters=hyperparameters, run_id=run_id,
    #         experiment_id=experiment_id, ignore_duplicate_params=True)
    # _logger.info('After the run: %s', run_id)
    # os.system('tree -a -L 4 /opt/ml')
    return


def upload_source_code(work_dir, bucket_name, prefix, model_name='model'):
    client = boto3.client('s3')
    transfer = boto3.s3.transfer.S3Transfer(client=client)
    s3_source_code_key = '{}/sourcedir.tar.gz'.format(prefix)
    target_archive_path = os.path.join(os.getcwd(), 'sourcedir.tar.gz')
    _create_code_archive(work_dir, target_archive_path, model_name)
    transfer.upload_file(filename=target_archive_path,
                         bucket=bucket_name, key=s3_source_code_key,
                         extra_args={'ServerSideEncryption': 'AES256'})
    file_uri = f's3://{bucket_name}/{s3_source_code_key}'
    _logger.info('Uploaded the source code to %s', file_uri)
    return file_uri


def upload_environment(bucket_name, prefix, environment_config='var.env'):
    client = boto3.client('s3')
    transfer = boto3.s3.transfer.S3Transfer(client=client)
    s3_environment_config_key = '{}/var.env'.format(prefix)
    # target_archive_path = os.path.join(os.getcwd(), environment_config)
    # _create_code_archive(work_dir, target_archive_path, model_name)
    transfer.upload_file(filename=environment_config,
                         bucket=bucket_name, key=s3_environment_config_key,
                         extra_args={'ServerSideEncryption': 'AES256'})
    file_uri = f's3://{bucket_name}/{s3_environment_config_key}'
    _logger.info('Uploaded the environment config to %s', file_uri)
    return file_uri


# def _compress_upload_project_to_s3(work_dir, bucket_name, prefix):
#     client = boto3.client('s3')
#     transfer = boto3.s3.transfer.S3Transfer(client=client)
#     s3_source_code_key = '{}//sourcedir.tar.gz'.format(prefix)
#     target_archive_path = os.path.join(os.getcwd(), 'sourcedir.tar.gz')
#     _create_code_archive(work_dir, target_archive_path, 'model')
#     transfer.upload_file(filename=target_archive_path,
#                          bucket=bucket_name, key=s3_source_code_key,
#                          extra_args={'ServerSideEncryption': 'AES256'})
#     _logger.info('Uploaded the source code to %s', 's3://{}/{}'.format(bucket_name, s3_source_code_key))
#     return 's3://{}/{}'.format(bucket_name, s3_source_code_key)
#
#
# def _compress_upload_project_to_s3_codebuild(work_dir, bucket_name, canonical_name, codebuild_stage,
#                                              codebuild_no, commit_id_short):
#     client = boto3.client('s3')
#     transfer = boto3.s3.transfer.S3Transfer(client=client)
#     s3_source_code_key = '{}/{}/{}-{}/sourcedir.tar.gz'.format(canonical_name, codebuild_stage,
#                                                                codebuild_no, commit_id_short)
#     target_archive_path = os.path.join(os.getcwd(), 'sourcedir.tar.gz')
#     _create_code_archive(work_dir, target_archive_path, canonical_name)
#     transfer.upload_file(filename=target_archive_path,
#                          bucket=bucket_name, key=s3_source_code_key,
#                          extra_args={'ServerSideEncryption': 'AES256'})
#     _logger.info('Uploaded the source code to %s', 's3://{}/{}'.format(bucket_name, s3_source_code_key))
#     return 's3://{}/{}'.format(bucket_name, s3_source_code_key)


def _create_code_archive(work_dir, result_path, model_name):
    """
    Creates build context tarfile containing Dockerfile and project code, returning path to tarfile
    """
    directory = tempfile.mkdtemp()
    try:
        contents = "mlflow-project-contents"
        archive = "src"
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


def run_emr_training_job(emr_config, uri, experiment_id, run_id, work_dir, project, synchronous, environment_config):
    job_runner = EmrCodeBuildJobRunner(experiment_id, run_id, emr_config, uri, work_dir, project, environment_config)
    job_runner.setup_tags()
    job_runner.setup_code_and_environment()
    job_runner.run()

    return EmrSubmittedRun(emr_config, experiment_id, run_id, uri, work_dir, project, synchronous)
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


def upload_bootstrap_scripts(bucket_name, prefix, source_location, environment_config_location):
    with TempDir() as tmp:
        cwd = tmp.path()
        setup_script = os.path.join(cwd, SETUP_SCRIPT)
        timeout_script = os.path.join(cwd, TIMEOUT_SCRIPT)
        run_script = os.path.join(cwd, RUN_SCRIPT)
        with open(setup_script, "w") as f:
            f.write(
                _EMR_BOOTSTRAP_SETUP_TEMPLATE.format(
                    source_location=source_location,
                    environment_config_location=environment_config_location,
                    mlflow_version=__version__,
                )
            )

        with open(timeout_script, "w") as f:
            f.write(
                _TIMEOUT_SCRIPT.format(
                    max_secs_idle=300,
                    min_secs_to_end_of_hour=3600,
                )
            )

        with open(run_script, "w") as f:
            f.write(
                _MLFLOW_RUN_SCRIPT.format(
                    environment_config_location=environment_config_location,
                )
            )
        s3_setup_script_key = '{}/{}'.format(prefix, SETUP_SCRIPT)
        s3_timeout_script_key = '{}/{}'.format(prefix, TIMEOUT_SCRIPT)
        s3_run_script_key = '{}/{}'.format(prefix, RUN_SCRIPT)
        client = boto3.client('s3')
        transfer = boto3.s3.transfer.S3Transfer(client=client)

        transfer.upload_file(filename=setup_script,
                             bucket=bucket_name, key=s3_setup_script_key,
                             extra_args={'ServerSideEncryption': 'AES256'})
        setup_script_uri = f's3://{bucket_name}/{s3_setup_script_key}'
        _logger.info('Uploaded the setup script to %s', setup_script_uri)

        transfer.upload_file(filename=timeout_script,
                             bucket=bucket_name, key=s3_timeout_script_key,
                             extra_args={'ServerSideEncryption': 'AES256'})
        timeout_script_uri = f's3://{bucket_name}/{s3_timeout_script_key}'
        _logger.info('Uploaded the timeout script to %s', timeout_script_uri)

        transfer.upload_file(filename=run_script,
                             bucket=bucket_name, key=s3_run_script_key,
                             extra_args={'ServerSideEncryption': 'AES256'})
        run_script_uri = f's3://{bucket_name}/{s3_run_script_key}'
        _logger.info('Uploaded the run script to %s', run_script_uri)

        return setup_script_uri, timeout_script_uri, run_script_uri



class EmrRunner(object):
    def __init__(self, mlflow_experiment_id, mlflow_run_id, emr_config, uri, work_dir, project, environment_config):
        self._mlflow_run_id = mlflow_run_id
        self._mlflow_experiment_id = mlflow_experiment_id
        self._uri = uri
        self._work_dir = work_dir
        self._project = project
        self.environment_config = environment_config
        self.emr_config = emr_config
        self.instance_groups = emr_config['ResourceConfig']['InstanceGroups']
        self.ec2_key_name = emr_config['ResourceConfig'].get('Ec2KeyName', '')
        self.ec2_subnet_id = emr_config['ResourceConfig'].get('Ec2SubnetId')
        self.master_sg = emr_config['ResourceConfig'].get('EmrManagedMasterSecurityGroup')
        self.slave_sg = emr_config['ResourceConfig'].get('EmrManagedSlaveSecurityGroup')
        self.release_label = emr_config['ReleaseLabel']
        self.jobflow_role = emr_config['JobFlowRole']
        self.service_role = emr_config['ServiceRole']
        self.applications = emr_config['Applications']
        self.log_uri = emr_config["LogUri"]
        self.visible_to_all_users = emr_config["VisibleToAllUsers"]
        self.cluster_name = emr_config[CLUSTER_NAME]
        self.mode = emr_config[MODE]
        self.tags = []
        self.session = boto3.Session()
        self.region = self.session.region_name or "us-east-2"
        store = _get_store()
        artifact_uri = store.get_run(self._mlflow_run_id).info.artifact_uri
        self.bucket_name, self.prefix = _parse_s3_uri(artifact_uri)
        self.source_location = upload_source_code(self._work_dir, self.bucket_name, self.prefix)
        self.environment_config_location = upload_environment(self.bucket_name,
                                                              self.prefix, self.environment_config)
        self.setup_bootstrap_script, self.timeout_script, self.run_script = upload_bootstrap_scripts(self.bucket_name, self.prefix,
                                                                                    self.source_location,
                                                                                    self.environment_config_location)

    def setup_tags(self):
        self._setup_mlflow_tags()
        self._setup_cluster_tags()

    def _setup_cluster_tags(self):
        self.tags.append({'Key': 'RunId', 'Value': self._mlflow_run_id})
        self.tags.append({'Key': 'ExperimentId', 'Value': self._mlflow_experiment_id})
        self.tags.append({'Key': 'WorkDir', 'Value': self._work_dir})
        self.tags.append({'Key': 'Project', 'Value': self._project.name})

    def run(self):
        client = self.session.client('emr')
        response = client.run_job_flow(
            Name=self.cluster_name,
            LogUri=self.log_uri,
            ReleaseLabel=self.release_label,
            Instances={
                'InstanceGroups': self.instance_groups,
                'Ec2KeyName': self.ec2_key_name,
                'KeepJobFlowAliveWhenNoSteps': False
            },
            Applications=self.applications,
            JobFlowRole=self.jobflow_role,
            ServiceRole=self.service_role,
            VisibleToAllUsers=self.visible_to_all_users,
            BootstrapActions=[
                {
                    'Name': 'setup',
                    'ScriptBootstrapAction': {
                        'Path': self.setup_bootstrap_script
                    }
                },
                {
                    'Name': 'idle timeout',
                    'ScriptBootstrapAction': {
                        'Path': self.timeout_script
                    }
                },
                {
                    'Name': 'Maximize Spark Default Config',
                    'ScriptBootstrapAction': {
                        'Path': 's3://support.elasticmapreduce/spark/maximize-spark-default-config',
                    }
                },
            ],
            Steps=[
                {
                    'Name': 'MLFlow {mode}'.format(mode=self.mode),
                    'ActionOnFailure': 'TERMINATE_CLUSTER',
                    'HadoopJarStep': {
                        'Jar': 's3://{region}.elasticmapreduce/libs/script-runner/script-runner.jar'
                            .format(region=self.region),
                        'Args': [self.run_script]
                        # 'Jar': 'command-runner.jar',
                        # 'Args': [
                            # "spark-submit",
                            # "python app",
                            # arguments
                        # ]
                    }
                },
            ]
        )
        response_code = response['ResponseMetadata']['HTTPStatusCode']
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            self.job_flow_id = response['JobFlowId']
            _logger.info('Successfully started the EMR cluster Id: %s', self.job_flow_id)
        else:
            _logger.error('Failed to start the EMR cluster Id. Code: %d Response: %s',
                          response['ResponseMetadata']['HTTPStatusCode'], response)

        # client = boto3.client("sagemaker")
        # with open(os.path.join(self._work_dir, 'final_training_job.json'), "w") as handle:
        #     json.dump(self.sagemaker_config, handle)
        # response = client.create_training_job(**self.sagemaker_config)
        # try:
        #     if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        #         training_job_arn = response['TrainingJobArn']
        #         training_job_region = training_job_arn.split(':')[3]
        #         training_job_name = training_job_arn.split(':')[-1].split('/')[-1]
        #         training_job_url = 'https://console.aws.amazon.com/sagemaker/home?region={}#/jobs/{}'.format(
        #             training_job_region, training_job_name)
        #         _logger.info('Successfully started the training job %s - %s', training_job_arn, training_job_url)
        #     else:
        #         raise MlflowException('Failed to start Sagemaker training job. Response: %s' % response)
        # except KeyError as ke:
        #     raise MlflowException('Failed to start Sagemaker training job. Response: %s' % response)

    def _setup_mlflow_tags(self):
        tracking.MlflowClient().set_tag(
            self._mlflow_run_id, MLFLOW_PROJECT_ENV, "conda"
        )
        tracking.MlflowClient().set_tag(
            self._mlflow_run_id, MLFLOW_PROJECT_BACKEND, "sagemaker"
        )
        for tag in self.emr_config['Tags']:
            key = tag['Key']
            value = tag['Value']
            tracking.MlflowClient().set_tag(
                self._mlflow_run_id, key, value
            )

    def setup_code_and_environment(self):
        pass


class EmrCodeBuildJobRunner(EmrRunner):

    def __init__(self, mlflow_experiment_id, mlflow_run_id, emr_config, uri, work_dir, project, environment_config):
        super(EmrCodeBuildJobRunner, self).__init__(mlflow_experiment_id, mlflow_run_id, emr_config, uri, work_dir,
                                                    project, environment_config)

        # Override Prefix and Bucket Name
        self.prefix = '{}/{}/{}-{}'.format(self._canonical_name, self._codebuild_stage,
                                           self._codebuild_no, self._commit_id_short)
        self.bucket_name = self.emr_config[S3_BUCKET_NAME]

    def setup_tags(self):
        super(EmrCodeBuildJobRunner, self).setup_tags()
        self._fetch_codebuild_tags()

    def _fetch_codebuild_tags(self):
        self._pipeline_bucket = self.emr_config[S3_BUCKET_NAME]
        self._codebuild_stage = self.emr_config[CODEBUILD_STAGE]
        self._codebuild_no = self.emr_config[CODEBUILD_NO]
        self._codebuild_url = self.emr_config[CODEBUILD_URL]
        self._commit_id_short = self.emr_config[MODEL_COMMIT]
        self._canonical_name = self.emr_config[CANONICAL_MODEL_NAME]
        self._source_name = self.emr_config[GIT_REPO_NAME]
        self._git_repo_url = self.emr_config[GIT_REPO_URL]

        # Mlflow Tags
        tracking.MlflowClient().set_tag(
            self._mlflow_run_id, MLFLOW_USER, "emr"
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
            value = self.emr_config[name]
            tracking.MlflowClient().set_tag(
                self._mlflow_run_id, name, value
            )
            del self.emr_config[name]


class EmrSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Kubernetes Job run launched to run an MLflow
    project.
    :param mlflow_run_id: ID of the MLflow project run.
    :param job_name: Sagemaker job name.
    :param job_namespace: Sagemaker job namespace.
    """

    def __init__(self, emr_config, mlflow_experiment_id, mlflow_run_id, uri, work_dir, project, synchronous):
        self._mlflow_run_id = mlflow_run_id
        self._mlflow_experiment_id = mlflow_experiment_id
        self._uri = uri
        self._work_dir = work_dir
        self._project = project
        self.emr_config = emr_config
        self.synchronous = synchronous
        self.cluster_name = emr_config[CLUSTER_NAME]
        super(EmrSubmittedRun, self).__init__()

    @property
    def run_id(self):
        return self._mlflow_run_id

    def wait(self):
        # sagemaker_session = aws_sm.Session()
        # sagemaker_session.logs_for_job(self.training_job_name, wait=self.synchronous, log_type='All')
        # training_job = sagemaker_session.describe_training_job(job_name=self.training_job_name)
        # job_status = training_job['TrainingJobStatus']

        job_status = ''
        if job_status != 'Completed':
            _logger.error('Cluster did not complete jobs successfully. Current Status: %s', job_status)
            raise Exception('Job: %s failed with a status - %s', self.cluster_name, job_status)
        return True
