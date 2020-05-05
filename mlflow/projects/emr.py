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
from botocore.exceptions import ClientError
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

TIMEOUT_SCRIPT = "timeout.sh"
SOURCE_ROOT_DIR = "/mnt/mlflow-code/"  # Directory where the source is setup.
SOURCE_ARCHIVE_NAME = "src"  # default name of the archive
SETUP_SCRIPT = "setup.sh"
RUN_SCRIPT = "run.sh"
SPARK_CONFIG_SCRIPT = "spark_config.sh"
EMR_CHECK_INTERVAL = 60  # seconds
TRAINING_MODE = 'Training'
INFERENCE_MODE = 'Inference'

_logger = logging.getLogger(__name__)

_EMR_BOOTSTRAP_SETUP_TEMPLATE = """#!/bin/bash
pre_reqs() {{
    echo "Installing Pre-requisites"
    sudo /usr/bin/easy_install-3.6 pip
    sudo /usr/local/bin/pip3 install pip --upgrade
    sudo /usr/local/bin/pip3 install --ignore-installed configparser==3.5.0 # ll /usr/lib/python2.7/dist-packages/backports/
    # sudo /usr/local/bin/pip3 install python-dateutil==2.5.0 pyarrow ipython sklearn tensorflow keras
    # sudo /usr/local/bin/pip3 install python-dateutil --upgrade

    wget -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p /mnt/miniconda
    sudo yum install git -y
    echo -e "alias python=python3\\nalias ll=\\\"ls -alF\\\"\\n" >> /home/hadoop/.bashrc
    echo -e "alias python=python3\\nalias ll=\\\"ls -alF\\\"\\nexport PATH=\\\"\$PATH:/usr/local/bin/\\\"" | sudo tee -a /root/.bashrc
    eval "$(/mnt/miniconda/bin/conda shell.bash hook)"
    conda init
}}
export MLFLOW_VERSION={mlflow_version}
export PATH="$PATH:/usr/local/bin/"

pre_reqs
aws s3 cp {source_location} source_dir.tgz
aws s3 cp {environment_config_location} var.env
chmod +x var.env
source var.env
echo "Current Directory $(pwd)"
env
mkdir -p {source_root_dir}
tar --warning=no-timestamp -xvf source_dir.tgz -C {source_root_dir}
cd {source_directory}
conda env create -f conda.yaml -n $MLFLOW_CONDA_ENV_NAME
"""

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
_TIMEOUT_SCRIPT = """#!/bin/sh 
MAX_SECS_IDLE={max_secs_idle}
if [ -z "$MAX_SECS_IDLE" ]; then MAX_SECS_IDLE=1800; fi

MIN_SECS_TO_END_OF_HOUR={min_secs_to_end_of_hour}
if [ -z "$MIN_SECS_TO_END_OF_HOUR" ]; then MIN_SECS_TO_END_OF_HOUR=300; fi


(
while true  # the only way out is to SHUT DOWN THE MACHINE
do
    # get the uptime as an integer (expr can't handle decimals)
    UPTIME=$(cat /proc/uptime | cut -f 1 -d .)
    SECS_TO_END_OF_HOUR=$(expr 3600 - $UPTIME % 3600)

    # if LAST_ACTIVE hasn't been initialized, hadoop hasn't been installed
    # yet (this happens on 4.x AMIs), or there are jobs running, just set
    # LAST_ACTIVE to UPTIME. This also checks yarn application if it
    # exists (see #1145)
    if [ -z "$LAST_ACTIVE" ] || \ 
        ! which hadoop > /dev/null || \ 
        nice hadoop job -list 2> /dev/null | grep -q '^\s*job_' || \ 
        (which yarn > /dev/null && \ 
            nice yarn application -list 2> /dev/null | \ 
            grep -v 'Total number' | grep -q RUNNING)
    then
        LAST_ACTIVE=$UPTIME
    else
	# the cluster is idle! how long has this been going on?
        SECS_IDLE=$(expr $UPTIME - $LAST_ACTIVE)
        if expr $SECS_IDLE '>' $MAX_SECS_IDLE '&' \ 
            $SECS_TO_END_OF_HOUR '<' $MIN_SECS_TO_END_OF_HOUR > /dev/null
        then
            sudo shutdown -h now
            exit
        fi
    fi
done
# close file handles to daemonize the script; otherwise bootstrapping
# never finishes
) 0<&- &> /dev/null &
"""

_SPARK_CONFIG_SCRIPT = """#!/bin/bash
aws s3 cp {environment_config_location} var.env
chmod +x var.env
source var.env
cnt=0
replaced=0
# Wait for maximum of 10 minutes for the Spark Application bootstrapping to finish 
while [ $cnt -lt 60 ]; do
    if [[ -e "/etc/spark/conf/spark-env.sh" ]]; then
        sudo sed -i -e '$a\export PYSPARK_PYTHON=/mnt/miniconda/envs/'"$MLFLOW_CONDA_ENV_NAME"'/bin/python3' /etc/spark/conf/spark-env.sh
        replaced=1
        break
    else
        echo "File /etc/spark/conf/spark-env.sh not found! Likely bootstrapping of Spark Application is ongoing"
    fi
    cnt=$[$cnt+1]
    sleep 10
done

if [[ $replaced -eq 0 ]]; then
    echo "/etc/spark/conf/spark-env.sh file was not replaced on all EMR nodes!"
    exit 1
fi

"""
_MLFLOW_RUN_SCRIPT = """#!/bin/bash
aws s3 cp {environment_config_location} var.env
chmod +x var.env
source var.env
echo "Current Directory $(pwd)"
env
{input_tasks}
cd {source_directory}
eval "$(/mnt/miniconda/bin/conda shell.bash hook)"
conda activate $MLFLOW_CONDA_ENV_NAME
# TODO: Either this or maintain on conda.yaml
pip install mlflow-sagemaker=={mlflow_version}
{folder_create_tasks}
mlflow run --ignore-duplicate-parameters --entry-point {entry_point} --run-id {run_id} {source_directory} $MLFLOW_PARSED_PARAMETERS
ret_code=$?
if [[ $ret_code -ne 0 ]]; then
    echo "MLFlow run: {run_id} has failed."
    exit 2
fi
{output_tasks}
"""


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
    _create_code_archive(work_dir, target_archive_path)
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


def _create_code_archive(work_dir, result_path):
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


def run_emr_training_job(emr_config, uri, experiment_id, run_id, work_dir, project, mode, entry_point, synchronous,
                         environment_config):
    job_runner = EmrCodeBuildJobRunner(experiment_id, run_id, emr_config, uri, work_dir, mode, entry_point, project,
                                       environment_config)
    job_runner.upload_files()
    job_runner.setup_tags()
    job_runner.setup_code_and_environment()
    job_runner.run()

    return EmrSubmittedRun(emr_config, mode, experiment_id, run_id, uri, work_dir, project, synchronous,
                           job_runner.job_flow_id)
    # env_vars = {
    #     tracking._TRACKING_URI_ENV_VAR: tracking_uri,
    #     tracking._EXPERIMENT_ID_ENV_VAR: experiment_id,
    # }


def parse_uri(uri):
    parsed = urllib.parse.urlparse(uri)
    path = parsed.path
    if parsed.scheme:
        if path.startswith('/'):
            path = path[1:]
    return parsed.scheme, parsed.netloc, path


def resolve_input(input):
    source = input['Source']
    scheme, bucket_name, key = parse_uri(source)
    if scheme != 's3':
        raise MlflowException('Unknown source specified as input for EMR Backend. %s', input)
    # Determine whether the input is S3 Prefix or a file
    if input.get('SourceType', 'prefix').lower() == 'prefix':
        command = 'sync'
    else:
        command = 'cp'
    destination = input['Destination']
    d_scheme, _, _ = parse_uri(destination)
    if d_scheme == 'hdfs':
        input_task = ['TMP_DIR=$(mktemp -d)',
                      f'aws s3 {command} {source} $TMP_DIR',
                      f'hdfs dfs -put $TMP_DIR {destination}'
                      ]
    else:

        # If relative path is specified then append the pre-defined source code directory
        if not destination.startswith('/'):
            destination = os.path.join(SOURCE_ROOT_DIR, SOURCE_ARCHIVE_NAME, destination)
        input_task = [f'aws s3 {command} {source} {destination}']
    return input_task


def resolve_output(output):
    source = output['Source']
    destination = output['Destination']
    folder_create_task = []
    output_task = []
    command = ''
    d_scheme, _, _ = parse_uri(destination)
    if d_scheme != 's3':
        raise MlflowException('Unknown destination specified as input for EMR Backend. %s', output)

    ####################
    # Setup Destination
    ####################

    # Setup SSE Encryption
    if output.get('KmsKeyId'):
        key = output.get('KmsKeyId')
        sse_suffix = f' --sse aws:kms --sse-kms-key-id {key}'
    elif output.get('KmsEncrypt') is not None and output.get('KmsEncrypt'):
        sse_suffix = ' --sse aws:kms'
    else:
        sse_suffix = ' --sse AES256'

    ####################
    # Setup Source
    ####################
    s_scheme, _, _ = parse_uri(source)
    if not s_scheme:  # Local/File type
        if not source.startswith('/'):
            source = os.path.join(SOURCE_ROOT_DIR, SOURCE_ARCHIVE_NAME, source)
        local_source_type = output.get('SourceType', 'folder').lower()
        if local_source_type == 'folder':
            command = 'sync'
            folder = source
            folder_create_task = [f'mkdir -p {folder}']
        elif local_source_type == 'file':
            command = 'cp'
            folder = '/'.join(source.split('/')[:-1])
            folder_create_task = [f'mkdir -p {folder}']
        output_task = [f'aws s3 {command} {source} {destination}{sse_suffix}']
    elif s_scheme == 'hdfs':  # HDFS type source
        local_source_type = output.get('SourceType', 'folder').lower()
        if local_source_type == 'folder':
            command = 'sync'
            if source.endswith('/'):
                source = f'{source}*'
            else:
                source = f'{source}/*'
        elif local_source_type == 'file':
            command = 'cp'
        output_task = ['TMP_DIR=$(mktemp -d)',
                       f'hdfs dfs -copyToLocal {source} $TMP_DIR',
                       f'aws s3 {command} $TMP_DIR {destination}'
                       ]
    return folder_create_task, output_task


class EmrRunner(object):
    def __init__(self, mlflow_experiment_id, mlflow_run_id, emr_config, uri, work_dir, mode, entry_point, project,
                 environment_config):
        _logger.info('EMR Runner %s', emr_config)
        _logger.info('Environment Config %s', environment_config)
        self._mlflow_run_id = mlflow_run_id
        self._mlflow_experiment_id = mlflow_experiment_id
        self._uri = uri
        self._work_dir = work_dir
        self._project = project
        self.emr_config = emr_config
        self.environment_config = environment_config
        self.instance_groups = emr_config['ResourceConfig']['InstanceGroups']
        self.ec2_key_name = emr_config['ResourceConfig'].get('Ec2KeyName', '')
        self.ec2_subnet_id = emr_config['ResourceConfig'].get('Ec2SubnetId')
        self.master_sg = emr_config['ResourceConfig'].get('EmrManagedMasterSecurityGroup')
        self.slave_sg = emr_config['ResourceConfig'].get('EmrManagedSlaveSecurityGroup')
        self.release_label = emr_config['ReleaseLabel']
        self.jobflow_role = emr_config['JobFlowRole']
        self.service_role = emr_config['ServiceRole']
        self.applications = emr_config['Applications']
        self.max_secs_idle = emr_config['MaxIdleTimeInSeconds']
        self.max_secs_end_of_hour = emr_config['MaxSecondsEndOfHour']
        self.log_uri = emr_config["LogUri"]
        self.mode = mode
        self.entry_point = entry_point
        self.visible_to_all_users = emr_config["VisibleToAllUsers"]
        self.cluster_name = emr_config[CLUSTER_NAME]
        self.mode = emr_config[MODE]
        self.tags = emr_config['Tags'] if emr_config['Tags'] else {}
        self.session = boto3.Session()
        self.region = self.session.region_name or "us-east-2"
        store = _get_store()
        artifact_uri = store.get_run(self._mlflow_run_id).info.artifact_uri
        _, self.bucket_name, self.prefix = parse_uri(artifact_uri)
        self.input_tasks_list = self._setup_input_tasks()
        self.folder_create_list, self.output_tasks_list = self._setup_output_tasks()
        self._setup_environment()

    def upload_files(self):
        self.source_location = upload_source_code(self._work_dir, self.bucket_name, self.prefix)
        self.environment_config_location = upload_environment(self.bucket_name,
                                                              self.prefix, self.environment_config)

        self.setup_bootstrap_script, self.timeout_script, self.run_script, self.spark_config_script = self.upload_scripts()

    def _setup_output_tasks(self):
        output_tasks_list = []
        folder_create_list = []
        if self.emr_config.get('Output'):
            outputs = self.emr_config['Output']
            if isinstance(outputs, list):
                cnt = 1
                for output in outputs:
                    # Append Cluster Name
                    destination = output['Destination']
                    cluster_name = self.cluster_name
                    if not destination.endswith('/'):
                        destination = f'{destination}/'
                    destination = f'{destination}{cluster_name}'
                    output['Destination'] = destination
                    self.tags.append({'Key': f'Output-{cnt}', 'Value': destination})
                    folder_create_task, output_tasks = resolve_output(output)
                    output_tasks_list = output_tasks_list + output_tasks
                    folder_create_list = folder_create_list + folder_create_task
                    cnt = cnt + 1
            elif isinstance(outputs, dict):
                # Append Cluster Name
                destination = outputs['Destination']
                cluster_name = self.cluster_name
                if not destination.endswith('/'):
                    destination = f'{destination}/'
                destination = f'{destination}{cluster_name}'
                outputs['Destination'] = destination
                self.tags.append({'Key': 'Output', 'Value': destination})
                folder_create_list, output_tasks_list = resolve_output(outputs)

        return folder_create_list, output_tasks_list

    def _setup_input_tasks(self):
        input_tasks_list = []
        if self.emr_config.get('Input'):
            inputs = self.emr_config['Input']
            if isinstance(inputs, list):
                for input in inputs:
                    input_tasks_list = input_tasks_list + resolve_input(input)
            elif isinstance(inputs, dict):
                input_tasks_list = resolve_input(inputs)

        return input_tasks_list

    def upload_scripts(self):
        with TempDir() as tmp:
            cwd = tmp.path()
            setup_script = os.path.join(cwd, SETUP_SCRIPT)
            timeout_script = os.path.join(cwd, TIMEOUT_SCRIPT)
            run_script = os.path.join(cwd, RUN_SCRIPT)
            spark_config_script = os.path.join(cwd, SPARK_CONFIG_SCRIPT)
            with open(setup_script, "w") as f:
                f.write(
                    _EMR_BOOTSTRAP_SETUP_TEMPLATE.format(
                        source_location=self.source_location,
                        environment_config_location=self.environment_config_location,
                        mlflow_version=__version__,
                        source_root_dir=SOURCE_ROOT_DIR,
                        source_directory=os.path.join(SOURCE_ROOT_DIR, SOURCE_ARCHIVE_NAME)
                    )
                )

            with open(timeout_script, "w") as f:
                f.write(
                    _TIMEOUT_SCRIPT.format(
                        max_secs_idle=self.max_secs_idle,
                        min_secs_to_end_of_hour=self.max_secs_end_of_hour,
                    )
                )

            with open(run_script, "w") as f:
                f.write(
                    _MLFLOW_RUN_SCRIPT.format(
                        environment_config_location=self.environment_config_location,
                        input_tasks='\n'.join(self.input_tasks_list),
                        output_tasks='\n'.join(self.output_tasks_list),
                        folder_create_tasks='\n'.join(self.folder_create_list),
                        source_directory=os.path.join(SOURCE_ROOT_DIR, SOURCE_ARCHIVE_NAME),
                        run_id=self._mlflow_run_id,
                        mlflow_version=__version__,
                        entry_point=self.entry_point
                    )
                )
            with open(spark_config_script, "w") as f:
                f.write(
                    _SPARK_CONFIG_SCRIPT.format(
                        environment_config_location=self.environment_config_location,
                    )
                )

            s3_setup_script_key = '{}/{}'.format(self.prefix, SETUP_SCRIPT)
            s3_timeout_script_key = '{}/{}'.format(self.prefix, TIMEOUT_SCRIPT)
            s3_run_script_key = '{}/{}'.format(self.prefix, RUN_SCRIPT)
            s3_spark_config_script_key = '{}/{}'.format(self.prefix, SPARK_CONFIG_SCRIPT)
            client = boto3.client('s3')
            transfer = boto3.s3.transfer.S3Transfer(client=client)

            transfer.upload_file(filename=setup_script,
                                 bucket=self.bucket_name, key=s3_setup_script_key,
                                 extra_args={'ServerSideEncryption': 'AES256'})
            setup_script_uri = f's3://{self.bucket_name}/{s3_setup_script_key}'
            _logger.info('Uploaded the setup script to %s', setup_script_uri)

            transfer.upload_file(filename=timeout_script,
                                 bucket=self.bucket_name, key=s3_timeout_script_key,
                                 extra_args={'ServerSideEncryption': 'AES256'})
            timeout_script_uri = f's3://{self.bucket_name}/{s3_timeout_script_key}'
            _logger.info('Uploaded the timeout script to %s', timeout_script_uri)

            transfer.upload_file(filename=run_script,
                                 bucket=self.bucket_name, key=s3_run_script_key,
                                 extra_args={'ServerSideEncryption': 'AES256'})
            run_script_uri = f's3://{self.bucket_name}/{s3_run_script_key}'
            _logger.info('Uploaded the run script to %s', run_script_uri)

            transfer.upload_file(filename=spark_config_script,
                                 bucket=self.bucket_name, key=s3_spark_config_script_key,
                                 extra_args={'ServerSideEncryption': 'AES256'})
            spark_config_script_uri = f's3://{self.bucket_name}/{s3_spark_config_script_key}'
            _logger.info('Uploaded the spark config script to %s', spark_config_script_uri)

            return setup_script_uri, timeout_script_uri, run_script_uri, spark_config_script_uri

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
                # {
                #     'Name': 'Spark Config Script',
                #     'ScriptBootstrapAction': {
                #         'Path': self.spark_config_script
                #     }
                # },
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
            ],
            Tags=self.tags
        )
        response_code = response['ResponseMetadata']['HTTPStatusCode']
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            self.job_flow_id = response['JobFlowId']  # Same as Cluster ID
            _logger.info('Successfully started the EMR cluster Id: %s', self.job_flow_id)
            _logger.info('Full Response: %s', response)
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

    def _setup_environment(self):
        yarn_environment_list = {}
        spark_environment_list = {}
        with open('var.env', 'r') as f:
            for line in f:
                if line.startswith('export '):
                    line = line.replace('export ', '')
                value = '='.join(line.split('=')[1:])
                value = value.strip('\n').strip('"').strip()
                if len(value.split(' ')) > 1:
                    value = "'{}'".format(value)
                yarn_environment_list[line.split('=')[0]] = value
        yarn_environment_list['MLFLOW_RUN_ID'] = self._mlflow_run_id
        yarn_environment_list['MLFLOW_EXPERIMENT_ID'] = self._mlflow_experiment_id
        mlflow_env_name = yarn_environment_list['MLFLOW_CONDA_ENV_NAME']
        yarn_environment_list['PYSPARK_PYTHON'] = f'/mnt/miniconda/envs/{mlflow_env_name}/bin/python3'
        spark_environment_list['PYSPARK_PYTHON'] = f'/mnt/miniconda/envs/{mlflow_env_name}/bin/python3'
        for instance in self.instance_groups:
            instance['Configurations'] = [
                {
                    "Classification": "yarn-env",
                    "Properties": {

                    },
                    "Configurations": [
                        {
                            "Classification": "export",
                            "Properties": yarn_environment_list,
                            "Configurations": [
                            ]
                        }
                    ]
                },
                {
                    "Classification": "spark-env",
                    "Properties": {

                    },
                    "Configurations": [
                        {
                            "Classification": "export",
                            "Properties": spark_environment_list,
                            "Configurations": [
                            ]
                        }
                    ]
                }

            ]
            _logger.info('Yarn Configuration: %s', instance['Configurations'])


class EmrCodeBuildJobRunner(EmrRunner):

    def __init__(self, mlflow_experiment_id, mlflow_run_id, emr_config, uri, work_dir, mode, entry_point, project,
                 environment_config):
        super(EmrCodeBuildJobRunner, self).__init__(mlflow_experiment_id, mlflow_run_id, emr_config, uri, work_dir,
                                                    mode, entry_point, project, environment_config)

        self._fetch_codebuild_tags()
        self.prefix = '{}/{}/{}-{}'.format(self._canonical_name, self._codebuild_stage,
                                           # Override Prefix and Bucket Name
                                           self._codebuild_no, self._commit_id_short)
        self.bucket_name = self.emr_config[S3_BUCKET_NAME]

    def setup_tags(self):
        super(EmrCodeBuildJobRunner, self).setup_tags()

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


class EmrSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Kubernetes Job run launched to run an MLflow
    project.
    :param mlflow_run_id: ID of the MLflow project run.
    :param job_name: Sagemaker job name.
    :param job_namespace: Sagemaker job namespace.
    """

    def __init__(self, emr_config, mode, mlflow_experiment_id, mlflow_run_id, uri, work_dir, project, synchronous,
                 job_flow_id):
        self._mlflow_run_id = mlflow_run_id
        self._mlflow_experiment_id = mlflow_experiment_id
        self._uri = uri
        self.mode = mode
        self._work_dir = work_dir
        self._project = project
        self.emr_config = emr_config
        self.synchronous = synchronous
        self.cluster_name = emr_config[CLUSTER_NAME]
        self.job_flow_id = job_flow_id
        super(EmrSubmittedRun, self).__init__()

    @property
    def run_id(self):
        return self._mlflow_run_id

    def wait(self):
        # sagemaker_session = aws_sm.Session()
        # sagemaker_session.logs_for_job(self.training_job_name, wait=self.synchronous, log_type='All')
        # training_job = sagemaker_session.describe_training_job(job_name=self.training_job_name)
        # job_status = training_job['TrainingJobStatus']

        client = boto3.client('emr')
        # Blocking wait until cluster exits with or without error
        while True:
            import time
            job_status = client.describe_cluster(ClusterId=self.job_flow_id)['Cluster']['Status']
            if job_status['State'] in ['TERMINATED_WITH_ERRORS']:
                _logger.error('Cluster did not complete jobs successfully. Current Status: %s', job_status)
                raise MlflowException('EMR Backend Job: {} failed with a State - {}'.format(self.cluster_name,
                                                                                            job_status['State']))
            elif job_status['State'] in ['TERMINATING']:
                _logger.info('Cluster completed the jobs. Terminating now. Current Status: %s', job_status)
                if job_status['StateChangeReason'].get('Code', '') == 'STEP_FAILURE':
                    _logger.error(
                        'Cluster completed the jobs. But some of steps/bootstrapping failed. Error Message: %s',
                        job_status['StateChangeReason'].get('Message', ''))
                    raise MlflowException('EMR Backend Job: {} failed. Reason: {}'.format(self.cluster_name,
                                                                                          job_status[
                                                                                              'StateChangeReason']))
                if job_status['StateChangeReason'].get('Code', '') == 'ALL_STEPS_COMPLETED':
                    _logger.info(
                        'EMR Cluster (%s) completed the job steps. Waiting for terminating the cluster. '
                        'Status Message: %s', self.job_flow_id, job_status)
            elif job_status['State'] in ['TERMINATED']:
                _logger.info('EMR Cluster (%s) terminated. Current Status: %s', self.job_flow_id, job_status)
                break
            else:
                _logger.info('%s mode EMR Cluster (%s) is currently processing the job. Current Status: %s',
                             self.mode.capitalize(), self.job_flow_id,
                             job_status)
            time.sleep(EMR_CHECK_INTERVAL)

        return True
