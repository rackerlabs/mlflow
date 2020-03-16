"""
File containing all of the run additional tags in the sagemaker job config when using codebuild.

See the REST API documentation for information on the meaning of these tags.
"""
S3_BUCKET_NAME = 'S3BucketName'
S3_ARTIFACT_URI = 'S3ArtifactUri'
CODEBUILD_STAGE = 'CodeBuildStage'
CODEBUILD_NO = 'CodeBuildNo'
EXECUTION_ID = 'PipelineExecutionID'
CANONICAL_MODEL_NAME = 'CanonicalName'
ESTIMATOR_TYPE = 'EstimatorType'
MODEL_COMMIT = 'ModelCommit'
TRAINING_JOB_NAME = 'TrainingJobName'
CODEBUILD_INHERENT_NAMES = [
    S3_BUCKET_NAME,
    S3_ARTIFACT_URI,
    CODEBUILD_STAGE,
    CODEBUILD_NO,
    EXECUTION_ID,
    CANONICAL_MODEL_NAME,
    ESTIMATOR_TYPE,
    MODEL_COMMIT
]