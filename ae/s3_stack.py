from aws_cdk import (
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    core
)

class S3Stack(core.Stack):
    @property
    def bucket(self):
        return self._bucket  

    def __init__(self, scope: core.Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # S3 Bucket 
        self._bucket = s3.Bucket(
            self, "AEModelingBucket", 
            bucket_name=f"ae-modeling-bucket-{self.account}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=core.RemovalPolicy.DESTROY, auto_delete_objects=True
        )

        # Upload notebooks from local to S3
        s3deploy.BucketDeployment(
            self, 'UploadModelingFiles',
            sources=[s3deploy.Source.asset('sagemaker')],
            destination_bucket=self._bucket,
            destination_key_prefix='sagemaker'
        )

        # Upload cloud9 scripts from local to S3
        s3deploy.BucketDeployment(
            self, 'UploadCloud9Scripts',
            sources=[s3deploy.Source.asset('cloud9')],
            destination_bucket=self._bucket,
            destination_key_prefix='cloud9'
        )
