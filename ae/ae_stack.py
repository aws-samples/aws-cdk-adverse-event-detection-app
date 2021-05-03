from aws_cdk import (
    aws_lambda as _lambda,
    aws_apigateway as apigw,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_sagemaker as sagemaker,
    aws_iam as iam,
    aws_cloud9 as cloud9,
    aws_ec2 as ec2,
    aws_codecommit as codecommit,
    aws_glue as glue,
    aws_lambda_event_sources as lambda_event_sources,
    core
)

from .modeling_stack import AeModelStack
from .dynamodb_stack import TweetsTable
from .glue_stack import GlueStack
AE_THRESHOLD = "0.6"

class AeStack(core.Stack):

    def __init__(self, scope: core.Construct, construct_id: str, bucket: object, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # Model Training and Deployment 
        ## SageMaker Resources
        AeModelStack(self, 'AEModeling', bucket_name=bucket.bucket_name)

        ## Run the notebook in SageMaker to train the model and keep a record of the endpoint 

        # Inference pipeline
        ## DynamoDB for recording streaming tweets
        tweets_processer = TweetsTable(
            self, 'TweetsWrittenByLambda', table_name="ae_tweets_ddb",
            AE_THRESHOLD=AE_THRESHOLD, bucket_name=bucket.bucket_name)

        # Twitter API listener with Cloud9
        cloud9_script_repo = codecommit.CfnRepository(
            self, 'Cloud9Script', repository_name='cloud9-script-repo',
            code=codecommit.CfnRepository.CodeProperty(s3=codecommit.CfnRepository.S3Property(                
                bucket=bucket.bucket_name, 
                key='cloud9/cloud9.zip'))
        )

        new_vpc = ec2.Vpc(self, "VPC")
        cloud9.CfnEnvironmentEC2(self, 'Cloud9Env',
            instance_type='t3.large',
            repositories=[cloud9.CfnEnvironmentEC2.RepositoryProperty(
                path_component = '/src/twitter-scripts', 
                repository_url = f'https://git-codecommit.us-east-1.amazonaws.com/v1/repos/{cloud9_script_repo.repository_name}')],
            subnet_id=new_vpc.public_subnets[0].subnet_id
        )

        # Grant read write access to AWS Glue Crawler
        bucket.grant_read_write(tweets_processer.handler)
        GlueStack(self, "GlueCrawler", bucket.bucket_name)