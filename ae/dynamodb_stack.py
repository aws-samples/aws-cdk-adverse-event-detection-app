from aws_cdk import (
    aws_lambda as _lambda,
    aws_dynamodb as ddb,
    aws_iam as iam,
    aws_lambda_event_sources as lambda_event_sources,
    core,
)

class TweetsTable(core.Construct):
    @property
    def table(self):
        return self._table  

    @property
    def handler(self):
        return self._handler

    def __init__(self, scope: core.Construct, id: str, table_name: str, AE_THRESHOLD: str, bucket_name: str, **kwargs):
        super().__init__(scope, id, **kwargs)


        self._table = ddb.Table(
            self, 'Tweets',
            partition_key=ddb.Attribute(name="id", type=ddb.AttributeType.NUMBER),
            read_capacity=5, write_capacity=5, table_name=table_name,
            stream=ddb.StreamViewType.NEW_AND_OLD_IMAGES,
            removal_policy=core.RemovalPolicy.DESTROY
        )

        self._lambda_iam = iam.Role(
            self, "LambdaServiceRole",
            role_name = "LambdaServiceRoleForDDB",
            assumed_by = iam.CompositePrincipal(
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
                iam.ServicePrincipal("lambda.amazonaws.com"),
                iam.ServicePrincipal("dynamodb.amazonaws.com")
            )
        )
        self._lambda_iam.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AmazonS3FullAccess'))
        self._lambda_iam.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AmazonDynamoDBFullAccess'))
        self._lambda_iam.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AmazonSageMakerFullAccess'))

        self._handler = _lambda.Function(
            self, 'InferenceHandler',
            runtime=_lambda.Runtime.PYTHON_3_7,
            code=_lambda.Code.from_asset('lambda'),
            handler='inference.lambda_handler',
            role=self._lambda_iam,
            environment={
                "PROJECT_BUCKET_NAME": bucket_name,
                "AE_THRESHOLD": AE_THRESHOLD,
                "ENDPOINT_NAME": "HF-BERT-AE-model"
            }
        )
        self.handler.add_event_source(
            lambda_event_sources.DynamoEventSource(
                self.table,
                starting_position=_lambda.StartingPosition.LATEST, 
                batch_size=10, retry_attempts=10000))


        self.table.grant_read_write_data(self.handler)

