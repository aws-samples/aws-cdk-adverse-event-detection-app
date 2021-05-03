from aws_cdk import (
    aws_iam as iam,
    aws_glue as glue,
    core
)

class GlueStack(core.Construct):

    def __init__(self, scope: core.Construct, construct_id: str, target_bucket: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self._PREFIX = construct_id

        # Create Role for the Glue crawler
        self._service_role = iam.Role(
            self, f'{self._PREFIX}-ServiceRole',
            role_name=f'{self._PREFIX}-ServiceRole',
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal('glue.amazonaws.com'),
                iam.ServicePrincipal('s3.amazonaws.com')
            )
        )
        self._service_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AmazonS3FullAccess'))
        self._service_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('service-role/AWSGlueServiceRole'))

        self._glue_crawler = glue.CfnCrawler(
            self, "InferenceResultCrawler",
            database_name="s3_tweets_db", name="s3_tweets_crawler",
            role=self._service_role.role_arn,
            schedule=glue.CfnCrawler.ScheduleProperty(schedule_expression="cron(29 0/1 * * ? *)"),
            targets={"s3Targets": [{"path": f"s3://{target_bucket}/lambda_predictions/"}]
                },
            )