#!/usr/bin/env python3

from aws_cdk import core

from ae.s3_stack import S3Stack
from ae.ae_stack import AeStack


app = core.App()

# Create bucket 
ae_bucket = S3Stack(app, 'ae-bucket', env={'region': 'us-east-1'})
AeStack(app, "ae", env={'region': 'us-east-1'}, bucket=ae_bucket.bucket)

app.synth()
