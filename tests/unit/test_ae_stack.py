import json
import pytest

from aws_cdk import core
from ae.ae_stack import AeStack


def get_template():
    app = core.App()
    AeStack(app, "ae")
    return json.dumps(app.synth().get_stack("ae").template)

def test_role_created():
    assert("AWS::IAM::Role" in get_template())

def test_sagemaker_created():
    assert("AWS::SageMaker::NotebookInstance" in get_template())

def test_lambda_created():
    assert("AWS::Lambda::Function" in get_template())

def test_ddb_created():
    assert("AWS::DynamoDB::Table" in get_template())