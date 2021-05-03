#
# Welcome to your Adverse Event Python project!

The Adverse Event project is templatized with Amazon CDK. The `cdk.json` file tells the CDK Toolkit how to execute your app.

This project is set up like a standard Python project.  The initialization process also creates
a virtualenv within this project, stored under the .venv directory.  To create the virtualenv
it assumes that there is a `python3` executable in your path with access to the `venv` package.
If for any reason the automatic creation of the virtualenv fails, you can create the virtualenv
manually once the init process completes.

To manually create a virtualenv on MacOS and Linux:

```
python3 -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following
step to activate your virtualenv.

```
$ source .venv/bin/activate
```

Once the virtualenv is activated, you can install the required dependencies.

```
pip install -r requirements.txt
```

Download `en_core_web_sm`

```
python -m spacy download en_core_web_sm
```

At this point you can now synthesize the CloudFormation template for this code.

```
cdk synth
cdk deploy --all
```

Alternatively, you can also deploy the stacks one by one, by doing:

```
cdk deploy ae-bucket
cdk deploy ae
```

Navigate to Cloud9, in stream_config.py add credentials to lines 2-6 and 17-18 and save the file.

Back in Cloud9, make sure you are in ae-blog-cdk and run the following command to initate the listener:

```
python cloud9/stream.py
```

You can now begin exploring the source code, contained in the hello directory.
There is also a very trivial test included that can be run like this:

```
pytest
```

To add additional dependencies, for example other CDK libraries, just add to
your requirements.txt file and rerun the `pip install -r requirements.txt`
command.

## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation

## Project Structure


## Citation


## Useful links

## Others

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

