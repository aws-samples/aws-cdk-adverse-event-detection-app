# Welcome to the Cloud9 Twitter listener

1. Create and activate virtual env

```
virtualenv .venv
source .venv/bin/activate
```

2. Install requirements.txt 

```
pip install -r requirements.txt
```

3. Install Spacy required files

```
python -m spacy download en_core_web_sm
```

4. Run the listener

```
python cloud9/stream.py
```
