# Twitter Authentication: update with your own credentials
APP_KEY =''
APP_SECRET = ''
OAUTH_TOKEN = ''
OAUTH_TOKEN_SECRET = ''
bearer_token='' 

# Twitter Rules List (stream filters), see Twitter docs for details
# Note that these rules are overwritten each time the script is run!
sample_rules = [
    { 'value': 'prozac OR fluoxetine OR cymbalta OR celexa OR citalopram OR amoxil OR amoxicillin OR moxatag OR lexapro OR escitalopram lang:en -is:retweet', "tag": "newest"},]

# List of drug brand names to be extract from raw tweets.
drug_names = [['prozac', 'fluoxetine'], ['celexa', 'citalopram'], 
              ['cymbalta', 'duloxetine'], ['lexapro', 'escitalopram'], 
              ['moxatag', 'amoxicillin','amoxil']]  # [[brand name, name variant 1, name variant 2, ...], ...]

# DynamoDB Information
region_name ='us-east-1'
aws_access_key_id = '' # update with your own credentials
aws_secret_access_key = '' #update with your own credentials 
table_name = 'ae_tweets_ddb'
