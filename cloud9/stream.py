import requests
import json
import boto3
import os
import datetime
import time
import pandas as pd
from requests.auth import AuthBase
from requests.auth import HTTPBasicAuth
from langdetect import detect
from twython import Twython
import geonamescache
import en_core_web_sm
import stream_config


def gen_dict_extract(var, key):
    """
    Helper function for location dictionary
    """
    if isinstance(var, dict):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, (dict, list)):
                yield from gen_dict_extract(v, key)
    elif isinstance(var, list):
        for d in var:
            yield from gen_dict_extract(d, key)
            
def drug_types(text, drug_names=stream_config.drug_names):
    """
    Search for drug mentions (substring) in raw text of Tweet

    Parameters:
    text (str): raw text of Tweet
    
    Returns:
    contains (list): List of drug names mentioned in a Tweet
    """
    
    contains = []
    for drug in drug_names:
        if len([True for name in drug if name.lower() in text.lower()]) > 0:
            contains.append(drug[0])
    return contains


def locate(u_id):
    """
    Parses location info from user (account) object (NOT Tweet object)

    Parameters:
    u_id (str): User ID in string form.

    Returns:
    (unnamed list): List of strings of length 2.
    """
    # Authenticate
    twitter = Twython(stream_config.APP_KEY, stream_config.APP_SECRET, stream_config.OAUTH_TOKEN, stream_config.OAUTH_TOKEN_SECRET)
    ids = str(u_id) # Can be a comma-separated string list if we want to retrieve by batch; pls visit docs

    # Query twitter 
    output = twitter.lookup_user(user_id=ids)

    # Get raw location info from user object
    raw_geo = str(output[0]['location'])

    # Decipher location
    gc = geonamescache.GeonamesCache()
    states = gc.get_us_states()
    cities = gc.get_cities()
    us_cities = [city for city in cities.values() if city['countrycode'] == 'US']
    us_cities_names = [*gen_dict_extract(us_cities, 'name')]
    states_names = [*gen_dict_extract(states, 'name')]

    nlp = en_core_web_sm.load()
    doc = nlp(raw_geo)
    # Loop through and identify entities recognized and extracted from raw location info
    for ent in doc.ents:
      # print(ent.text, ent.start_char, ent.end_char, ent.label_)
      if ent.label_ is 'GPE':
            if ent.text in us_cities_names:
                return ['city', ent.text]
            elif ent.text in states_names:
                return ['state', ent.text]
            else:
                print("raw_geo: " + str(raw_geo))
                return ['other', raw_geo]
    else:
        return ['other', raw_geo]

# Dict: Finds state that given city is located in
city_to_state_dict = {}
with open("cloud9/city_to_state_dict.py", "r") as config_file:
    city_to_state_dict = json.load(config_file)

# Sets up authentication to connect to Twitter API:
consumer_key = stream_config.APP_KEY # Add your API key here
consumer_secret = stream_config.APP_SECRET # Add your API secret key here
stream_url = "https://api.twitter.com/2/tweets/search/stream" #"https://api.twitter.com/labs/1/tweets/stream/filter"
rules_url = "https://api.twitter.com/2/tweets/search/stream/rules" #"https://api.twitter.com/labs/1/tweets/stream/filter/rules"

# Gets a bearer token
class BearerTokenAuth(AuthBase):
    def __init__(self, consumer_key, consumer_secret):
        self.bearer_token_url = "https://api.twitter.com/oauth2/token"
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.bearer_token = self.get_bearer_token()

    def get_bearer_token(self):
        
        response = requests.post(
        self.bearer_token_url, 
        auth=(self.consumer_key, self.consumer_secret),
        data={'grant_type': 'client_credentials'},
        headers={'User-Agent': 'TwitterDevFilteredStreamQuickStartPython'})
        if response.status_code is not 200:
            raise Exception(f"Cannot get a Bearer token (HTTP %d): %s" % (response.status_code, response.text))
        body = response.json()
        return body['access_token']
    
    def __call__(self, r):
        r.headers['Authorization'] = f"Bearer %s" % self.bearer_token
        r.headers['User-Agent'] = 'TwitterDevFilteredStreamQuickStartPython'
        return r

# Resets filters/rules 
bearer_token=stream_config.bearer_token
headers = {"Authorization": "Bearer {}".format(bearer_token)}

def get_all_rules(headers, bearer_token):
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream/rules", headers=headers
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot get rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    # print(json.dumps(response.json()))
    return response.json()


def delete_all_rules(headers, bearer_token, rules):
    if rules is None or "data" not in rules:
        return None

    ids = list(map(lambda rule: rule["id"], rules["data"]))
    payload = {"delete": {"ids": ids}}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        headers=headers,
        json=payload
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot delete rules (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    # print(json.dumps(response.json()))


def set_rules(headers, bearer_token, rules):
    if rules is None:
        return
  
    payload = {
        'add': rules
      }
    
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        headers=headers,
        json=payload,
    )

    if response.status_code != 201:
        raise Exception(
            "Cannot add rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    # print(json.dumps(response.json()))


# Set up session and connect to a DynamoDB table
session = boto3.Session(region_name=stream_config.region_name,
                        aws_access_key_id= stream_config.aws_access_key_id,
                        aws_secret_access_key= stream_config.aws_secret_access_key)
ddb = session.resource('dynamodb')
table = ddb.Table(stream_config.table_name) # TABLE NAME


def stream_connect(headers, set, bearer_token):
    """
    Connects and starts the stream, processes data via functions (e.g. locate users) and loads processed Tweet data
    to DynamoDB table.
    
    Parameters:
    stream_url (str): Twitter-provided URl for the filtered stream v1.
    """
    response = requests.get("https://api.twitter.com/2/tweets/search/stream?tweet.fields=created_at,author_id", headers=headers, stream=True)
    for response_line in response.iter_lines():
        if response_line:
            try:
                data = json.loads(response_line)
                content = {}
                content['id'] = int(data['data']['id'])
                content['text'] = data['data']['text']
                content['timestamp'] = str(data['data']['created_at']).replace('T', ' ').replace('Z', '')
                content['user_id'] = data['data']['author_id']
              
                # Drug mentioned by Tweet
                content['drug_mentions'] = ', '.join(drug_types(content['text']))
                
                # Add location info
                content['user_city'] = ""
                content['user_state'] = ""
                content['user_location'] = ""
                      
                # Filter out retweets and non-English Tweets
                if not str(content['text']).startswith('RT ') and detect(content['text']) == 'en':
                    location_list = locate(str(content['user_id'])) # List format: ['city OR state OR other', 'city_name OR state_name OR other_name']
                    if location_list[0] is not None and location_list[0] == 'city':
                        city = location_list[1]
                        content['user_city'] = location_list[1]
                        # States often abbreviated if there is city, we must find state by city, not abbreviation
                        if city is not None and city in city_to_state_dict:
                            content['user_state'] = city_to_state_dict[city]
                        print("CITY: " + str(content['user_city']))
                    elif location_list[0] is not None and location_list[0] == 'state':
                        content['user_state'] = location_list[1]
                        print("STATE: " + str(content['user_state']))
                    elif location_list[0] is not None and location_list[0] == 'other':
                        content['user_location'] = location_list[1]
                        print("LOCATION: " + str(content['user_location']))
                    
                
                    table.put_item(Item=content)
                print(content)
                print(json.dumps(data, indent=4, sort_keys=True))
              
            except Exception as e:
                print(str(e))
    

def setup_rules(headers, bearer_token):
    current_rules = get_all_rules(headers, bearer_token)
    delete_all_rules(headers, bearer_token, current_rules)
    set_rules(headers, bearer_token, stream_config.sample_rules)


setup_rules(headers, bearer_token)

# Listen to the stream.
# This reconnection logic will attempt to reconnect when a disconnection is detected.
# To avoid rate limits, this logic implements exponential backoff, so the wait time
# will increase if the client cannot reconnect to the stream.
timeout = 0
while True:
    stream_connect(headers, set, bearer_token)