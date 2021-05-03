import json
import boto3
import os
import datetime

s3_client = boto3.client('s3')
# grab environment variables
ENDPOINT_NAME = 'HF-BERT-AE-model'
bucket = os.environ['PROJECT_BUCKET_NAME']
AE_threshold = os.environ['AE_THRESHOLD']
runtime = boto3.client('runtime.sagemaker')
cm_client = boto3.client(service_name='comprehendmedical')

print('Loading function')

# LAMBDA EVENT HANDLER
def lambda_handler(event, context):
    now = datetime.datetime.now()
    print("Received event: " + json.dumps(event, indent=2))

    for record in event['Records']:
        if "NewImage" not in record['dynamodb']:
            return

        text = record['dynamodb']['NewImage']['text']['S']
        id = record['dynamodb']['NewImage']['id']['N']
        timestamp = record['dynamodb']['NewImage']['timestamp']['S']
        user_id = record['dynamodb']['NewImage']['user_id']['S']

        user_city = record['dynamodb']['NewImage']['user_city']['S']
        user_state = record['dynamodb']['NewImage']['user_state']['S']
        user_location = record['dynamodb']['NewImage']['user_location']['S']

        drug_mentions = record['dynamodb']['NewImage']['drug_mentions']['S']

        response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                   ContentType='application/json',
                                   Body=json.dumps(text))
                                   
        prob = eval(response['Body'].read())
        prd_prob = prob[1]
        pred_label = "Adverse_Event" if prd_prob >= float(AE_threshold) else "Not_AE"

        # Retrieve AE type data IFF marked as AE by model
        ae_type = ""
        icd_codes = []
        if pred_label == 'Adverse_Event':
            aetype_dict = {}
            # Extract entities using Amazon Comprehend Medical
            result_symptom = cm_client.detect_entities_v2(Text=text)
            entities_symptom = result_symptom['Entities']
            # Look for entities that detects signs, symptoms, and diagnosis of medical conditions 
            # Filter based on confidence score
            for entity in entities_symptom:
                    if (entity['Category']=='MEDICAL_CONDITION') & (entity['Score']>=0.60):
                        aetype_dict[entity['Text']] = entity['Score']
                        # Extract entity with maximum score
                        ae_type = max(aetype_dict, key=aetype_dict.get)

            _dict = {}
            icdc_list = []
            
            # Amazon Comprehend Medical lists the matching ICD-10-CM codes
            result_icd = cm_client.infer_icd10_cm(Text=text)
            entities_icd = result_icd['Entities']
            for entity in entities_icd:
                for codes in entity['ICD10CMConcepts']:
                    # Filter based on confidence score
                    if codes['Score'] >= 0.70:
                        _dict[codes['Description']] = codes['Score']
                        # Extract entity with maximum score
                        icd_ = max(_dict, key=_dict.get)
                        icdc_list.append(icd_)
            icd_codes = list(set(icdc_list))

        else:
            ae_type="UNDOCUMENTED"
        
        save_data = {'id': id, 'text': text, 'pred_label': pred_label, 'pred_prob': prd_prob, 'user_id': user_id, 
        'timestamp': timestamp, 'user_city': user_city, 'user_state': user_state, 'user_location': user_location, 
        'drug_mentions': drug_mentions, 'ae_type': ae_type, 'icd_codes': icd_codes
        }
        
        with open('/tmp/tweeter_data.txt', 'w') as outfile:
            json.dump(save_data, outfile)

        s3_client.upload_file('/tmp/tweeter_data.txt', bucket,
                              'lambda_predictions/tweeter_' + str(id) + '_' + str(now) + '.txt')

    return save_data