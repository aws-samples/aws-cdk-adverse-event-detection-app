from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
import random
import logging
import sys
import argparse
import os
import torch
import json
import numpy as np


# Set up logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.getLevelName("DEBUG"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


try:
    output_data_dir = os.environ["SM_OUTPUT_DIR"]
except KeyError:
    output_data_dir = "/opt/ml/output"
    
try:
    model_dir = os.environ["SM_MODEL_DIR"]
except KeyError:
    model_dir = "/opt/ml/model"
    
try:
    training_dir = os.environ["SM_CHANNEL_TRAINING"]
except KeyError:
    training_dir = "/opt/ml/input/data/training"  

parser = argparse.ArgumentParser()

# hyperparameters sent by the client are passed as command-line arguments to the script.
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--max_seq_length", type=int, default=64)
parser.add_argument("--train-batch-size", type=int, default=16)
parser.add_argument("--eval-batch-size", type=int, default=16)
parser.add_argument("--warmup_steps", type=int, default=10)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--model_name", type=str, default="bert-base-uncased")
parser.add_argument("--text_column", type=str)
parser.add_argument("--label_column", type=str)

# Data, model, and output directories
parser.add_argument("--output-data-dir", type=str, default=output_data_dir)
parser.add_argument("--model-dir", type=str, default= model_dir)
#parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
parser.add_argument("--training_dir", type=str, default= training_dir)
parser.add_argument("--test_dir", type=str, default= training_dir)

args, _ = parser.parse_known_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def set_seed(seed):
    """
    Sets random seeds for training.
    :param seed: Integer used for seed.
    :return: void
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tokenize(batch):
    return tokenizer(batch['text'], max_length=args.max_seq_length, padding='max_length', truncation=True)

def _get_dataset(data_dir, data_file_name, text_column, label_column):
    """generate dataset for model training"""
    
    dataset = load_dataset('csv', data_files={'train': os.path.join(data_dir, data_file_name)})

    if not 'labels' in dataset['train'].column_names:
        dataset = dataset.rename_column(label_column, 'labels')
    if not 'text' in dataset['train'].column_names:
        dataset = dataset.rename_column(text_column, 'text')

    dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))
    dataset.set_format('torch', columns=['labels', 'attention_mask', 'input_ids'])
    
    return dataset['train']


def train(args):
    """Model training"""
    
    set_seed(args.seed)
    
    train_dataset = _get_dataset(args.training_dir, "train.csv", args.text_column, args.label_column)
    valid_dataset = _get_dataset(args.training_dir, "valid.csv", args.text_column, args.label_column)
    
    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    
    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    
    # define training args
    training_args = TrainingArguments(
        output_dir=args.output_data_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        seed = args.seed,
        save_steps = 500,
        save_total_limit = 2,
        evaluation_strategy="steps",
        eval_steps = 50,
        logging_steps=50,
        logging_dir=args.output_data_dir,
        learning_rate=float(args.learning_rate),
    )
    
    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    
    # train model
    trainer.train()
    
    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=valid_dataset)
    
    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(args.model_dir)
    
############ Functions for SageMaker endpoint ##################
def model_fn(model_dir):
    """Load model"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(os.listdir(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model.to(device)

def input_fn(request_body, request_content_type):
    """An input_fn that loads model input data"""
    
    if request_content_type == "application/json":
        data = json.loads(request_body)
        
        if isinstance(data, str):
            data = [data]
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            pass
        else:
            raise ValueError("Unsupported input type. Input type can be a string or an non-empty list. \
                             I got {}".format(data))
                       
        tokenized_inputs = tokenizer.batch_encode_plus(data, max_length=args.max_seq_length, padding='max_length', truncation=True, return_tensors="pt")
        tokenized_inputs.pop("token_type_ids")
        
        return tokenized_inputs
    raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    """Model prediction for a single input"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    sm = torch.nn.Softmax(dim=1)
    input_data = input_data.to(device)
    with torch.no_grad():
        output = model(**input_data)
        
        output = sm(output['logits'])
        y = output.detach().numpy()[0]

    return y

if __name__ == "__main__":
    train(args)
