import logging
import os

from sklearn.model_selection import train_test_split

from examples.semeval2019.config import TEMP_DIRECTORY, MODEL_TYPE, MODEL_NAME, SEED, config
from examples.semeval2019.reader import ner_concatenate
from src.ner.transformers.run_model import NERModel
import torch

from src.util.seed import seed_everything

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

seed_everything(SEED)

train = ner_concatenate("data/train")
dev = ner_concatenate("data/test")

logging.info("Training size is {}".format(train.shape[0]))

logging.info("Training size is {}".format(train.shape[0]))
logging.info("Dev size is {}".format(dev.shape[0]))

train["words"] = train["token"]
train["labels"] = train["tag"]

dev["words"] = dev["token"]
dev["labels"] = dev["tag"]

train = train[['sentence_id', 'words', 'labels']]
dev = dev[['sentence_id', 'words', 'labels']]

train['labels'] = train['labels'].str.strip()
dev['labels'] = dev['labels'].str.strip()

logging.info("Training size is {}".format(train.shape[0]))

tags = train['labels'].unique()

# Create a NER model
model = NERModel(MODEL_TYPE, MODEL_NAME, args=config, labels=tags,
                 use_cuda=torch.cuda.is_available())  # You can set class weights by using the optional weight argument

# Train the model
logging.info("Started Training")
if config["evaluate_during_training"]:
    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED, shuffle=False)
    model.train_model(train_df, eval_df=eval_df)

else:
    model.train_model(train)

logging.info("Finished Training")

logging.info("Started Evaluation")

if config["evaluate_during_training"]:
    model = NERModel(MODEL_TYPE, config["best_model_dir"], args=config, labels=tags,
                     use_cuda=torch.cuda.is_available())

result, model_outputs, predictions = model.eval_model(dev)
