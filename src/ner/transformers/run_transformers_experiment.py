import glob
import logging
import os
import shutil

import torch
from sklearn.model_selection import train_test_split

from src.ner import TEMP_DIRECTORY, MODEL_TYPE, MODEL_NAME, global_args, \
    DEV_RESULT_FILE, RESULT_DIRECTORY, SUBMISSION_FILE
from src.ner import NERModel
from project_config import SEED, NER_TRAIN_FOLDER, \
    NER_DEV_FOLDER, NER_TEST_FOLDER
from util.logginghandler import TQDMLoggingHandler
from util.reader import ner_concatenate

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[TQDMLoggingHandler()])

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)
if not os.path.exists(os.path.join(TEMP_DIRECTORY, RESULT_DIRECTORY)): os.makedirs(
    os.path.join(TEMP_DIRECTORY, RESULT_DIRECTORY))

train = ner_concatenate(NER_TRAIN_FOLDER)
dev = ner_concatenate(NER_DEV_FOLDER)

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
model = NERModel(MODEL_TYPE, MODEL_NAME, labels=tags,
                 use_cuda=torch.cuda.is_available())  # You can set class weights by using the optional weight argument

# Train the model
logging.info("Started Training")
if global_args["evaluate_during_training"]:
    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED, shuffle=False)
    model.train_model(train_df, eval_df=eval_df)

else:
    model.train_model(train)

logging.info("Finished Training")

logging.info("Started Evaluation")

if global_args["evaluate_during_training"]:
    model = NERModel(MODEL_TYPE, global_args["best_model_dir"], args=global_args, labels=tags,
                     use_cuda=torch.cuda.is_available())

result, model_outputs, predictions = model.eval_model(dev)

flatten_predictions = list()
for i in predictions:
    for j in i:
        flatten_predictions.append(j)

logging.info(result)
logging.info("Dev size is {}".format(dev.shape[0]))
logging.info("Predictions length is {}".format(len(flatten_predictions)))
dev['predictions'] = flatten_predictions
dev.to_csv(os.path.join(TEMP_DIRECTORY, DEV_RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')

logging.info("Finished Evaluation")

logging.info("Started Testing")

test_files = glob.glob(os.path.join(NER_TEST_FOLDER, "*.deft"))
for test_file in test_files:
    logging.info("Predictions for {}".format(test_file))
    lines = []
    sentence = ""
    with open(test_file, 'rt') as fd:
        for line in fd:
            processed_line = line.split("\t")
            if len(processed_line) > 1:
                sentence = sentence + processed_line[0] + " "
            else:
                if len(sentence) > 1:
                    lines.append(sentence.strip())
                sentence = ""

    predictions, raw_outputs = model.predict(lines)

    flatten_predictions = list()
    for i in predictions:
        for j in i:
            flatten_predictions.append(j)

    logging.info("Predictions length {}".format(len(flatten_predictions)))
    logging.info("Test length {}".format(len(lines)))

    with open(test_file, 'rt') as ft:
        prediction_id = 0
        predicted_lines = []
        for line in ft:
            if len(line.split("\t")) > 1:
                term = line.split("\t")[0]

                try:
                    tag = flatten_predictions[prediction_id].get(term)
                except IndexError:
                    line = line.replace('\n', '\t') + 'O'

                if tag is None:
                    line = line.replace('\n', '\t') + 'O'

                else:
                    line = line.replace('\n', '\t') + str(tag)

                line = line.replace('OO', 'O')
                predicted_lines.append(line)
                prediction_id = prediction_id + 1

            else:
                predicted_lines.append("")

        with open(os.path.join(TEMP_DIRECTORY, RESULT_DIRECTORY, os.path.basename(test_file)), "w") as output:
            output.write("\n".join(predicted_lines))

logging.info("Finished Testing")

logging.info("Started Zipping")
shutil.make_archive(os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE), 'zip',
                    os.path.join(TEMP_DIRECTORY, RESULT_DIRECTORY))
logging.info("Finished Zipping")
