import torch
import pandas as pd
import yaml
import numpy as np
import csv

from utils import preprocessing
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

with open("config.yaml") as f:
    config = yaml.safe_load(f)

#########################################
########## LOAD DATA AND MODEL ##########
#########################################
    
model = torch.load("models/NER_Bert (1).pt", map_location=torch.device('cpu'))
test = pd.read_csv("data/test.csv")

test["TokenId"] = test["TokenId"].astype(str)

tokens_id = list(test.groupby("sentence_id")["TokenId"].apply(list).values)
test_sentences = list(test.groupby("sentence_id")["token"].apply(list).values)

######################################
########## TOKENIZE AND PAD ##########
######################################

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

tokenized_sentences = []
tokenized_sentences_ids = []
for sentence, sentence_ids in zip(test_sentences, tokens_id):
    tokenized_sentence = []
    tokenized_sentence_id = []
    
    for word, id in zip(sentence, sentence_ids):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        tokenized_sentence.extend(tokenized_word)
        tokenized_sentence_id.extend([id] * n_subwords)
        
    tokenized_sentences.append(tokenized_sentence)
    tokenized_sentences_ids.append(tokenized_sentence_id)
    
MAX_LEN = config["MAX_LEN"]
bs = config["bs"]

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_sentences],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

attention_mask = [[float(i != 0.0) for i in ii] for ii in input_ids]

###############################
########## TO TENSOR ##########
###############################

test_data = torch.tensor(input_ids)
test_masks = torch.tensor(attention_mask)

test_dataset = TensorDataset(test_data, test_masks)
test_sampler = SequentialSampler(test_dataset)
valid_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=bs)

#####################################
########## MAKE PREDICTION ##########
#####################################

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model.eval()
predictions, true_labels = [], []

for batch in valid_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask = batch

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)        
        logits = outputs[0].detach().cpu().numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

########################################
########## PREDICTIONS TO CSV ##########
########################################

#print("len(predictions[0]")
#print(len(predictions[0]))

#print("len(tokens_id[0])")
#print(len(tokens_id[0]))

#print("len(tokenized_sentences_ids[0])")
#print(len(tokenized_sentences_ids[0]))

df_predictions = pd.DataFrame(predictions)
