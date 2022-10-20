import torch
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import json

import keras
import torch
import tensorflow
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModel, BigBirdPegasusForConditionalGeneration

# Setting up model and tokenizer

tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")
model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed", block_size=16, num_random_blocks=2)

## What was used locally
# Loading data 

data = pd.read_json('trainset.jsonl', lines=True)

# Summarizing data

def summarize_text(report):
    text = report
    inputs = tokenizer(text, return_tensors='pt')
    prediction = model.generate(**inputs)
    prediction = tokenizer.batch_decode(prediction)
    return(prediction)

# Converting labels

def convert_labelling(not_summarized_report, summarized_report, original_label_list):
    
    converted_labels = []
    
    if original_label_list == []:
        return([])
    
    else:
        for label in original_label_list:
            word_to_locate = not_summarized_report[label[0]:label[1]]
            start_new_pos = summarized_report[0].find(word_to_locate)
            end_new_pos = start_new_pos + len(word_to_locate)
            category = label[2]
            converted_label = [start_new_pos, end_new_pos, category]

            if converted_label[0] != -1:
                converted_labels.append(converted_label)
               
        return(converted_labels)



def augment_data_summarized(dataframe):
    
    # Dataframe must have columns text and labels
    
    dataframe['summarized_text'] = data.apply(lambda x: summarize_text(x.text), axis=1)
    dataframe['converted_labels'] = data.apply(lambda x: summarize_text(x.text, x.summarized_text, x.labels), axis=1)
    
    dataframe.drop(columns=['text', 'labels'], axis=1, inplace=True)
    
    dataframe.rename({"converted_labels" : "label"}, axis=1,inplace=True)
    dataframe.rename({"summarized_text" : "text"}, axis=1,inplace=True)

    return dataframe
    
    
    
    