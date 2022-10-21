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

import spacy
import textwrap
import fr_core_news_sm
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

# Setting up model and tokenizer

punctuation += '\n' 
stopwords = list(STOP_WORDS)

reduction_rate = 0.1  #defines how small the output summary should be compared with the input
nlp_pl = spacy.load('fr_core_news_sm') #process original text according with the Spacy nlp pipeline for french

## What was used locally
# Loading data 

data = pd.read_json('trainset.jsonl', lines=True)

# Defininf summarization protocols

def get_frequencies(document):
    word_frequencies = {}
    for word in document:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    max_frequency = max(word_frequencies.values())
    # print(max_frequency)

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]/max_frequency

    # print(word_frequencies)
    return(word_frequencies)

def get_sentence_scores(sentence_tok, word_frequencies, len_norm=True):
    sentence_scores = {}
    for sent in sentence_tok:
        if len(str(sent).strip()) != 0:
            word_count = 0
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    word_count += 1
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]
            if len_norm and len(sentence_scores.keys()) > 0 and sent in sentence_scores :
                sentence_scores[sent] = sentence_scores[sent]/word_count
    return sentence_scores

def get_summary(sentence_sc, rate):
    summary_length = int(len(sentence_sc)*rate)
    summary = nlargest(summary_length, sentence_sc, key = sentence_sc.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    return summary

# Summarizing data

def summarization_pipeline(text, reduction_rate):
    
    document = nlp_pl(text) #doc object
    tokens = [token.text for token in document] #tokenized text
    sentence_tokens = [sent for sent in document.sents]
    
    word_frequencies = get_frequencies(document)
    sentence_scores = get_sentence_scores(sentence_tokens,word_frequencies, len_norm=False) #sentence scoring without lenght normalization
    sentence_scores_rel = get_sentence_scores(sentence_tokens,word_frequencies, len_norm=True) #sentence scoring with length normalization
    
    summary_non_rel = get_summary(sentence_scores, reduction_rate)
    summary_rel = get_summary(sentence_scores_rel, reduction_rate)
    
    reduction_rate_temp = reduction_rate
    
    while (len(summary_non_rel) < 5) and (reduction_rate_temp < 1):
        reduction_rate_temp = reduction_rate_temp * 1.1
        summary_non_rel = get_summary(sentence_scores, reduction_rate)
        summary_rel = get_summary(sentence_scores_rel, reduction_rate)
    
    final_summary = summary_non_rel + ' ' + summary_rel

    return(final_summary)

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

# Dataframe augmentation

def augment_data_summarized(dataframe, rotation_rate=0.1 ):
    
    # Dataframe must have columns text and labels
    
    dataframe['summarized_text'] = data.apply(lambda x: summarization_pipeline(x.text, rotation_rate), axis=1)
    dataframe['converted_labels'] = data.apply(lambda x: convert_labelling(x.text, x.summarized_text, x.labels), axis=1)
    
    dataframe.drop(columns=['text', 'labels'], axis=1, inplace=True)
    
    dataframe.rename({"converted_labels" : "labels"}, axis=1,inplace=True)
    dataframe.rename({"summarized_text" : "text"}, axis=1,inplace=True)
    
    return dataframe
    
    
    
    