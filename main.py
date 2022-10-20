import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForTokenClassification, AdamW, BertTokenizer, BertConfig, get_linear_schedule_with_warmup
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from seqeval.metrics import accuracy_score
import seaborn    
from sklearn import metrics
from sklearn.metrics import f1_score
from tqdm import trange

from utils import getter, augmentation, preprocessing, vizualisation


with open("config.yaml") as f:
    config = yaml.safe_load(f)

###############################################
################## GET DATA ###################
###############################################
df = getter.get_data(config["jsonl_filepath"])

########################################################
################## DATA AUGMENTATION ###################
########################################################
print("Starting data augmentation...")

# to be updated 
# with augmentation functions
df = augmentation.augment_data(df)
# df.loc[:,"has_label"] = df['label'].apply(lambda x: True if len(x)>0 else False)
# df = df.loc[df['has_label'] == True]
# df
#########################################################
################## DATA PREPROCESSING ###################
#########################################################
print("Starting data preprocessing...")

df = preprocessing.reformat_doccano_output(df)

df = preprocessing.pre_tokenize(df)


# to be updated
# Keeping rows with less than 512 pre-tokens because 512 is the max
df["len_pre-tokens"] = df["pre-tokens"].apply(lambda x: len(x))
df.loc[df["len_pre-tokens"] < 512]["len_pre-tokens"]
print(df)
sentences = [row for row in df["pre-tokens"].values]
labels = [row for row in df["labels"].values]

tag2idx = config["tag_values"]
tag2idx = {v: k for k, v in tag2idx.items()}

# to be updated
# Is it necessary to add "PAD" ?
tag2idx["PAD"] = 8
tag_values = list(tag2idx.values())

MAX_LEN = config["MAX_LEN"]
bs = config["bs"]

# to be updated
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#n_gpu = torch.cuda.device_count()
#print(f'Number of GPUs :{n_gpu}')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(sentences, labels)
]


tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")

attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

### Load the inputs to the GPU
tr_inputs.to(device)
val_inputs.to(device)
tr_tags.to(device)
val_tags.to(device)
val_masks.to(device)


train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)


#####################################################
#################### MODEL SETUP ####################
#####################################################
print("Starting model setup...")

model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(tag2idx),
    output_attentions = False,
    output_hidden_states = False
)

model = model.to(device)

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)

epochs = 15
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

########################################################
#################### MODEL TRAINING ####################
########################################################
print("Starting model training...")

# Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []

for _ in trange(epochs, desc="Epoch"):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        loss = outputs[0]
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)


    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags), average=None))



# # ========================================
# #               Save model
# # ========================================
# # 
# torch.save(model, "./models/NER_Bert.pt")
# print("Model saved!")
confusion_matrix =  seaborn.heatmap(metrics.confusion_matrix(pred_tags, valid_tags))
confusion_matrix.savefig('confusion_matrix.png', dpi=400)

vizualisation.plot_learning_curve(loss_values, validation_loss_values)


# # ========================================
# #          Generate submission CSV
# # ========================================

import torch
import pandas as pd
import yaml
import numpy as np

from utils import preprocessing
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

with open("config.yaml") as f:
    config = yaml.safe_load(f)

#########################################
########## LOAD DATA AND MODEL ##########
#########################################
print("Loading data and model ...")
    
### INPUT HERE ###
test_csv_filepath = ""
submission_csv_filepath = ""

test = pd.read_csv(test_csv_filepath)

def testcsv_to_sentences(test_df, sentences_len=100):
    # change type for text concat
    test_df["TokenId"] = test_df["TokenId"].astype(str)
    
    # concat by sentences
    records = pd.DataFrame()
    records["tokens"] = list(test_df.groupby("sentence_id")["token"].apply(list).values)
    records["tokensId"] = list(test_df.groupby("sentence_id")["TokenId"].apply(list).values)
    
    # split a tokens list every  elements
    records["tokens"] = records["tokens"].apply(lambda x: [x[i*sentences_len:(i+1)*sentences_len] for i in range((len(x)//sentences_len)+1)])
    records["tokensId"] = records["tokensId"].apply(lambda x: [x[i*sentences_len:(i+1)*sentences_len] for i in range((len(x)//sentences_len)+1)])
    
    # expand vertically when there are more than one sentence part
    sentences = pd.DataFrame()
    sentences["tokens"] = records[["tokens"]].explode("tokens")["tokens"]
    sentences["tokensId"] = records[["tokensId"]].explode("tokensId")["tokensId"]

    return sentences

test_sentences = list(testcsv_to_sentences(test, 60)["tokens"])
tokens_id = list(testcsv_to_sentences(test, 60)["tokensId"])

######################################
########## TOKENIZE AND PAD ##########
######################################
print("Tokenizing and padding ...")

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


all_token_ids = []
for sentence_ids in tokenized_sentences_ids:
    for token_id in sentence_ids:
        all_token_ids.append(int(token_id))

missing_token_ids = list(set(list(range(3557))) - set(all_token_ids))
print("There are {} missing tokens".format(len(missing_token_ids)))

MAX_LEN = config["MAX_LEN"]
bs = config["bs"]

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_sentences],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

attention_mask = [[float(i != 0.0) for i in ii] for ii in input_ids]

###############################
########## TO TENSOR ##########
###############################
print("Converting to tensors ...")

test_data = torch.tensor(input_ids)
test_masks = torch.tensor(attention_mask)

test_dataset = TensorDataset(test_data, test_masks)
test_sampler = SequentialSampler(test_dataset)
valid_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=bs)

#####################################
########## MAKE PREDICTION ##########
#####################################
print("Making the predictions ...")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model.eval()
predictions, true_labels = [], []

for batch in valid_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask = batch

    with torch.no_grad():
        # BERT model
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        # other model
        #outputs = model(b_input_ids, attention_mask=b_input_mask)
        
        logits = outputs[0].detach().cpu().numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

########################################
########## PREDICTIONS TO CSV ##########
########################################
print("Saving predictions as csv ...")

new_tokens, new_labels, new_tokens_ids = [], [], []
for sentence, sentence_predictions, sentence_tokens_ids in zip(tokenized_sentences, predictions, tokenized_sentences_ids):
    for token, prediction, token_id in zip(sentence, sentence_predictions, sentence_tokens_ids):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(prediction)
            new_tokens.append(token)
            new_tokens_ids.append(token_id)


predictions = pd.DataFrame([new_tokens_ids, new_tokens, new_labels]).T
predictions = predictions.rename(columns={"0": "TokenId", "1": "Token", "2":"Predicted", 0: "TokenId", 1: "Token", 2:"Predicted"})
predictions = predictions.drop_duplicates(subset=["TokenId"])
predictions["TokenId"] = predictions["TokenId"].astype(int)

test = pd.read_csv(test_csv_filepath)
test["TokenId"] = test["TokenId"].astype(int)

df = test.merge(predictions, how="left", on="TokenId")
df["Predicted"] = df["Predicted"].replace(to_replace=8, value=5)
df["Predicted"] = df["Predicted"].fillna(5).astype(int).astype(str)
df = df[["TokenId", "Predicted"]]

df.to_csv(submission_csv_filepath, index=False)
