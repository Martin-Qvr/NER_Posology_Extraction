import torch
import numpy as np

from torch.utils.data import DataLoader, SequentialSampler, TensorDataset


def testcsv_to_sentences(test_csv_filepath, sentences_len=100):
    """
    Transforms test.csv into DataFrame. One row contains a subset of of a sentence of size sentences_len at most.
    """
    test_df = pd.read_csv(test_csv_filepath)

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

    # convert both tokens and tokens ids as list of lists
    test_sentences = list(sentences["tokens"])
    tokens_id = list(sentences["tokensId"])

    return test_sentences, tokens_id


def tokenize_and_adjust_ids(tokenizer, test_sentences, tokens_id):
    """
    Tokenizes a sentence, and offsets the tokens ids in accordance with the number of subtokens generated.
    """
    tokenized_sentences = []
    tokenized_sentences_ids = []

    # loop through sentences
    for sentence, sentence_ids in zip(test_sentences, tokens_id):
        tokenized_sentence = []
        tokenized_sentence_id = []

        # loop thourgh pre-tokens of a sentence
        for word, id in zip(sentence, sentence_ids):
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            tokenized_sentence.extend(tokenized_word)
            tokenized_sentence_id.extend([id] * n_subwords)
            
        tokenized_sentences.append(tokenized_sentence)
        tokenized_sentences_ids.append(tokenized_sentence_id)

    return tokenized_sentences, tokenized_sentences_ids


def check_missing_tokens(tokenized_sentences_ids, test_csv_length=3557):
    """
    Looks for missing ids in tokens ids, and prints the test tokens ids that are missing.
    Typically, these are \n and \t
    """
    all_token_ids = []
    for sentence_ids in tokenized_sentences_ids:
        for token_id in sentence_ids:
            all_token_ids.append(int(token_id))

    missing_token_ids = list(set(list(range(3557))) - set(all_token_ids))
    print("There are {} missing tokens".format(len(missing_token_ids)))


def lists_to_tensors(input_ids, attention_mask):
    """
    Transforms list of tokens ids and the attention mask to hugging face DataLoader object suit for model prediction.
    """
    test_data = torch.tensor(input_ids)
    test_masks = torch.tensor(attention_mask)

    test_dataset = TensorDataset(test_data, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    valid_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=bs)

    return valid_dataloader


def make_predictions(device, dataloader, model):
    """
    Makes a class predictions for all data contained in DataLoader object.
    """
    predictions = []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            # BERT model
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            
            # CamemBert model
            #outputs = model(b_input_ids, attention_mask=b_input_mask)
            
            # keep class id with largest probability (logit)
            logits = outputs[0].detach().cpu().numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

    return predictions


def subtokens_to_tokens(tokenized_sentences, predictions, tokenized_sentences_ids):
    """
    Keeps the class predicted for the first subtoken for each token.
    """
    new_tokens, new_labels, new_tokens_ids = [], [], []
    for sentence, sentence_predictions, sentence_tokens_ids in zip(tokenized_sentences, predictions, tokenized_sentences_ids):
        for token, prediction, token_id in zip(sentence, sentence_predictions, sentence_tokens_ids):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(prediction)
                new_tokens.append(token)
                new_tokens_ids.append(token_id)

    return new_tokens, new_labels, new_tokens_ids


def prediction_lists_to_csv(new_tokens_ids, new_tokens, new_labels, test_csv_filepath):
    """
    Creates final csv submissions from predictions, and fixes some predictions (PAD and NaN)
    """
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
