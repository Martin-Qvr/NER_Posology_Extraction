def normalize_text(text):
    return text


def add_trailing_label(text, label_list):
    """
    As part of doccano reformatting, adds the O label to all the unlabellised words from the last labellised word to the end.
    """
    label_list.append([label_list[-1][1], len(text), "O"])
    label_list.sort()
    return label_list


def add_heading_label(label_list):
    """
    As part of doccano reformatting, adds the O label to all the unlabellised words from the first word to the fist labellised word of the sentence.
    """
    label_list.append([0, label_list[0][0], "O"])
    label_list.sort()
    return label_list


def add_in_between_label(label_list):
    """
    As part of doccano reformatting, adds the O label to all the unlabellised words in between labellised words.
    """
    for i in range(len(label_list)-1):
        label_list.append([label_list[i][1], label_list[i+1][0], "O"])
    label_list.sort()
    return label_list


def fill_with_null_labels(text, label_list):
    """
    Takes a sentence and a label list, and fills it up with O labels for all unlabellised intervals.
    """
    return add_trailing_label(text, add_heading_label(add_in_between_label(label_list)))


def reformat_doccano_output(df):
    """
    Takes doccano output, and reformats it classical Bert Tokenizer input.
    """
    # Offset by one, because doccano indices output are not exact
    df["label"] = df["label"].apply(lambda x: [[x[i][0]+1, x[i][1]+1, x[i][2]] for i in range(len(x))])

    # Add null labels for the rest of the text
    df["label_full"] = df\
        .apply(lambda x: [(0, len(x["text"]), "O")] if x["label"] == [] else fill_with_null_labels(x["text"], x["label"]), 
            axis=1)

    # Rename label columns and drop old one
    df["label_raw"] = df["label"]
    df["label"] = df["label_full"]
    df.drop(columns="label_full", inplace=True)

    # Split text and match each part with its label
    df["text_labellised"] = df\
        .apply(lambda x: [[x["text"][i: j], label_id] for i, j, label_id in x["label"]], axis=1)

    # Rename text columns and drop old one
    df["text_raw"] = df["text"]
    df["text"] = df["text_labellised"]
    df.drop(columns="text_labellised", inplace=True)

    return df


def pre_tokenize(df):
    """
    Applies pre_tokenize_text to all the sentence parts contained in a DataFrame's text column.
    """
    df["text"] = df["text"].apply(lambda x: [pre_tokenize_sentence(x[i][0], x[i][1]) for i in range(len(x))])
    df["text"] = df["text"].apply(lambda x: sum(x, []))

    df["pre-tokens"] = df["text"].apply(lambda x: [x[i][0] for i in range(len(x))])
    df["labels"] = df["text"].apply(lambda x: [x[i][1] for i in range(len(x))])
    return df


def pre_tokenize_sentence(sentence, label):
    """
    Cleans text and splits a sentence into list of words.
    """
    sentence = sentence.replace("\\", "")
    sentence = sentence.replace("  ", " ")
    sentence = sentence.replace("/", " ")
    sentence = sentence.replace('\n','')
    sentence = sentence.split(" ")
    return [(sentence[t], label) for t in range(len(sentence))]

def tokenize_and_preserve_labels(tokenizer, sentence, labels):
    """
    Performs tokenization on a sentence (list of words), and extends the label to all subwords resulting from tokenisation
    """
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

def tokenize(tokenizer, sentences, labels):
    """
    Performs tokenization on a list of sentences, and returns tokenized sentences with labels extended.
    """
    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(tokenizer, sentence, label)
        for sentence, label in zip(sentences, labels)
    ]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    return tokenized_texts, labels