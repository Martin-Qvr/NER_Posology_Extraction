from transformers import CamemBertForTokenClassification, AdamW, CamemBertTokenizer, BertConfig, get_linear_schedule_with_warmup


tokenizer = CamemBertTokenizer.from_pretrained('camembert-base-cased', do_lower_case=False)



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
