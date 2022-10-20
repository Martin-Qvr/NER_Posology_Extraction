from transformers import pipeline
import pandas as pd
import random
from nltk.tokenize import sent_tokenize


def split_in_sentences(text: str, labels: list):
    # replacing newlines by dots so char length does not differ after tokenizing by sentences
    text = text.replace("\n", ".")
    sentences = sent_tokenize(text)
    sentences_indices = []
    start = 0
    for sentence in sentences :
        # the +1 accounts for the space after a dot which is not taken into account otherwise
        end = start + len(sentence) + 1
        sentences_indices.append((start, end))
        start = end
    sentences_indices = list(zip(sentences, sentences_indices))

    sentences_labels = []
    for tup in sentences_indices:
        label_sentence = (tup[0],
        [[lab[0]-tup[1][0], lab[1]-tup[1][0], lab[2]] for lab in labels if (lab[0] >= tup[1][0] and lab[1] <= tup[1][1])])
        sentences_labels.append(label_sentence)

    # is of type (sentence, list of [start (in sentence), end (in sentence), label])

    return sentences_labels


def man_replace(text: str, labels: list):
    to_keep = [text[lab[0]:lab[1]] for lab in labels]
    to_replace = [tk.replace(' ', '##') for tk in to_keep]
    for i, tk in enumerate(to_keep):
        text = text.replace(tk, to_replace[i])
    return text


def create_text_to_mask(text: str, labels: list, replacement_ratio: float = 0.1):

    words_to_keep_labeled = [(text[f[0]:f[1]], f[2]) for f in labels]
    words_to_keep_labeled = [(f[0].strip().replace(' ', '##'), f[1]) for f in words_to_keep_labeled]
    words_to_keep_labeled_punct = [(w[0] + ".", w[1]) for w in words_to_keep_labeled] + [(w[0] + ",", w[1]) for w in words_to_keep_labeled]
    words_to_keep_labeled = words_to_keep_labeled + words_to_keep_labeled_punct
    words_to_keep = [t[0] for t in words_to_keep_labeled]
    words_to_keep_labeled = dict(words_to_keep_labeled)

    text_man = man_replace(text, labels)

    text_list = text_man.split()
    ind_to_keep = [ind for ind, word in enumerate(text_list) if word in words_to_keep]
    labeled_words_ind = [(word, ind, words_to_keep_labeled[word]) for ind, word in enumerate(text_list) if word in words_to_keep]
    to_mask = random.sample(range(len(text_list)), int(len(text_list) * replacement_ratio))
    to_mask.sort()
    to_mask = [ind for ind in to_mask if ind not in ind_to_keep]
    masked_words_ind = [(text_list[ind], ind) for ind in to_mask]

    new_text_list = [word if ind not in to_mask else "<mask>" for ind, word in enumerate(text_list)]
    new_text = ' '.join(new_text_list)
    new_text = new_text.replace("##", ' ')

    return (new_text, masked_words_ind, labeled_words_ind)


def randomly_replace_with_synonyms_sentence(masked_sentence: str, masked_words_ind: list, labeled_words_ind: list, unmasker, replacement_ratio: float = 0.1):

    if "<mask>" not in masked_sentence:
        return (masked_sentence, labeled_words_ind)

    suggested_synonyms = unmasker(masked_sentence)
    word_replacement = []
    for i, tup in enumerate(masked_words_ind):
        masked_word = tup[0]
        ind = tup[1]
        if type(suggested_synonyms[i]) == list:
            # this handles the case where there are multiple masked words in the sentence
            word_synonyms = [d['token_str'] for d in suggested_synonyms[i] if d['token_str'] != masked_word]
        else :
            # this handles the case where there is only one masked word in the sentence
            word_synonyms = [d['token_str'] for d in suggested_synonyms if d['token_str'] != masked_word]
        if len(word_synonyms)==0:
            replacement = masked_word
        else:
            # word synonyms are naturally orderered from the most likely to the least likely
            replacement = word_synonyms[0]
        word_replacement.append((masked_word, replacement, ind))

    new_sentence = masked_sentence
    for tup in word_replacement:
        replacement = tup[1]
        ind = tup[2]
        new_sentence =  new_sentence.replace("<mask>", replacement, 1)


    return (new_sentence, labeled_words_ind)


def randomly_replace_with_synonyms_full_text(text: str, labels: list, unmasker, replacement_ratio: float = 0.1):

    sentences_labels = split_in_sentences(text, labels)

    sentences_list = []
    labeled_words_new = []
    for tup in sentences_labels:
        text = tup[0]

        labels = tup[1]
        masked_sentence, masked_words_ind, labeled_words_ind = create_text_to_mask(text, labels, replacement_ratio=replacement_ratio)
        sentences_list.append(randomly_replace_with_synonyms_sentence(masked_sentence=masked_sentence, 
                                                                masked_words_ind=masked_words_ind, 
                                                                labeled_words_ind=labeled_words_ind, 
                                                                unmasker=unmasker,
                                                                replacement_ratio=replacement_ratio))
        full_text = ' '.join([tup[0] for tup in sentences_list])
        labeled_words_new += labeled_words_ind
    word_label_dict = dict([(tup[0].replace("##", ' '), tup[2]) for tup in labeled_words_new])
    labeled_words = []
    for tup in sentences_list:
        labeled_words += [(t[0].replace('##', ' '), word_label_dict[t[0].replace('##', ' ')]) for t in tup[1]]

    labels = []
    for tup in labeled_words:
        word = tup[0]
        label = tup[1]
        ind = full_text.find(word)
        labels.append([ind, ind + len(word), label])
        
    return (full_text, labels)

def create_new_samples(json_path: str, replacement_ratio: float = 0.1):
    file_path = json_path
    origin_df = pd.read_json(path_or_buf=file_path, lines=True)
    unmasker = pipeline('fill-mask', model='xlm-roberta-base')
    n_lines = origin_df.shape[0]
    output_dict = {'text': [], 'labels': []}

    for i in range(n_lines):
        text = origin_df['text'].iloc[i]
        labels = origin_df['label'].iloc[i]
        full_text, labels = randomly_replace_with_synonyms_full_text(text=text, labels=labels, unmasker=unmasker, replacement_ratio=replacement_ratio)
        output_dict['text'].append(full_text)
        output_dict['labels'].append(labels)

    return pd.DataFrame(output_dict)