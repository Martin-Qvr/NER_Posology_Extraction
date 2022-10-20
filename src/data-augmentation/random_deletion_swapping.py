# -*- coding: utf-8 -*-

import json
import random

import jsonlines
import pandas as pd

# jsonl and dictionary manipulation


def load_jsonl(file_path: str) -> list:
    """
    Loading function: it loads the jsonl file
    
    Args:
        file_path: File path
    
    Returns: List of dictionaries
    """
    data = list()
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data


def extract_labelized_sequences(element: dict) -> list:
    """
    Extracts labeled words (element['label']) from the text (element['text']) of line of the
    jsonl file. Necessary to relabel the data-augmented text.
    
    Args:
        element: A line of the jsonl file
    
    Returns: The labeled words
    """
    labelized_sequences = list()
    if len(element['label']) > 0:
        for label in element['label']:
            labelized_sequences.append(element['text'][label[0]: label[1]])
    return labelized_sequences


def append_labelized_sequences_to_label(element: dict) -> list:
    """
    Appends the labelized_sequences (from extract_labelized_sequences()) to element['label']
    
    Args:
        element: A line of the jsonl file
    
    Returns: The extended label list
    """
    labelized_sequences = extract_labelized_sequences(element)
    enriched_label = list()
    for label, sequence in zip(element['label'], labelized_sequences):
        new_label = label + [sequence]
        enriched_label.append(new_label)
    return enriched_label


def add_enriched_label_to_element(element: dict) -> dict:
    """
    Adds the enriched_label (from append_labelized_sequences_to_label()) as a new item 
    to a dictionary (a jsonl line)
    
    Args:
        element: A line of the jsonl file
    
    Returns: The input dict with the enriched_label item. 
    """
    enriched_element = element.copy()
    enriched_label = append_labelized_sequences_to_label(element)
    enriched_element['enriched_label'] = enriched_label
    return enriched_element


def concatenate_a_sequence_in_a_text(text: str, sequence: str, concatenator='#') -> str:
    """
    Concatenates a sequence of words into one. Necessary not too modify the labeled sequences.
    
    Args:
        text: The text of a dictionary (element['text'])
        sequence: The sequence to concatenate
    
    Returns: The input text with the original sequence concatenated
    """
    if len(sequence.split(' ')) > 1:
        concatenated_sequence = sequence.replace(' ', concatenator)
        concatenated_text = text.replace(sequence, concatenated_sequence)
    else: 
        concatenated_text = text
    return concatenated_text


def add_text_for_data_aug_to_element(enriched_element: dict) -> dict:
    """
    Adds the concatenated_text (from concatenate_a_sequence_in_a_text()) as a new item 
    to the enriched_element (input dict with the enriched_label item in it)
    
    Args:
        enriched_element: The input dict with the enriched_label item in it 
    
    Returns: The input dict with the enriched_label and concatenated_text items
    """
    element = enriched_element.copy()
    new_text = element['text']
    if 'enriched_label' not in element.keys():
        raise KeyError('enriched_label should be added first')
    for label in element['enriched_label']:
        new_text = concatenate_a_sequence_in_a_text(new_text, label[-1])
    element['text_for_data_aug'] = new_text
    return element


def create_new_dictionary(id: int, text: str, label: list, comments=[]) -> dict:
    return {
          'id': id
        , 'text': text
        , 'label': label
        , 'Comments': comments
    }


# data augmentation functions


def random_deletion(text: str, p: float) -> str:
    """
    Randomly deletes the words of a text with a probability of p.
    A word will be deleted if a uniformly generated number between 0 and 1 is smaller than 
    a pre-defined threshold. This allows for a random deletion of some words of the sentence.
    
    Args:
        text: The text on which to perform random deletion
        p: The probability for each word to be deleted
    
    Returns: The ramdomly deleted text
    """
    text = text.split()
    # if only one word, don't delete it
    if len(text) == 1:
        return text
    # randomly delete text with probability p
    new_text = []
    for word in text:
        if '.' not in word:
            r = random.uniform(0, 1)
            if r > p:
                new_text.append(word)
        else:
            new_text.append(word)
    # if end up deleting all text, just return a random word
    if len(new_text) == 0:
        rand_int = random.randint(0, len(text)-1)
        return [text[rand_int]]
    text_with_deletion = ' '.join(new_text)
    return text_with_deletion


def swap_word(text: str) -> str:
    """
    Randomly swaps 2 words in a text
    
    Args:
        text: The text on which is performed 1 random swapping
    
    Returns: The once-ramdomly-swapped text
    """
    random_idx_1 = random.randint(0, len(text)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(text)-1)
        counter += 1
        if counter > 3:
            return text
    text[random_idx_1], text[random_idx_2] = text[random_idx_2], text[random_idx_1] 
    return text


def random_swap(text: str, p: float) -> str:
    """
    Randomly swaps p % of the words in a text.
    
    Args:
        text: The text on which to random swapping
        p: The proportion of words to be swapped
    
    Returns: The ramdomly swapped text
    """
    n = int(len(text) * p / 2)
    text = text.split()
    new_text = text.copy()
    for _ in range(n):
        new_text = swap_word(new_text)
    new_text = ' '.join(new_text)
    return new_text


def generate_new_label(new_text: str, enriched_element: dict) -> list:
    """
    Generates the new label list for the randomly deleted/swapped text. Necassary as the text
    and changed, the indexes may have changed too.
    
    Args:
        new_text: The text on which a random operation (deletion or swapping) was performed
        enriched_element : The input dict with the enriched_label and concatenated_text items
    
    Returns: The new label list [start_idx, end_idx, label]
    """
    new_label = list()
    for label in enriched_element['enriched_label']:
        if label[-1] in new_text:
            new_label.append([new_text.find(label[-1]), new_text.find(label[-1]) + len(label[-1]), label[-2]])
    return new_label


def generate_a_swap_element(element: dict, p: float) -> dict:
    """
    Generates a new dictionary with the swapping-augmented data - updated id, text and label.
    
    Args:
        element: A line of the jsonl file
        p: The proportion of words to be swapped
    
    Returns: New dictionary with the swapping-augmented data - updated id, text and label.
    """
    enriched_element = add_enriched_label_to_element(element)
    enriched_element = add_text_for_data_aug_to_element(enriched_element)
    text = random_swap(enriched_element['text_for_data_aug'], p).replace('#', " ")
    label = generate_new_label(text, enriched_element)
    # return create_new_dictionary(id=element['id'], text=text, meta=element['meta'], label=label, comments=element['Comments'])
    return create_new_dictionary(id=element['id'], text=text, label=label, comments=element['Comments'])


def generate_n_swap_elements(data: list, n: int, p: float) -> list:
    """
    Generates n new dictionaries with swapping-augmented data - updated id, text and label.
    
    Args:
        data: input list of dictionaries (from load_jsonl())
        n: The number of new dictionaries
        p: The proportion of words to be swapped
    
    Returns: New list of n dictionaries with swapping-augmented data
    """
    elements = random.sample(data, n)
    swap_elements = list()
    for element in elements:
        swap_elements.append(generate_a_swap_element(element, p))
    return swap_elements


def generate_a_deletion_element(element: dict, p: float) -> dict:
    """
    Generates a new dictionary with the deletion-augmented data - updated id, text and label.
    
    Args:
        element: A line of the jsonl file
        p: The proportion of words to be deleted
    
    Returns: New dictionary with the deletion-augmented data
    """
    enriched_element = add_enriched_label_to_element(element)
    enriched_element = add_text_for_data_aug_to_element(enriched_element)
    text = random_deletion(enriched_element['text_for_data_aug'], p).replace('#', " ")
    label = generate_new_label(text, enriched_element)
    # return create_new_dictionary(id=element['id'], text=text, meta=element['meta'], label=label, comments=element['Comments'])
    return create_new_dictionary(id=element['id'], text=text, label=label, comments=element['Comments'])


def generate_n_deletion_elements(data: list, n: int, p: float) -> list:
    """
    Generates n new dictionaries with deletion-augmented data - updated id, text and label.
    
    Args:
        data: input list of dictionaries (from load_jsonl())
        n: The number of new dictionaries
        p: The proportion of words to be deleted
    
    Returns: New list of n dictionaries with swapping-augmented data
    """
    elements = random.sample(data, n)
    del_elements = list()
    for element in elements:
        del_elements.append(generate_a_deletion_element(element, p))
    return del_elements


# perform random deletion and random swapping


def to_jsonl(data: pd.DataFrame, new_jsonl_name: str):

    """
    Converts pandas DataFrame to JSON file

    Args:
        data: Pandas DataFrame to export in jsonl
        new_jsonl_name: The name of the output jsonl file
    
    Returns: None
    """

    with open(new_jsonl_name, 'w', encoding='utf-8') as file:
        data.to_json(file, force_ascii=False, orient='records', lines=True)


def perform_random_swapping(file_path: str, n: int, p=0.2) -> None:
    """
    Generates a pandas DataFrame with n new random swaps observations
    
    Args:
        file_path: File path
        n: The number of new dictionaries
        p: The proportion of words to be swapped

    
    Returns: pd.DataFrame
    """
    data = load_jsonl(file_path)
    n_swap_elements = generate_n_swap_elements(data, n, p)
    return pd.DataFrame(n_swap_elements)


def perform_random_deletion(file_path: str, n: int, p=0.3) -> None:
    """
    Generates a pandas DataFrame with n new random swaps observations
    
    Args:
        file_path: File path
        n: The number of new dictionaries
        p: The proportion of words to be deleted
    
    Returns: pd.DataFrame
    """
    data = load_jsonl(file_path)
    n_deletion_elements = generate_n_deletion_elements(data, n, p)
    return pd.DataFrame(n_deletion_elements)
