import jsonlines
import pandas as pd

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


def create_new_dictionary(id: int, text: str, meta: dict, label: list, comments=[]) -> dict:
    return {
          'id': id
        , 'text': text
        , 'meta': meta
        , 'label': label
        , 'Comments': comments
    }


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

def get_data(filepath):
    data = load_jsonl(filepath)
    new_data = list()

    for element in data:
        text = element['text']
        label = element['label']
        enriched_label = append_labelized_sequences_to_label(element)
        text_split = text.split('.')
        offset = 0
        used_labels = list()
        for sentence in text_split:
            new_label = list()
            for labelized_sequence in [el for el in enriched_label if el not in used_labels and len(el[-1])>1]:
                if labelized_sequence[-1] in sentence:
                    new_start, new_end = labelized_sequence[0] - offset, labelized_sequence[1] - offset
                    if new_end < len(sentence):
                        new_label.append([new_start, new_end, labelized_sequence[2]])
                        used_labels.append(labelized_sequence)
            sentence_element = create_new_dictionary(id=element['id'], text=sentence, meta=element['meta'], label=new_label, comments=[])
            offset += len(sentence) + 1
            new_data.append(sentence_element)
    return pd.DataFrame(new_data)