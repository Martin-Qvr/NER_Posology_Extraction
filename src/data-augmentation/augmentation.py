import os

import pandas as pd
import yaml

from back_translation import perform_back_translation
from paraphrase_generator import new_dataframe_creation
from random_deletion_swapping import (perform_random_deletion,
                                      perform_random_swapping)
from sentence_masker import create_new_samples
from summarization_stat import augment_data_summarized

with open("./config.yaml") as f:
    config = yaml.safe_load(f)


def to_jsonl(data: pd.DataFrame, new_jsonl: str):

    """
    Pandas DataFrame to JSON file
    """

    with open(new_jsonl, 'w', encoding='utf-8') as file:
        data.to_json(file, force_ascii=False, orient='records', lines=True)


def augment_data(json_path: str,
                 data_augmented_path: str,
                 back_translation: bool=True,
                 rd_swapping: bool=True,
                 rd_deletion: bool=True,
                 paraphrase: bool=True,
                 synonyms: bool=True,
                 summarization:bool=False):
    
    """
    Generates a JSONL file with the chosen augmented data.

    Params
    ------

    json_path: srt, path to raw data
    back_translation: bool, activate back translation or not
    rd_swapping: bool, activate random swapping or not
    rd_deletion: bool, activate random deletion or not
    paraphrase: bool, activate paraphras or not
    synonyms: bool, activate synonyms or not
    summarization:bool, activate summarization or not

    Output
    ------

    Generate the 'jsonl' file to be used for NER model.
    
    """
    
    data = pd.read_json(json_path, lines=True)

    if back_translation:
        df_backtranslation = perform_back_translation(json_path)
        data = pd.concat([data, df_backtranslation])
        print(f" New data points from back translation : {len(df_backtranslation.index)} lines")

    if rd_deletion:
        df_rd_deletion = perform_random_deletion(json_path, n=config["n_deletion_swap"], p=config["prop_del"])
        data = pd.concat([data, df_rd_deletion])
        print(f" New data points from random deletion : {len(df_rd_deletion.index)} lines")

    if rd_swapping:
        df_rd_swapping = perform_random_swapping(json_path, n=config["n_deletion_swap"], p=config["prop_swap"])
        data = pd.concat([data, df_rd_swapping])
        print(f" New data points from random swapping : {len(df_rd_swapping.index)} lines")

    if paraphrase:
        df_paraphrase = new_dataframe_creation(json_path)
        data = pd.concat([data, df_paraphrase])
        print(f" New data points from paraphrase generation : {len(df_paraphrase.index)} lines")

    if synonyms:
        df_synonyms = create_new_samples(json_path, n_new_samples=2000)
        data = pd.concat([data, df_synonyms])
        print(f" New data points from synonyms generation : {len(df_synonyms.index)} lines")
    
    if summarization:
        df_summarization = augment_data_summarized(data)
        data = pd.concat([data, df_summarization])
        print(f" New data points from summarization : {len(df_synonyms.index)} lines")

    to_jsonl(data, data_augmented_path)
    
if __name__ == "__main__":
    os.system(f"curl {config['raw_data_dropbox']} -L -o raw_data.json") # Load raw data

    json_path = config["jsonl_filepath"]
    data_augmented_path = config["data_augmented_filepath"]

    augment_data(json_path)

    os.system("rm raw_data.json") # Remove raw data
