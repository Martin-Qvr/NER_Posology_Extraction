import pandas as pd
import yaml

import random_deletion_swapping
from back_translation import back_translation

with open("config.yaml") as f:
    config = yaml.safe_load(f)


def to_jsonl(data: pd.DataFrame, new_jsonl: str):

    """
    Pandas DataFrame to JSON file
    """

    with open(new_jsonl, 'w', encoding='utf-8') as file:
        data.to_json(file, force_ascii=False, orient='records', lines=True)


#### TO DO : VERIFIER QUE NOS COLONNES DE DATA FRAME SONT HARMONIEUSE #####
def augment_data(json_path: str,
                 data_augmented_path: str,
                 back_translation: bool,
                 rd_swapping: bool,
                 rd_deletion: bool,
                 n: int,
                 prop_swap: float,
                 prop_del: float,
                 paraphrase: bool,
                 synonyms: bool,
                 summarization:bool):
    
    """
    Generates a JSONL file with the chosen augmented data.

    Params
    ------

    json_path: srt, path to raw data
    back_translation: bool, activate back translation or not
    rd_swapping: bool, activate random swapping or not
    rd_deletion: bool, activate random deletion or not
    n: int, numbers of data for random swapping an deletion to add
    prop_swap: float, proportion of swapped words
    prop_del: float, proportion of deleted words
    paraphrase: bool, activate paraphras or not
    synonyms: bool, activate synonyms or not
    summarization:bool, activate summarization or not

    Output
    ------

    Generate the 'jsonl' file to be used for NER model.
    
    """
    
    data = pd.read_json("data.json", lines=True)

    if back_translation:
        df_backtranslation = back_translation(json_path)
        data = pd.concat([data, df_backtranslation])
    if rd_deletion:
        # Add Cam function
        data = pd.concat([data, df_rd_deletion])
    if rd_swapping:
        # Add Cam function
        data = pd.concat([data, df_rd_swapping])
    if paraphrase:
        # Add Sarah function
        data = pd.concat([data, df_paraphrase])
    if synonyms:
        # Add Nathan function
        data = pd.concat([data, df_synonyms])
    if summarization:
        # Add Charles G function
        data = pd.concat([data, df_summarization])
    
    # to_jsonl(data, f"data_augmented{'_bt' if back_translation else ''}{'_rdd' if rd_deletion else ''}{'_rds' if rd_swapping else ''}{'_para' if paraphrase else ''}{'_syn' if synonyms else ''}{'_sum' if summarization else ''}.jsonl")
    to_jsonl(data, data_augmented_path)
    
    if __name__ == "__main__":
        json_path = config["jsonl_filepath"]
        data_augmented_path = config["data_augmented.jsonl"]

        augment_data(json_path, data_augmented_path)