import pandas as pd
import yaml

from back_translation import perform_back_translation
from random_deletion_swapping import (perform_random_deletion,
                                      perform_random_swapping)

with open("../../config.yaml") as f:
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
                 back_translation: bool=True,
                 rd_swapping: bool=True,
                 rd_deletion: bool=True,
                 paraphrase: bool=True,
                 synonyms: bool=True,
                 summarization:bool=True):
    
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
        print(df_backtranslation.columns)
        print(len(df_backtranslation.index))

    if rd_deletion:
        df_rd_deletion = perform_random_deletion(json_path, n=config["n_deletion_swap"], p=config["prop_del"])
        data = pd.concat([data, df_rd_deletion])
        print(df_rd_deletion.columns)
        print(len(df_rd_deletion.index))

    if rd_swapping:
        df_rd_swapping = perform_random_swapping(json_path, n=config["n_deletion_swap"], p=config["prop_swap"])
        data = pd.concat([data, df_rd_swapping])
        print(df_rd_swapping.columns)
        print(len(df_rd_swapping.index))

    """
    if paraphrase:
        # Add Sarah function
        data = pd.concat([data, df_paraphrase])

    if synonyms:
        # Add Nathan function
        data = pd.concat([data, df_synonyms])

    if summarization:
        # Add Charles G function
        data = pd.concat([data, df_summarization])
    """
    # to_jsonl(data, f"data_augmented{'_bt' if back_translation else ''}{'_rdd' if rd_deletion else ''}{'_rds' if rd_swapping else ''}{'_para' if paraphrase else ''}{'_syn' if synonyms else ''}{'_sum' if summarization else ''}.jsonl")
    to_jsonl(data, data_augmented_path)
    
if __name__ == "__main__":

    json_path = config["jsonl_filepath"]
    data_augmented_path = config["data_augmented_filepath"]
    augment_data(json_path, data_augmented_path)
