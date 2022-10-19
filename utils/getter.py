import pandas as pd


def jsonl_to_dataframe(jsonl_filepath):
    df = pd.read_table(jsonl_filepath, header=None)
    columns = ["id", "text", "label"]
    for col in columns:
        df[col] = df[0].apply(lambda x: eval(x)[col])
    return df[columns]