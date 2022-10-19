import pandas as pd


def jsonl_to_dataframe(jsonl_filepath):
    df = pd.read_table(jsonl_filepath, header=None)
    columns = ["id", "text", "label"]
    for col in columns:
        df[col] = df[0].apply(lambda x: eval(x)[col])
    return df[columns]


def augment_data(df):
    return df


def normalize_text(text):
    return text


def pre_tokenize(text, label):
    text = text.replace("\\", "")
    text = text.replace("  ", " ")
    text = text.replace("/", " ")
    text = text.replace('\n','')
    text = text.split(" ")
    return [(text[t], label) for t in range(len(text))]


def add_trailing_label(text, label_list):
    label_list.append([label_list[-1][1], len(text), "O"])
    label_list.sort()
    return label_list


def add_heading_label(label_list):
    label_list.append([0, label_list[0][0], "O"])
    label_list.sort()
    return label_list


def add_in_between_label(label_list):
    for i in range(len(label_list)-1):
        label_list.append([label_list[i][1], label_list[i+1][0], "O"])
    label_list.sort()
    return label_list


def fill_with_null_labels(text, label_list):
    return add_trailing_label(text, add_heading_label(add_in_between_label(label_list)))


def reformat_doccano_output(df):
    # Offset by one, doccano indices output are not exact
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