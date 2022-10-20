import re
from difflib import SequenceMatcher
from sys import argv

import goslate
import numpy as np
import pandas as pd
import translators as ts


def find_occurrences(s: str, ch: str) -> np.array:

    """
    Find all occurences of a character in a string an output the list of its position.
    """

    return np.array([0] + [i for i, letter in enumerate(s) if letter == ch])


def get_best_match(query: str, corpus: str, step=4, flex=3, case_sensitive=False, verbose=False) -> tuple:

    """Return best matching substring of corpus.

    Parameters
    ----------
    query : str
    corpus : str
    step : int
        Step size of first match-value scan through corpus. Can be thought of
        as a sort of "scan resolution". Should not exceed length of query.
    flex : int
        Max. left/right substring position adjustment value. Should not
        exceed length of query / 2.

    Outputs
    -------
    output0 : str
        Best matching substring.
    output1 : float
        Match ratio of best matching substring. 1 is perfect match.
    """

    def _match(a, b):
        """Compact alias for SequenceMatcher."""
        return SequenceMatcher(None, a, b).ratio()

    def scan_corpus(step: int):
        """Return list of match values from corpus-wide scan."""
        match_values = []

        m = 0
        while m + qlen - step <= len(corpus):
            match_values.append(_match(query, corpus[m : m-1+qlen]))
            if verbose:
                print(query, "-", corpus[m: m + qlen], _match(query, corpus[m: m + qlen]))
            m += step

        return match_values

    def index_max(v):
        """Return index of max value."""
        return max(range(len(v)), key=v.__getitem__)

    def adjust_left_right_positions():
        """Return left/right positions for best string match."""
        # bp_* is synonym for 'Best Position Left/Right' and are adjusted 
        # to optimize bmv_*
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + qlen] * 2

        # bmv_* are declared here in case they are untouched in optimization
        bmv_l = match_values[p_l // step]
        bmv_r = match_values[p_l // step]

        for f in range(flex):
            ll = _match(query, corpus[p_l - f: p_r])
            if ll > bmv_l:
                bmv_l = ll
                bp_l = p_l - f

            lr = _match(query, corpus[p_l + f: p_r])
            if lr > bmv_l:
                bmv_l = lr
                bp_l = p_l + f

            rl = _match(query, corpus[p_l: p_r - f])
            if rl > bmv_r:
                bmv_r = rl
                bp_r = p_r - f

            rr = _match(query, corpus[p_l: p_r + f])
            if rr > bmv_r:
                bmv_r = rr
                bp_r = p_r + f

            if verbose:
                print("\n" + str(f))
                print("ll: -- value: %f -- snippet: %s" % (ll, corpus[p_l - f: p_r]))
                print("lr: -- value: %f -- snippet: %s" % (lr, corpus[p_l + f: p_r]))
                print("rl: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r - f]))
                print("rr: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r + f]))

        return bp_l, bp_r, _match(query, corpus[bp_l : bp_r])

    if not case_sensitive:
        query = query.lower()
        corpus = corpus.lower()

    qlen = len(query)

    if flex >= qlen/2:
        print("Warning: flex exceeds length of query / 2. Setting to default.")
        flex = 3

    match_values = scan_corpus(step)
    pos = index_max(match_values) * step

    pos_left, pos_right, match_value = adjust_left_right_positions()

    return corpus[pos_left: pos_right].strip(), match_value


def perform_back_translation(json_file: str) -> pd.DataFrame:

    """
    Iterate through existing database to append new data using traduction 
    to English and then back to french on relevant paragraphs.
    Create a JSON file

    Params
    ------
    json_file: JSONL file name path with existing data.

    Outputs
    """

    data = pd.read_json(json_file, lines=True) # JSONL to pandas

    data_frame_augmented = pd.DataFrame(columns=['id','text','label','Comments'])

    # Traductor package
    gs = goslate.Goslate()

    for index, row in data.iterrows():
        if len(row.label) > 1:
            try:
                first_character_pos = row.label[0][0] # Get the position of the first character of the first label annotated
                last_character_pos = row.label[-1][1] # Same for last
                paragraphs_stops = np.append(find_occurrences(row.text, "\n"), np.array(len(row.text) - 1)) # Get the position of the paragraph delimiter
                paragraph_start = paragraphs_stops[paragraphs_stops < first_character_pos].max() 
                paragraph_end = paragraphs_stops[paragraphs_stops > last_character_pos].min() # Identify closest paragraph delimiters including all the labels (start if not, end if not)

                list_of_labels = [(row.text[label[0]:label[1]], label[2]) for label in row.label]
                list_of_labels = [(re.sub(r"/(?=\S+)", " / ", label[0]), label[1]) for label in list_of_labels]
                list_of_labels = [(label[0].replace("per", "par"), label[1]) for label in list_of_labels]
                list_of_labels = [(re.sub(r"J(?=\d+)", "j", label[0]), label[1]) for label in list_of_labels]
                list_of_labels = [(label[0].lower(), label[1]) for label in list_of_labels]
                

                # Traduction
                fr_text_to_traduce = row.text[paragraph_start:paragraph_end]
                en_text = ts.google(fr_text_to_traduce, 'fr', 'en')
                fr_text_traduced = ts.google(en_text, 'en', 'fr')
                fr_text_traduced = fr_text_traduced.replace('gris', 'gray')

                new_text = row.text[:paragraph_start] + fr_text_traduced + row.text[paragraph_end:] # New text
                new_text = new_text.lower()
                new_text_labels = [[new_text.find(label[0]), new_text.find(label[0]) + len(label[0]), label[1], label[0]] for label in list_of_labels]
                # Next line needed because traduction change or small default of traduction could be relabeled with time, so we use best match (similarity)
                new_text_labels = [[label[0], label[1], label[2]] if label[0] != - 1 else [new_text.find(get_best_match(label[3], new_text)[0]), new_text.find(get_best_match(label[3], new_text)[0]) + len(get_best_match(label[3], new_text)[0]), label[2]] for label in new_text_labels]
                data_frame_augmented.loc[len(data_frame_augmented.index)] = [f"da-bt-{index}", new_text, new_text_labels, []]
            except:
                pass
        else:
            pass
    
    return data_frame_augmented
