from encodings import utf_8
import langcodes
import spacy
import pandas as pd
import nltk
import re
import json

from nltk.corpus import wordnet
from nltk import tokenize
from nltk.tokenize import word_tokenize
from torch import threshold

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')



#en_ner_bc5cdr_md


file_name = "/Users/sarahmayer/Documents/DSB_Y2/Quinten/NER_Posology_Extraction/src/all.jsonl"
lang='fra'
th = 0.9

def to_jsonl(data: pd.DataFrame, new_jsonl: str):

    """
    Pandas DataFrame to JSON file
    """

    with open(new_jsonl, 'w', encoding='utf-8') as file:
        data.to_json(file, force_ascii=False, orient='records', lines=True)


def get_wordvector_similarity(nlp,replacements):
    """
    From the list of synonyms obtained from Wordnet, apply the 
    similarity score to filter out non-relevant synonyms. The word pair who has similarity score less than 
    THRESHOLD is neglected.
    """
    replacements_refined = {}
    THRESHOLD = th
    for key, values in replacements.items():
        key_vec = nlp(key.lower())
        synset_refined = []
        for each_value in values:
            value_vec = nlp(each_value.lower())
            if (len(value_vec)>0):
                if key_vec.similarity(value_vec) > THRESHOLD:
                    synset_refined.append(each_value)
        if len(synset_refined) > 0:
            replacements_refined[key] = synset_refined
    return replacements_refined

nlp = spacy.load('fr_core_news_md')

# Load data set
print("Reading input file...")
dataset_df = pd.read_json(file_name, lines=True)
dataset_df.head()
phrases = dataset_df['text']
print("Number of phrases in input file:", len(phrases))

# Generate paraphrases
print("Generating paraphrases...")
augmented_data = {}
for i in range(len(phrases)):
    tokenized_phrase = tokenize.sent_tokenize(phrases[i], language='french')
    for current_sentence in tokenized_phrase:
        # print("\tCurrent input sentence:",current_sentence)
        doc = nlp(current_sentence)
        replacements = {}
        for token in doc: 
            for j in range(len(dataset_df["label"].iloc[i])):
                if token.idx not in [dataset_df["label"].iloc[i][j][0],dataset_df["label"].iloc[i][j][1]]:
                    if ('NOUN' in token.tag_):
                        if (token.ent_type == 0): # if its a noun and not a NER
                            """Augment the noun with possible synonyms from Wordnet"""            
                            syns = wordnet.synsets(token.text,'n', lang=lang)
                            synonyms = set()
                            for eachSynSet in syns:
                                for eachLemma in eachSynSet.lemmas(lang):
                                    current_word = eachLemma.name()
                                    if current_word.lower() != token.text.lower() and current_word != token.lemma_:
                                        synonyms.add(current_word.replace("_"," "))
                            synonyms = list(synonyms)
                            #print("\tCurrent noun word:", token.text, "(",len(synonyms),")")
                            if len(synonyms) > 0:
                                replacements[token.text] = synonyms
                    if 'ADJ' in token.tag_: # if its an adjective
                        """Augment the adjective with possible synonyms from Wordnet"""
                        syns = wordnet.synsets(token.text,'a', lang=lang)
                        synonyms = set()
                        for eachSynSet in syns:
                            for eachLemma in eachSynSet.lemmas(lang):
                                current_word = eachLemma.name()
                                if current_word.lower() != token.text.lower() and current_word != token.lemma_:
                                    synonyms.add(current_word.replace("_"," "))
                        synonyms = list(synonyms)
                        #print("\tCurrent adjective word:", token.text, "(",len(synonyms),")")
                        if len(synonyms) > 0:
                            replacements[token.text] = synonyms
                    if 'VERB' in token.tag_: # if its a verb
                        """Augment the verb with possible synonyms from Wordnet"""
                        syns = wordnet.synsets(token.text,'v', lang=lang)
                        synonyms = set()
                        for eachSynSet in syns:
                            for eachLemma in eachSynSet.lemmas(lang):
                                current_word = eachLemma.name()
                                if current_word.lower() != token.text.lower() and current_word != token.lemma_:
                                    synonyms.add(current_word.replace("_"," "))
                        synonyms = list(synonyms)
                        #print("\tCurrent verb word:", token.text, "(",len(synonyms),")")
                        if len(synonyms) > 0:
                            replacements[token.text] = synonyms
        #print("Input(before filtering):\n",sum(map(len, replacements.values())))
        replacements_refined = get_wordvector_similarity(nlp,replacements)
        #print("Output(after filtering based on similarity score):\n",sum(map(len, replacements_refined.values())))
        #print ("\tReplacements:", replacements_refined)
        generated_sentences = []
        generated_sentences.append(current_sentence)
        for key, value in replacements_refined.items():
            replaced_sentences = []
            for each_value in value:
                for each_sentence in generated_sentences:
                    new_sentence = re.sub(r"\b%s\b" % key,each_value,each_sentence)
                    replaced_sentences.append(new_sentence)
            generated_sentences.extend(replaced_sentences)
        augmented_data[current_sentence] = generated_sentences  

print("#####################--Paraphrase generation completed--#####################")
#print("Total variations created:", sum(map(len, augmented_data.values())))
# print("Each set is shown below:")
# for key in augmented_data.keys():
#     print("Seed sentence:-", key)
#     print("Augmented sentence:-", augmented_data[key],"\n")

print("Saving to disk as CSV...")
# Save to disk as csv
augmented_dataset = {'Phrases':[],'Paraphrases':[]}
phrases = []
paraphrases = []
for key,value in augmented_data.items():
    for each_value in value:
        phrases.append(key)
        paraphrases.append(each_value)
augmented_dataset['Phrases'] = phrases
augmented_dataset['Paraphrases'] = paraphrases
augmented_dataset_df = pd.DataFrame.from_dict(augmented_dataset)

data_paraphrase = dataset_df.copy()
for i in range(len(phrases)):
    tokenized_phrase = tokenize.sent_tokenize(phrases[i], language='french')
    for current_sentence in tokenized_phrase:
        cnt = 0
        for i in range(0, len(augmented_dataset_df)):
            if augmented_dataset_df["Phrases"][i].find(str(current_sentence)) !=-1:
                cnt = i
                break
            if str(augmented_dataset_df["Phrases"].iloc[cnt]) != str(augmented_dataset_df["Paraphrases"].iloc[cnt]):
                phrase_new  = phrases[i].replace(str(current_sentence), str(augmented_dataset_df["Paraphrases"].iloc[cnt]))
                data_paraphrase[i] = phrase_new
        

to_jsonl(data_paraphrase, "new_json_file")