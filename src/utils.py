import os
from ast import literal_eval
from pathlib import Path
import numpy as np
import pandas as pd
from unidecode import unidecode

german_special_chars = {'Ä':'Ae', 'Ö':'Oe', 'Ü':'Ue', 'ä':'ae', 'ö':'oe', 'ü':'ue', 'ß':'ss'}


def get_doc_paths(docs_dir, lang):
    doc_paths = [os.path.join(docs_dir, lang, doc_name) for doc_name in os.listdir(os.path.join(docs_dir, lang)) if doc_name[-4:] == ".txt"]
    return doc_paths


def load_list_of_lines(path, line_type):
    if line_type == "str":
        with open(path, "r") as reader:
            lines = [line.strip() for line in reader]
    elif line_type == "np":
        lines = list(np.load(path)["arr_0"])
    else:
        raise Exception(f"Not a valid line_type {line_type}")
    return lines


def save_list_of_lines(lst, path, line_type):
    os.makedirs(str(Path(path).parent), exist_ok=True)
    if line_type == "str":
        with open(path, "w") as writer:
            for item in lst:
                writer.write(str(item) + "\n")
    elif line_type == "np":
        np.savez_compressed(path, np.array(lst))
    else:
        raise Exception(f"Not a valid line_type {line_type}")


def read_labels(labels_dir):
    labels_df = pd.read_csv(os.path.join(labels_dir, "210907_regression_predict_02_setp3_FINAL.csv"), sep=";")[["file_name", "m3"]]
    print(type(labels_df))
    labels = dict(labels_df.values)
    return labels


def read_sentiment_scores(sentiment_dir, canonization_labels_dir):
    canonization_scores = pd.read_csv(canonization_labels_dir + "210907_regression_predict_02_setp3_FINAL.csv", sep=';', header=0)[["id", "file_name"]]
    scores = pd.read_csv(sentiment_dir + "ENG_reviews_senti.csv", sep=";", header=0)[["text_id", "sentiscore_average"]]
    scores = scores.merge(right=canonization_scores, how="left", right_on="id", left_on="text_id", validate="many_to_one")
    scores = scores.rename(columns={"sentiscore_average": "y", "file_name": "book_name"})[["book_name", "y"]]
    # scores = dict(scores[["file_name", "sentiscore_average"]].values)
    return scores

def read_new_sentiment_scores(sentiment_dir, canonization_labels_dir):
    canonization_scores = pd.read_csv(canonization_labels_dir + "210907_regression_predict_02_setp3_FINAL.csv", sep=';', header=0)[["id", "file_name"]]
    scores = pd.read_csv(sentiment_dir + "ENG_reviews_senti_classified.csv", sep=";", header=0)[["text_id", "sentiscore_average", "classified"]]
    scores = scores.merge(right=canonization_scores, how="left", right_on="id", left_on="text_id", validate="many_to_one")
    scores = scores.rename(columns={"sentiscore_average": "y", "classified": "c", "file_name": "book_name"})[["book_name", "y", "c"]]
    # scores = dict(scores[["file_name", "sentiscore_average"]].values)
    return scores



def read_extreme_cases(labels_dir):
    extreme_cases_df = pd.read_csv(os.path.join(labels_dir, "210907_classified_data_02_m3_step3_FINAL.csv"), sep=";")[["file_name"]]

    # for key, value in file_name_mapper.items():
    #     extreme_cases_df.loc[extreme_cases_df["file_name"] == key, "file_name"] = value

    # extreme_cases_df = extreme_cases_df[~extreme_cases_df["file_name"].isin(extra_file_names)]
    return extreme_cases_df


def unidecode_custom(text):
    for char, replacement in german_special_chars.items():
        text = text.replace(char, replacement)
    text = unidecode(text)
    return text
