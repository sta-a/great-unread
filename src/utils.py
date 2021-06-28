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
    labels_df = pd.read_csv(os.path.join(labels_dir, "210616_regression_predict.csv"), sep=";")[["file_name", "m3"]]

    file_name_mapper = {'Blackmore_R-D_Lorna-Doone_1869': 'Blackmore_R-D_Lorna_Doone_1869',
                        'Bulwer-Lytton_Edward_Paul-Clifford_1830': 'Bulwer-Lytton_Edward_Paul-Clifford_1832',
                        'Conrad_Joseph_The-Secret-Sharer_1910': 'Conrad_Joseph_The-Secret-Sharer_1895',
                        'Parsons_Eliza_The-Castle-of-Wolfenbach_1793': 'Parsons_Eliza_The-Castle-of_Wolfenbach_1793',
                        'Richardson_Samuel_Sir-Charles-Grandison_1753': 'Richardson_Sir-Charles-Grandison_1753',
                        'Eichendorff_Joseph_Auch-ich-war-in-Arkadien_1832': 'Eichendorff_Joseph-von_Auch-ich-war-in-Arkadien_1832',
                        'Eichendorff_Joseph_Die-Gluecksritter_1841': 'Eichendorff_Joseph-von_Die-Gluecksritter_1841',
                        'Eichendorff_Joseph_Libertas-und-ihrer-Freier_1848': 'Eichendorff_Joseph-von_Libertas-und-ihrer-Freier_1848',
                        'Eichendorff_Joseph_Viel-Laermen-um-Nichts_1832': 'Eichendorff_Joseph-von_Viel-Laermen-um-Nichts_1832',
                        'Goethe_Johann-Wolfgang_Unterhaltungen-deutscher-Ausgewanderten_1795': 'Goethe_Johann-Wolfgang-von_Unterhaltungen-deutscher-Ausgewanderten_1795',
                        'Zschokke_Johann_Addrich-im-Moos_1825': 'Zschokke_Johann-Heinrich_Addrich-im-Moos_1825',
                        'Zschokke_Johann_Der-Freihof-von-Aarau_1823': 'Zschokke_Johann-Heinrich_Der-Freihof-von-Aarau_1823',
                        'Zschokke_Johann_Die-Rose-von-Disentis_1844': 'Zschokke_Johann-Heinrich_Die-Rose-von-Disentis_1844'}

    for key, value in file_name_mapper.items():
        labels_df.loc[labels_df["file_name"] == key, "file_name"] = value
        
    extra_file_names = [
         'Austen_Jane_Northanger-Abbey_1818',
         'Cleland_John_Fanny-Hill_1748',
         'Defoe_Daniel_Roxana_1724',
         'Fielding_Henry_Amelia_1752',
         'Kingsley_Charles_The-Water-Babies_1863',
         'Le-Queux_William_The-Invasion-of-1910_1906',
         'Surtees_Robert_Jorrocks-Jaunts-and-Jollities_1831'
    ]
    labels = dict(labels_df[~labels_df["file_name"].isin(extra_file_names)].values)
    return labels


def unidecode_custom(text):
    for char, replacement in german_special_chars.items():
        text = text.replace(char, replacement)
    text = unidecode(text)
    return text