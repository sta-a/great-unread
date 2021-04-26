import os
import pandas as pd


def get_doc_paths(docs_dir, lang):
    doc_paths = [os.path.join(docs_dir, lang, doc_name) for doc_name in os.listdir(os.path.join(docs_dir, lang)) if doc_name[-4:] == ".txt"]
    return doc_paths

def get_pickle_paths(pickles_dir, lang):
    pickle_paths = [os.path.join(pickles_dir, lang, pickle_name) for pickle_name in os.listdir(os.path.join(pickles_dir, lang)) if pickle_name[-7:] == ".pickle"]
    return pickle_paths

def read_labels(labels_dir, lang):
    labels_df = pd.read_csv(os.path.join(labels_dir, lang, "canonisation_scores.csv"), sep=";")
    if lang == "eng":
        file_name_mapper = {
            'The Wild Irish Girl': 'Owenson_Sydney_The-Wild-Irish-Girl_1806',
            'Somerville-Ross_The-Real-Charlotte_1894': "Somerville-Ross_Edith-Martin_The-Real-Charlotte_1894",
            'LeFanu_Joseph-Sheridan_Schalken-the-Painter_1851.txt': 'LeFanu_Joseph-Sheridan_Schalken-the-Painter_1851',
        }

        for key, value in file_name_mapper.items():
            labels_df["file_name"][labels_df["file_name"] == key] = value
        
        extra_file_names = [
            "Austen_Jane_Northanger-Abbey_1818",
            "Cleland_John_Fanny-Hill_1748",
            "Defoe_Daniel_Roxana_1724",
            "Fielding_Henry_Amelia_1752",
            "Kingsley_Charles_The-Water-Babies_1863",
            "Le-Queux_William_The-Invasion-of-1910_1906",
            "Surtees_Robert_Jorrocks-Jaunts-and-Jollities_1831"
        ]
        labels = dict(labels_df[~labels_df["file_name"].isin(extra_file_names)][["file_name", "percent"]].values)
    elif lang == "ger":
        file_name_mapper = {
            'Ebner-Eschenbach_Marie-von_Bozena_1876': 'Ebner-Eschenbach_Marie_Bozena_1876',
            'Ebner-Eschenbach_Marie-von_Das-Gemeindekind_1887': 'Ebner-Eschenbach_Marie_Das-Gemeindekind_1887',
            'Ebner-Eschenbach_Marie-von_Der-Kreisphysikus_1883': 'Ebner-Eschenbach_Marie_Der-Kreisphysikus_1883',
            'Ebner-Eschenbach_Marie-von_Der-Muff_1896': 'Ebner-Eschenbach_Marie_Der-Muff_1896',
            'Ebner-Eschenbach_Marie-von_Die-Freiherren-von-Gemperlein_1889': 'Ebner-Eschenbach_Marie_Die-Freiherren-von-Gemperlein_1889',
            'Ebner-Eschenbach_Marie-von_Die-Poesie-des-Unbewussten_1883': 'Ebner-Eschenbach_Marie_Die-Poesie-des-Unbewussten_1883',
            'Ebner-Eschenbach_Marie-von_Die-Resel_1883': 'Ebner-Eschenbach_Marie_Die-Resel_1883',
            'Ebner-Eschenbach_Marie-von_Ein-kleiner-Roman_1887': 'Ebner-Eschenbach_Marie_Ein-kleiner-Roman_1887',
            'Ebner-Eschenbach_Marie-von_Krambabuli_1883': 'Ebner-Eschenbach_Marie_Krambabuli_1883',
            'Ebner-Eschenbach_Marie-von_Lotti-die-Uhrmacherin_1874': 'Ebner-Eschenbach_Marie_Lotti-die-Uhrmacherin_1874',
            'Ebner-Eschenbach_Marie-von_Rittmeister-Brand_1896': 'Ebner-Eschenbach_Marie_Rittmeister-Brand_1896',
            'Ebner-Eschenbach_Marie-von_Unsuehnbar_1890': 'Ebner-Eschenbach_Marie_Unsuehnbar_1890',
            'Hunold_Christian-Friedrich_Adalie_1702': 'Hunold_Christian_Friedrich_Die-liebenswuerdige-Adalie_1681'
        }
        for key, value in file_name_mapper.items():
            labels_df["file_name"][labels_df["file_name"] == key] = value
        labels = dict(labels_df[["file_name", "percent"]].values)
    return labels