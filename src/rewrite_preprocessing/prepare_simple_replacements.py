
# %%
import re
import pandas as pd
import os

separator = 'ƒ' # special separator that is not contained in to_replace strings
output_file = '/home/annina/scripts/great_unread_nlp/data/preprocess/replacement_values.csv'


# Use raw text to write \n 
more_lines = r"""
    Kingsley_Charles_Westward-Ho_1855ƒ[Greek text]ƒ''ƒ0
    Edgeworth_Maria_Helen_1834ƒ[Footnoteƒ''ƒ0
    Blackmore_R-D_Lorna-Doone_1869ƒ[Greek word]ƒ''ƒ0
    Forrester_Andrew_The-Female-Detective_1864ƒfrequently. utteredƒfrequently utteredƒ4
    Forrester_Andrew_The-Female-Detective_1864ƒwhat. suchƒwhat suchƒ5
    Forrester_Andrew_The-Female-Detective_1864ƒWnoƒWhoƒ6
    Forrester_Andrew_The-Female-Detective_1864ƒgir1ƒgirlƒ7
    Galbraith_Lettice_In-the-Seance-Room_1893ƒ\Vhatƒ'What'ƒ0
    Galbraith_Lettice_The-Case-of-Lady-Lukestan_1893ƒedu~ationƒ'education'ƒ0
    Lamb_Charles_Hamlet_1807ƒvdthƒ'with'ƒ0
    Lamb_Charles_Hamlet_1807ƒseeino^ƒ'seeing'ƒ1
    Lamb_Charles_Hamlet_1807ƒw^oundƒwoundƒ2
    Lohenstein_Daniel_Arminius_1689ƒ \nƒ'ön'ƒ0
    Lohenstein_Daniel_Arminius_1689ƒ \ƒ'ö'ƒ1
    Lohenstein_Daniel_Arminius_1689ƒ /ƒ\nƒ2
    Lohenstein_Daniel_Arminius_1689ƒ[ƒ''ƒ3
    Shelley_Mary_Mathilda_1820ƒ[ƒ''ƒ0
    Shelley_Mary_Mathilda_1820ƒ]ƒ''ƒ1
    Radcliffe_Ann_The-Romance-of-the-Forest_1791ƒt\orpidityƒtorpidityƒ0
    Radcliffe_Ann_The-Romance-of-the-Forest_1791ƒ[ƒ''ƒ1
    Radcliffe_Ann_The-Romance-of-the-Forest_1791ƒ]ƒ''ƒ2
    Wells_H-G_The-First-Men-in-the-Moon_1901ƒ* Footnoteƒ''ƒ0
    Wells_H-G_The-First-Men-in-the-Moon_1901ƒ[ƒ''ƒ1
    Wells_H-G_The-First-Men-in-the-Moon_1901ƒ]ƒ''ƒ2
    Edgeworth_Maria_The-Grateful-Negro_1804ƒ{Empty page}ƒ''ƒ0
    Edgeworth_Maria_The-Modern-Griselda_1804ƒi*ƒiƒ0
    Amory_Thomas_The-Life-of-John-Buncle_1756ƒ628.ƒ'628,'ƒ0
    """
# ƒƒ''ƒ0

# Split the string into lines and remove leading/trailing whitespace
more_lines = [line.strip() for line in more_lines.strip().split('\n')]


# Combine into one file
file_list = [file for file in os.listdir() if file.endswith('.txt')]
# Combine the content of all .txt files into a new file
with open(output_file, 'w') as f:
    f.write(f'file_name{separator}to_replace{separator}replace_with{separator}priority\n')
    for line in more_lines:
        f.write(line + '\n')
    for file_name in file_list:
        with open(file_name, 'r') as input_file:
            f.write(input_file.read())


# Check df
df = pd.read_csv(output_file, sep=separator, header=0, engine='python').fillna("''")
print(df.shape)
dup_rows = df.duplicated().any()
if dup_rows:
    print('Duplicated rows: \n', df[df.duplicated()])

# Check if to_replace values are duplicated for a text
duplicated = []
grouped = df.groupby('file_name')
for file_name, group_df in grouped:
    if group_df['to_replace'].duplicated().any():
        print(group_df)
        duplicated.append(file_name)
if duplicated:
    print(f'Duplicated to_replace values: {duplicated}.')


# Check if priority numbers are unique
filtered_df = df[df['priority'].str.isdigit()]
print(filtered_df.shape)
grouped = filtered_df.groupby('file_name')
duplicated = []
for file_name, group_df in grouped:
    if group_df['priority'].duplicated().any():
        print(group_df)
        duplicated.append(file_name)
if duplicated:
    print('Duplicated priority values: {duplicated}.')


df = df.sort_values(by=['file_name', 'priority'])
df.to_csv(output_file, index=False, sep=separator)




# %%
# Check if there are old filenames in the preprocessing files
import pandas as pd
import os
import sys
sys.path.append("..")
from utils import search_string_in_files

# Define the paths to the directories
dir1 = '/home/annina/scripts/great_unread_nlp/data/corpus_corrections'
dir2 = '/home/annina/scripts/great_unread_nlp/data/preprocess'

for lang in ['eng', 'ger']:
    # Define the filenames
    filename1 = f'compare_filenames_{lang}.csv'
    filename2 = 'replacement_values.csv'

    # Construct full paths to the files
    tok_path = os.path.join(dir1, filename1)
    chunk_path = os.path.join(dir2, filename2)

    # Load the CSV files as Pandas DataFrames
    df1 = pd.read_csv(tok_path)
    df2 = pd.read_csv(chunk_path, sep='ƒ', header=0, index_col=None, engine='python').fillna("''")


    # Get the set of entries in the "file_name" column of df2
    file_name_set = set(df2['file_name'])

    # Check if any entries in the "metadata-fn" column of df1 are in the file_name_set
    matching_entries = df1[df1['metadata-fn'].isin(file_name_set)]


    if not matching_entries.empty:
        print("Entries found in metadata-fn that are also in file_name:")
        print(matching_entries)
    else:
        print("No matching entries found.")


    # Check if old filesnams are in preprocessing script
    directory_path = '/home/annina/scripts/great_unread_nlp/src/'
    search_string = ''
    extensions = ['.py']
    for value in df1['metadata-fn']:
        search_string_in_files(directory_path, value, extensions)
# %%
