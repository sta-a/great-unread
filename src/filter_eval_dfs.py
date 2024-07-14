# %%
# Keep only combinations for one set of parameters
# Creating a new file is more efficient than filtering during every experiment
import pandas as pd
import os


def get_filepaths(data_dir, language, level):
    old_filename = f'{level}_results.csv'
    new_filename = f'{level}_results_paramcomb.csv'

    df_path = f'/media/annina/MyBook/back-to-computer-240615/{data_dir}/s2v/{language}/mxeval/'


    old_filepath = os.path.join(df_path, old_filename)
    new_filepath = os.path.join(df_path, new_filename)

    print(old_filepath, '\n', new_filepath)
    return old_filepath, new_filepath


# Rename original data
for data_dir in ['data', 'data_author']:
    for language in ['eng', 'ger']:
        for level in ['cat', 'cont']:
            old_filepath, new_filepath = get_filepaths(data_dir, language, level)

            try:
                os.rename(old_filepath, new_filepath)
                print(f"File renamed successfully from {old_filepath} to {new_filepath}")
            except FileNotFoundError:
                print(f"Error: The file {old_filepath} was not found in the specified directory.")
            except PermissionError:
                print("Error: You don't have permission to rename this file.")
            except OSError as e:
                print(f"Error occurred while renaming the file: {e}")



# Filter
for data_dir in ['data', 'data_author']:
    for language in ['eng', 'ger']:
        for level in ['cat', 'cont']:
            old_filepath, new_filepath = get_filepaths(data_dir, language, level)
            df = pd.read_csv(new_filepath, header=0, index_col=False)
            original_nrows = df.shape[0]
            param_string = 'dimensions-16_walklength-30_numwalks-200_windowsize-15_untillayer-5'
            df = df[df['mxname'].str.contains(param_string, na=False)]
            filtered_nrows = df.shape[0]
            print(original_nrows, filtered_nrows)
            df.to_csv(old_filepath, index=False, header=True)



    # def get_bestparams_mode_params(self):
    #     # Return few parameter combinations for creating the embeddings for the actual data
    #     params = {
    #         'dimensions': [16],
    #         'walk-length': [30],
    #         'num-walks': [200],
    #         'window-size': [15],
    #         'until-layer': [5],
    #         'OPT1': ['True'],
    #         'OPT2': ['True'],
    #         'OPT3': ['True']
    #     }
    #     return params
# %%
