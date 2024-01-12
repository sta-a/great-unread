# %%
import numpy as np
np.random.seed(42)
import random
random.seed(42)
import pandas as pd
import os
import shutil


def create_simmx(size=5):
    np.random.seed(42)
    random.seed(42)

    # Create a symmetric matrix with random values between 0 and 1
    lower_triangle = np.tril(np.random.rand(size, size), k=-1)
    upper_triangle = lower_triangle.T
    symmetric_matrix = lower_triangle + upper_triangle

    # Set diagonal elements to 1
    np.fill_diagonal(symmetric_matrix, 0)  ######### no self-loops


    # Create a Pandas DataFrame
    nr_digits = len(str(size))
    columns = [f'col{str(i).zfill(nr_digits)}' for i in range(0, size)]
    index = columns

    df = pd.DataFrame(data=symmetric_matrix, columns=columns, index=index)
    
    # Set percentage of entries to 0
    zero_percent = 0
    i1 = random.choices(range(size), k=int(size*size * (zero_percent / 100)))
    i2 = random.choices(range(size), k=int(size*size * (zero_percent / 100)))
    for i1,i2 in zip(i1,i2):
        if i1!=i2:
            df.iloc[i1,i2] = 0
            df.iloc[i2,i1] = 0

    print('symmetric: ', df.equals(df.T))
    # Display the DataFrame
    return df.round(3)


def get_simmx_mini(n=5):
    path = '/home/annina/scripts/great_unread_nlp/data/similarity/eng/simmxs/burrows-500.csv'
    df = pd.read_csv(path)
    df = df.set_index("file_name")
    df = df.iloc[:n, :n]
    print(df.shape)
    return df


def remove_directories(directory_paths):
    for path in directory_paths:
        if os.path.exists(path):  # Check if the directory exists
            try:
                shutil.rmtree(path)
                print(f"Directory '{path}' removed successfully.")
            except OSError as e:
                print(f"Error removing directory '{path}': {e}")
        else:
            print(f"Directory '{path}' does not exist. Skipping removal.")


def delete_png_files(directory_paths):
    for path in directory_paths:
        # Get a list of all files in the directory
        if os.path.exists(path):
            file_list = os.listdir(path)

            # Iterate through the files and delete those with a '.png' extension
            for filename in file_list:
                if filename.endswith('.png'):
                    file_path = os.path.join(path, filename)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

