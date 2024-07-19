# %%


import pandas as pd

# Assuming your CSV file is named 'data.csv'
file_path = '/media/annina/MyBook/back-to-computer-240615/data_author/analysis_s2v/eng/MxSingleViz2D3DHzAnalysisSelect/results.txt'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path, header=0, index_col=None, sep=',')


# Filter rows where 'label' column is NaN
df_nan = df[df['label'].isna()]

# Filter rows where 'label' column is not NaN
df_not_nan = df[~df['label'].isna()]

# Print the results
print('df original', df.shape)
print("DataFrame with NaN in 'label' column:")
print(df_nan.shape)
print("\nDataFrame without NaN in 'label' column:")
print(df_not_nan.shape)


# Drop rows with identical values in mxname, dim, and label columns
df = df.drop_duplicates(subset=['mxname', 'curr_attr', 'dim', 'label'])
print(df_not_nan)

# %%
