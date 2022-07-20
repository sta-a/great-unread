# %%
import pandas as pd
path = '/home/annina/scripts/great_unread_nlp/data/sentiscores/eng/ENG_reviews_senti_FINAL.csv'

# All reviewed texts
df = pd.read_csv(path, header=0, sep=';')[['file_name']]
# Nr of different texts that were reviewed
len(df.file_name.unique())
# Nr reviewed texts after contradicting labels were removed: 191
