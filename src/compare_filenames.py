# %%

import pandas as pd
import os

data_dir = '../data'
for language in ['eng', 'ger']:
    print('\n----------------------------------\n', language, '\n----------------------------------')
    distances_dir = os.path.join(data_dir, 'distances', language)
    if not os.path.exists(distances_dir):
        os.makedirs(distances_dir, exist_ok=True)
    sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
    metadata_dir = os.path.join(data_dir, 'metadata', language)
    canonscores_dir = os.path.join(data_dir, 'canonscores')
    features_dir = os.path.join(data_dir, 'features_None', language)

    # Distance files
    dist_path = os.path.join(distances_dir, 'distances_burrows500.csv')
    distdf = pd.read_csv(dist_path, index_col=0)
    print(f'Distance matrix has {distdf.shape} entries.')


    # author gender
    authorsdf = pd.read_csv(os.path.join(metadata_dir, 'authors.csv'), header=0, sep=';')[['author_viaf','name', 'first_name', 'gender']]
    print(f'Authors has {authorsdf.shape} entries.')

    metadata = pd.read_csv(os.path.join(metadata_dir, f'{language.upper()}_texts_meta.csv'), header=0, sep=';')[['author_viaf', 'file_name']]
    print(f'Metadata has {distdf.shape} entries.')
    file_group_mapping = metadata.merge(authorsdf, how='left', on='author_viaf', validate='many_to_one')
    file_group_mapping['file_name'] = file_group_mapping['file_name'].replace(to_replace={ ############################3
        # new -- old
        'Storm_Theodor_Immensee_1850': 'Storm_Theodor_Immersee_1850',
        'Hoffmansthal_Hugo_Reitergeschichte_1899': 'Hoffmansthal_Hugo-von_Reitergeschichte_1899'
        })
    file_group_mapping['gender'] = file_group_mapping['gender'].replace({'w': 'f'})
    file_group_mapping['gender'] = file_group_mapping['gender'].fillna('anonymous')

    print(file_group_mapping['gender'].unique())


    df = distdf.merge(file_group_mapping, left_index=True, right_on='file_name', validate='one_to_one')
    #file_group_mapping.to_csv(os.path.join(distances_dir, 'author_gender_filenamescorrected.csv'), sep=',', index=False)
# %%
# Hoffmansthal_Hugo-von_Ein-Brief_1902 # canon scores
# Hoffmansthal_Hugo_Ein-Brief_1902 # raw docs

# Hegelers_Wilhelm_Mutter-Bertha_1893 # canon scores
# Hegeler_Wilhelm_Mutter-Bertha_1893 # raw docs
