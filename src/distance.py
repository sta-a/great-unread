# %%
%load_ext autoreload
%autoreload 2
import pandas as pd
import os
from scipy.spatial.distance import minkowski
from sklearn_extra.cluster import KMedoids
from matplotlib import pyplot as plt
import sys
from matplotlib import pyplot as plt
sys.path.insert(1, '/home/annina/scripts/pydelta')
import delta
from distances_functions import ImprtDistance, PydeltaDist, is_symmetric, show_distance_distribution, get_importances_mx, get_pydelta_mx
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from hpo_functions import get_author_groups, get_data
import matplotlib as mpl

nr_texts = None
languages = ['ger']
data_dir = '../data'
nmfw = 500


# %%


# %%
# kmedoids = KMedoids(n_clusters=2, metric='precomputed', method='pam', init='build', random_state=8).fit(burrowsmx)
from sklearn.manifold import MDS
X_transform = MDS(n_components=2, dissimilarity='precomputed', random_state=8).fit_transform(burrowsmx)


# distance, dissimilarity'
# Std of whole corpus or only mfw????
# function registry
#similarity or dissimiliarity
# use all distances

# agglomerative hierarchical clustering, k-means, or density-based clustering (DBSCAN)


# %%
class Clustering():
    def __init__(self, mx):
        self.mx = mx

    def get_clusters(self, type):
        if type == 'hierarchical':
            clustering = self.hierarchical()
        return clustering

    def hierarchical(self):
        # from Pydelta
        # Ward

        # Linkage matrix
        clustering = sch.ward(ssd.squareform(self.mx, force="tovector"))
        return clustering

    def draw_dendrogram(self, Z):
        d = Dendrogram(labels=self.mx.index.to_list(), Z=Z)

    def map_groups_colors(self, type='author'):
        if type == 'author':
            author_groups = get_author_groups(self.mx)


class Dendrogram():
    def __init__(self, mx, Z, group):
        self.mx = mx
        self.Z = Z
        self.group = group
        self.file_group_mapping = self._init_colormap()


        plt.clf()
        plt.figure(figsize=(12,12),dpi=1000)
        self.dendro_data = sch.dendrogram(
            Z=self.Z, 
            orientation='left', 
            labels=self.mx.index.to_list(),
            show_leaf_counts=True,
            leaf_font_size=1)
        self.ax = plt.gca() 
        self._relabel_axis()
        plt.title(f'Burrows Delta {nmfw} {language}')
        #plt.xlabel('Samples')
        #plt.ylabel('Euclidean distances')
        plt.savefig(os.path.join(data_dir, 'distances', language, f'{self.group}.png'))
        plt.show()



    def _init_colormap(self):
        if self.group == 'author':
            x = get_author_groups(self.mx)
            file_group_mapping = pd.DataFrame(x).reset_index().rename({'index': 'file_name'}, axis=1)
            groups = file_group_mapping['author'].unique()
            props = mpl.rcParams['axes.prop_cycle']
            colormap = {x: y['color'] for x,y in zip(groups, props())}
            colormap = pd.DataFrame(colormap, index=['color']).T.reset_index().rename({'index': 'author'}, axis=1)
            file_group_mapping = file_group_mapping.merge(colormap, how='left', on='author', validate='many_to_one')

        elif self.group == 'unread':
            X, file_group_mapping = get_data(
                language=language, 
                task='regression-importance', 
                label_type='canon', 
                features='book', 
                features_dir=features_dir, 
                canonscores_dir=canonscores_dir, 
                sentiscores_dir=sentiscores_dir, 
                metadata_dir=metadata_dir)
            threshold = 0.5
            file_group_mapping = file_group_mapping.reset_index().rename({'index': 'file_name'}, axis=1)
            print(file_group_mapping)
            file_group_mapping['color'] = file_group_mapping['y'].apply(lambda x: 'r' if x > threshold else 'b')

        elif self.group == 'gender':
            # Combine author metadata and file_name
            authors = pd.read_csv(os.path.join(metadata_dir, 'authors.csv'), header=0, sep=';')[['author_viaf','name', 'first_name', 'gender']]
            metadata = pd.read_csv(os.path.join(metadata_dir, f'{language.upper()}_texts_meta.csv'), header=0, sep=';')[['author_viaf', 'file_name']]
            file_group_mapping = metadata.merge(authors, how='left', on='author_viaf', validate='many_to_one')
            file_group_mapping['file_name'] = file_group_mapping['file_name'].replace(to_replace={ ############################3
                # new -- old
                'Storm_Theodor_Immensee_1850': 'Storm_Theodor_Immersee_1850',
                'Hoffmansthal_Hugo_Reitergeschichte_1899': 'Hoffmansthal_Hugo-von_Reitergeschichte_1899'
                })
            # check if all file names have metadata
            #ytest = df.merge(self.mx, left_on='file_name', right_index=True, validate='one_to_one', how='outer')
            file_group_mapping['color'] = file_group_mapping['gender'].apply(lambda x: 'r' if x=='f' else 'b')
            print(file_group_mapping)
            
        return file_group_mapping

    def _relabel_axis(self):
        labels = self.ax.get_ymajorticklabels()
        for label in labels:
            color = self.file_group_mapping.loc[self.file_group_mapping['file_name'] ==label.get_text(), 'color']
            print('color before setting', color)
            label = label.set_color(str(color.values[0]))

for language in languages:
    distances_dir = os.path.join(data_dir, 'distances', language)
    if not os.path.exists(distances_dir):
        os.makedirs(distances_dir, exist_ok=True)
    sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
    metadata_dir = os.path.join(data_dir, 'metadata', language)
    canonscores_dir = os.path.join(data_dir, 'canonscores')
    features_dir = os.path.join(data_dir, 'features_None', language)


    # imprtmx = get_importances_mx(language, data_dir)
    burrowsmx = get_pydelta_mx(language, data_dir, nmfw, nr_texts)



    c = Clustering(mx=burrowsmx)
    clusters = c.get_clusters(type='hierarchical')
    # dendro = Dendrogram(mx=burrowsmx, Z=clusters, group='unread')
    # dendro = Dendrogram(mx=burrowsmx, Z=clusters, group='author')
    dendro = Dendrogram(mx=burrowsmx, Z=clusters, group='gender')



# %%
x = burrowsmx.index.to_frame(index=True).rename({0:'file_name'}, axis=1)
canon_file = '210907_regression_predict_02_setp3_FINAL.csv'
canon_scores = pd.read_csv(os.path.join(canonscores_dir, canon_file), header=0, sep=';')

y = x.merge(canon_scores, how='left', on='file_name')




# %%
X, file_group_mapping = get_data(
    language=language, 
    task='regression-importance', 
    label_type='canon', 
    features='book', 
    features_dir=features_dir, 
    canonscores_dir=canonscores_dir, 
    sentiscores_dir=sentiscores_dir, 
    metadata_dir=metadata_dir)
threshold = 0.5
print('after get data', X.shape, file_group_mapping.shape)
# %%
X.merge(y, how='outer', left_index=True, right_on='file_name')
# %%
# Hoffmansthal_Hugo-von_Ein-Brief_1902 # canon scores
# Hoffmansthal_Hugo_Ein-Brief_1902 # raw docs

# Hegelers_Wilhelm_Mutter-Bertha_1893 # canon scores
# Hegeler_Wilhelm_Mutter-Bertha_1893 # raw docs
# %%
authors = pd.read_csv(os.path.join(metadata_dir, 'authors.csv'), header=0, sep=';')[['author_viaf','name', 'first_name', 'gender']]
metadata = pd.read_csv(os.path.join(metadata_dir, f'{language.upper()}_texts_meta.csv'), header=0, sep=';')[['author_viaf', 'file_name']]
df = metadata.merge(authors, how='left', on='author_viaf', validate='many_to_one')
df['file_name'] = df['file_name'].replace(to_replace={ ############################3
    # new -- old
    'Storm_Theodor_Immensee_1850': 'Storm_Theodor_Immersee_1850',
    'Hoffmansthal_Hugo_Reitergeschichte_1899': 'Hoffmansthal_Hugo-von_Reitergeschichte_1899'
    })
# check if all file names have metadata
ytest = df.merge(burrowsmx, on='file_name', validate='one_to_one', how='outer')
# %%
