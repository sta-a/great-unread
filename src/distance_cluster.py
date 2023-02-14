import hpo_functions
from sklearn_extra.cluster import KMedoids
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from sklearn.manifold import MDS
import pandas as pd
#from matplotlib import pyplot as plt
import sys
from matplotlib import pyplot as plt
import matplotlib as mpl
import os


class Clustering():
    def __init__(
            self, 
            draw,
            language, 
            dist_name,
            mx,
            distances_dir = None,
            sentiscores_dir = None,
            metadata_dir = None,
            canonscores_dir = None,
            features_dir = None):

        self.draw = draw
        self.language = language
        self.dist_name = dist_name
        self.mx = mx
        self.distances_dir = distances_dir
        self.sentiscores_dir = sentiscores_dir
        self.metadata_dir = metadata_dir
        self.canonscores_dir = canonscores_dir
        self.features_dir = features_dir
        self.group_params ={
            'unread': {'n_clusters': 2, 'type': 'kmedoids'},
            'gender': {'n_clusters': 2, 'type': 'kmedoids'},
            'author': {'type': 'hierarchical'}}


    def get_clusters(self):
        for group, param_dict in self.group_params.items():

            vis = None
            if self.draw == True:
                vis = ClusterVis(
                    language=self.language,
                    dist_name = self.dist_name,
                    mx = self.mx,
                    group=group,
                    distances_dir=self.distances_dir,
                    sentiscores_dir=self.sentiscores_dir,
                    metadata_dir=self.metadata_dir,
                    canonscores_dir=self.canonscores_dir,
                    features_dir=self.features_dir)

            if param_dict['type'] == 'hierarchical':
                clustering = self.hierarchical(vis)
            elif param_dict['type'] == 'kmedoids':
                clustering = self.kmedoids(group, vis)
        return clustering

    def hierarchical(self, vis):
        # from Pydelta
        # Ward

        # Linkage matrix
        clustering = sch.ward(ssd.squareform(self.mx, force="tovector"))
        vis.draw_dendrogram(clustering)
        return clustering
        
    def kmedoids(self, group, vis):
        n_clusters = self.group_params[group]['n_clusters']
        kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', method='pam', init='build', random_state=8)
        clustering = kmedoids.fit_predict(self.mx)
        clustering = pd.DataFrame(clustering, index=self.mx.index).rename({0: 'cluster'}, axis=1)
        vis.draw_mds(clustering)
        return clustering


class ClusterVis():
    def __init__(
            self, 
            language, 
            dist_name,
            mx,
            group, 
            distances_dir,
            sentiscores_dir,
            metadata_dir,
            canonscores_dir,
            features_dir):
        self.language = language
        self.dist_name = dist_name
        self.mx = mx
        self.group = group
        self.distances_dir = distances_dir
        self.sentiscores_dir = sentiscores_dir
        self.metadata_dir = metadata_dir
        self.canonscores_dir = canonscores_dir
        self.features_dir = features_dir
        self.file_group_mapping = self._init_colormap()
        self.plot_name = f'{self.dist_name}_{self.group}_{self.language}'

    def _relabel_axis(self):
        labels = self.ax.get_ymajorticklabels()
        for label in labels:
            color = self.file_group_mapping.loc[self.file_group_mapping['file_name'] ==label.get_text(), 'group_color']
            label = label.set_color(str(color.values[0]))

    def save(self,plt, vis_type, dpi):
        plt.savefig(os.path.join(self.distances_dir, f'{self.plot_name}_{vis_type}.png'), dpi=dpi)

    def draw_dendrogram(self, clustering):
        plt.clf()
        plt.figure(figsize=(12,12),dpi=1000)
        dendro_data = sch.dendrogram(
            Z=clustering, 
            orientation='left', 
            labels=self.mx.index.to_list(),
            show_leaf_counts=True,
            leaf_font_size=1)
        self.ax = plt.gca() 
        self._relabel_axis()
        plt.title = self.plot_name
        #plt.xlabel('Samples')
        #plt.ylabel('Euclidean distances')
        self.save(plt, 'hierarchical-dendrogram', 1000)

    def draw_mds(self, clustering):
        df = MDS(n_components=2, dissimilarity='precomputed', random_state=6, metric=True).fit_transform(self.mx)
        df = pd.DataFrame(df, columns=['comp1', 'comp2'], index=self.mx.index)
        df = df.merge(self.file_group_mapping, how='inner', left_index=True, right_on='file_name', validate='one_to_one')
        df = df.merge(clustering, how='inner', left_on='file_name', right_index=True, validate='1:1')

        def _group_cluster_color(row):
            color = None
            if row['group_color'] == 'b' and row['cluster'] == 0:
                color = 'darkblue'
            elif row['group_color'] == 'b' and row['cluster'] == 1:
                color = 'royalblue'
            elif row['group_color'] == 'r' and row['cluster'] == 0:
                color = 'crimson'
            #elif row['group_color'] == 'r' and row['cluster'] == 0:
            else:
                color = 'deeppink'
            return color

        df['group_cluster_color'] = df.apply(_group_cluster_color, axis=1)


        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)
        plt.scatter(df['comp1'], df['comp2'], color=df['group_cluster_color'], s=2, label="MDS")
        plt.title = self.plot_name
        self.save(plt, 'kmedoids-MDS', dpi=500)

    def _init_colormap(self):
        if self.group == 'author':
            x = hpo_functions.get_author_groups(self.mx)
            file_group_mapping = pd.DataFrame(x).reset_index().rename({'index': 'file_name'}, axis=1)
            groups = file_group_mapping['author'].unique()
            props = mpl.rcParams['axes.prop_cycle']

            colormap = {x: y['color'] for x,y in zip(groups, props())}
            colormap = pd.DataFrame(colormap, index=['group_color']).T.reset_index().rename({'index': 'author', 'color': 'group_color'}, axis=1)
            file_group_mapping = file_group_mapping.merge(colormap, how='left', on='author', validate='many_to_one')

        elif self.group == 'unread':
            X, file_group_mapping = hpo_functions.get_data(
                language=self.language, 
                task='regression-importance', 
                label_type='canon', 
                features='book', 
                features_dir=self.features_dir, 
                canonscores_dir=self.canonscores_dir, 
                sentiscores_dir=self.sentiscores_dir, 
                metadata_dir=self.metadata_dir)
            threshold = 0.5
            file_group_mapping = file_group_mapping.reset_index().rename({'index': 'file_name'}, axis=1)
            file_group_mapping['group_color'] = file_group_mapping['y'].apply(lambda x: 'r' if x > threshold else 'b')

        elif self.group == 'gender':
            # Combine author metadata and file_name
            authors = pd.read_csv(os.path.join(self.metadata_dir, 'authors.csv'), header=0, sep=';')[['author_viaf','name', 'first_name', 'gender']]
            metadata = pd.read_csv(os.path.join(self.metadata_dir, f'{self.language.upper()}_texts_meta.csv'), header=0, sep=';')[['author_viaf', 'file_name']]
            file_group_mapping = metadata.merge(authors, how='left', on='author_viaf', validate='many_to_one')
            file_group_mapping['file_name'] = file_group_mapping['file_name'].replace(to_replace={ ############################3
                # new -- old
                'Storm_Theodor_Immensee_1850': 'Storm_Theodor_Immersee_1850',
                'Hoffmansthal_Hugo_Reitergeschichte_1899': 'Hoffmansthal_Hugo-von_Reitergeschichte_1899'
                })
            # check if all file names have metadata
            #ytest = df.merge(self.mx, left_on='file_name', right_index=True, validate='one_to_one', how='outer')
            file_group_mapping['group_color'] = file_group_mapping['gender'].apply(lambda x: 'r' if x=='f' else 'b')
            
        return file_group_mapping