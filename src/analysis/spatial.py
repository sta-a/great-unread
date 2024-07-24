# %%
'''
Activate env networkclone
'''
import pandas as pd
import os
import sys
sys.path.append("..")
import networkx as nx
from copy import deepcopy
import numpy as np
import pandas as pd
from pysal.explore import esda
from pysal.lib import weights
from utils import DataHandler
from cluster.network import NXNetwork
from cluster.combinations import InfoHandler


class Moran(DataHandler):
    def __init__(self, language, output_dir, by_author=False):
        super().__init__(language=language, output_dir=output_dir, data_type='csv')
        self.by_author = by_author
        self.ih = InfoHandler(language=language, add_color=False, cmode=None, by_author=by_author)
        self.mxdir = os.path.join(self.ih.output_dir, 'sparsification')
        print(self.mxdir)
        self.ndist = 58
        if self.by_author:
            self.nspars = 7
        else:
            self.nspars = 9


    def load_mxnames(self):
        mxs = [filename for filename in os.listdir(self.mxdir) if filename.startswith('sparsmx')]
        if self.by_author:
            noedges_sparsmethods = ['authormin', 'authormax'] # these distance matrices have no edges if author-based
            mxs = [filename for filename in mxs if all(substr not in filename for substr in noedges_sparsmethods)]
        mxs = sorted(mxs)
        assert len(mxs) == (self.ndist * self.nspars)
        return mxs


    def iterate_mxs(self):
        results = []
        df = deepcopy(self.ih.metadf)
        mxs = self.load_mxnames()
        for mxname in mxs:
            network = NXNetwork(self.language, path=os.path.join(self.mxdir, mxname))

            # Create adjacency matrix as spatial weights
            adj_matrix = nx.to_numpy_array(network.graph, nodelist=df.index.tolist())
            w = weights.util.full2W(adj_matrix)

           # Calculate Moran's I statistic for 'year'
            moran_year = esda.moran.Moran(df['year'], w)

            # Calculate Moran's I statistic for 'canon'
            moran_canon = esda.moran.Moran(df['canon'], w)

            # Append results to the list
            results.append([
                mxname.split('.')[0],
                moran_year.I, moran_year.p_norm,
                moran_canon.I, moran_canon.p_norm
            ])

            print(f"Moran's I for 'year': {moran_year.I}")
            print(f"P-value for 'year': {moran_year.p_norm}")
            print(f"Moran's I for 'canon': {moran_canon.I}")
            print(f"P-value for 'canon': {moran_canon.p_norm}")

        # Convert results to DataFrame with one row per mxname
        results_df = pd.DataFrame(results, columns=[
            'mxname',
            'moran_I_year', 'moran_pval_year',
            'moran_I_canon', 'moran_pval_canon'
        ])



        results = pd.DataFrame(results, columns=[
            'mxname',
            'moran_I_year', 'moran_pval_year',
            'moran_I_canon', 'moran_pval_canon'
        ])        # Save DataFrame to CSV file with header and without index
        self.save_data(file_name='moran', data=results)


m  = Moran('eng', 'moran', True)
m.iterate_mxs()
# %%
