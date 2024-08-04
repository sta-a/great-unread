# %%
from cluster.combinations import MxCombinations
from .labels_pipeline import LabelPredictCont, LabelPredict
from utils import DataHandler
import os

class SparsmxEval(DataHandler):
    def __init__(self, language, by_author=False, attr='author'):
        super().__init__(language, output_dir='sparsmx_eval', by_author=by_author)
        self.attr = attr
        self.mxdir = os.path.join(self.data_dir, 'similarity', self.language, 'sparsification')
        self.nr_mxs = 58
        self.nr_spars = 7 if self.by_author else 9

    def load_mxnames(self):
        mxs = [filename for filename in os.listdir(self.mxdir) if filename.startswith('sparsmx')]
        if self.by_author:
            noedges_sparsmethods = ['authormin', 'authormax'] # these distance matrices have no edges if author-based
            mxs = [filename for filename in mxs if all(substr not in filename for substr in noedges_sparsmethods)]
        mxs = sorted(mxs)
        assert len(mxs) == (self.nr_mxs * self.nr_spars)
        return mxs
    
    def iterate_mxs(self):
        mxs = self.load_mxnames()
        mxcounter = 0
        for mxname in mxs[:1]:
            mxcounter += 1
            mxpath = os.path.join(self.mxdir, mxname)
            lp = LabelPredict(language=self.language, by_author=self.by_author, attr=self.attr, simmx_path=mxpath)
            df, X, y = lp.load_data()
            print(df.shape)
            same_attr = df.loc[df['attr_left'] == df['attr_right']]
            diff_attr = df.loc[df['attr_left'] != df['attr_right']]
            print(same_attr.shape)
            print(same_attr)
            tp = same_attr.loc[same_attr['weight'] != 0]
            fn = same_attr.loc[same_attr['weight'] == 0]
            fp = diff_attr.loc[same_attr['weight'] != 0]
            tn = diff_attr.loc[same_attr['weight'] == 0]
            

    # def load_mxs(self): # for non-sparsified mxs
    #     mc = MxCombinations(self.language, by_author=self.by_author)
    #     mxcounter = 0
    #     for mx in mc.load_mxs():
    #         self.mx = mx
    #         self.mxname = mx.name
    #         print('mxname', self.mxname)
    #         print('mx dimensions', self.mx.mx.shape)
    #         mxcounter += 1

    #     print('mxcounter', mxcounter)

# %%
