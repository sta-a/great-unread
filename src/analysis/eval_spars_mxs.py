# %%
from cluster.combinations import MxCombinations
from .labels_pipeline import LabelPredictCont, LabelPredict
from utils import DataHandler
import os
import pandas as pd

class SparsmxEval(DataHandler):
    def __init__(self, language, by_author=False, attr='author'):
        super().__init__(language, output_dir='label_predict', by_author=by_author, data_type='csv')
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
    
    def calculate_metrics(self, tp_count, fn_count, fp_count, tn_count):
        # Calculate precision, recall, accuracy, and F1 score
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) != 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) != 0 else 0
        accuracy = (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count) if (tp_count + tn_count + fp_count + fn_count) != 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        return [precision, recall, accuracy, f1_score]
    
    def get_measures_for_all_mxs(self):
        mxs = self.load_mxnames()
        mxcounter = 0
        results = []
        for mxname in mxs[:3]:####################### 
            print('\n', mxname)
            mxcounter += 1
            mxpath = os.path.join(self.mxdir, mxname)
            lp = LabelPredict(language=self.language, by_author=self.by_author, attr=self.attr, simmx_path=mxpath)
            df, X, y = lp.load_data()
            same_attr = df.loc[df['attr_left'] == df['attr_right']]
            diff_attr = df.loc[df['attr_left'] != df['attr_right']]

            tpdf = same_attr.loc[same_attr['weight'] != 0]
            fndf = same_attr.loc[same_attr['weight'] == 0]
            fpdf = diff_attr.loc[diff_attr['weight'] != 0]
            tndf = diff_attr.loc[diff_attr['weight'] == 0]

            tp_count = len(tpdf)
            fn_count = len(fndf)
            fp_count = len(fpdf)
            tn_count = len(tndf)

            confusion_matrix = pd.DataFrame({
                'weight positive': [tp_count, fp_count],
                'weight 0': [fn_count, tn_count]
            }, index=['Same Attr', 'Diff Attr'])

            # Display the confusion matrix
            print(confusion_matrix)


            metrics = self.calculate_metrics(tp_count, fn_count, fp_count, tn_count)
            mxname = mxname.replace('.pkl', '')
            metrics.insert(0, mxname)
            metrics.extend([tp_count, fn_count, fp_count, tn_count])
            results.append(metrics)

        results = pd.DataFrame(results, columns=['mxname', 'precision', 'recall', 'accuracy', 'f1_score', 'tp_count', 'fn_count', 'fp_count', 'tn_count'])
        results = results.round(3)

        self.save_data(data=results, file_name=f'evaluation_metrics_{self.attr}')

                    

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
