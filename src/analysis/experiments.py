import sys
sys.path.append("..")
from copy import deepcopy

from utils import DataHandler
from .mxviz import MxViz
from .nkviz import NkViz
from .topeval import TopEval
from cluster.network import NXNetwork
from cluster.combinations import CombinationsBase

# import logging
# logging.basicConfig(level=logging.DEBUG)



class ExpBase(DataHandler):
    def __init__(self, language, cmode):
        super().__init__(language, output_dir='analysis')
        self.cmode = cmode
        # self.mh = MetadataHandler(self.language)
        # self.metadf = self.mh.get_metadata(add_color=True)


    def get_experiments(self):
        # Default values
        maxsize = 0.9
        embmxs = ['both', 'full']
        cat_evalcol = 'ARI'
        cont_evalcol = 'logreg_acc'
        if self.cmode == 'mx':
            evalcol = 'silhouette_score'
        else:
            evalcol = 'modularity'


        # Overall best performance
        topdicts = [
            {'name': 'topcat', 'maxsize': maxsize, 'evalcol': cat_evalcol, 'special': True, 'dfs': ['cat']},
            {'name': 'topcont', 'maxsize': maxsize, 'evalcol': cont_evalcol, 'special': True, 'dfs': ['cont'], 'ntop': self.nr_texts}, #########################3ntop
            {'name': 'topcont_bal', 'maxsize': maxsize, 'evalcol': 'logreg_acc_balanced', 'special': True, 'dfs': ['cont']}
        ]

        # Visualize canon
        canondicts = [
            {'name': 'topcanon', 'maxsize': maxsize, 'evalcol': cont_evalcol, 'attr': ['canon'], 'special': False, 'dfs': ['cont']},
            {'name': 'topcanon_bal', 'maxsize': maxsize, 'evalcol': 'logreg_acc_balanced', 'attr': ['canon'], 'special': False, 'dfs': ['cont']}
        ]

        # Get best performance of embedding distances
        topdicts_emb = []
        for cdict in topdicts + canondicts:
            d = deepcopy(cdict)
            d['mxname'] = embmxs
            d['name'] = d['name'] + '_emb'
            topdicts_emb.append(d)


        # Internal evaluation criterion
        interesting_attrs = ['author', 'gender', 'canon', 'year']
        intdicts = [
            {'name': 'intfull', 'maxsize': self.nr_texts, 'attr': interesting_attrs, 'evalcol': evalcol, 'special': True, 'dfs': ['cat', 'cont']},
            {'name': 'intmax', 'maxsize': maxsize, 'attr': interesting_attrs, 'evalcol': evalcol, 'special': True, 'dfs': ['cat', 'cont']}
        ]

        # Check if clustering is constant over multiple parameter combinations
        compclust = [{'name': 'compclust', 'maxsize': maxsize, 'dfs': ['cat'], 'attr': ['author'], 'clst_alg_params': 'alpa', 'ntop': self.nr_texts}]
        if self.cmode == 'mx':
            compclust[0]['clst_alg_params'] = 'hierarchical-nclust-5-method-average'

        exps = topdicts + topdicts_emb + intdicts + canondicts + compclust
        exps = [topdicts[1]] ##########################
        return exps


    def run_experiments(self, ntop=1):
        exps = self.get_experiments()
        for exp in exps:
            expname = exp['name']
            if 'ntop' not in exp:
                exp['ntop'] = ntop

            self.add_subdir(f'{self.cmode}{expname}')
            te = TopEval(self.language, self.cmode, exp, expdir=self.subdir)

            # if expname == 'compclust':
            #     self.compare_clusters(exp, te)
            # else:
            #     self.run_experiment(exp, te)
            self.run_experiment(exp, te)


    def compare_clusters(self, exp, te):
        df = te.load_data()



class MxExp(ExpBase):
    def __init__(self, language):
        super().__init__(language, 'mx')


    def run_experiment(self, exp, te):
        expname = exp['name']

        for topk in te.get_top_combinations():
            info, plttitle = topk
            print(info.as_string())
            if exp['special'] == True:
                info.add('special', 'canon')

            # Get matrix
            cb = CombinationsBase(self.language, add_color=False, cmode='mx')
            mx = [mx for mx in cb.mxs if mx.name == info.mxname]
            assert len(mx) == 1
            mx = mx[0]
            info.add('order', 'olo')
            viz = MxViz(self.language, mx, info, plttitle=plttitle, expname=expname)
            viz.visualize()



class NkExp(ExpBase):
    def __init__(self, language):
        super().__init__(language, 'nk')


    def run_experiment(self, exp, te):
        expname = exp['name']
        self.add_subdir(f'{self.cmode}{expname}')
        te = TopEval(self.language, self.cmode, exp, expdir=self.subdir)

        for topk in te.get_top_combinations():
            info, plttitle = topk
            print(info.as_string())
            if 'special' in exp:
                info.add('special', 'canon')
            network = NXNetwork(self.language, path=info.spmx_path)
            viz = NkViz(self.language, network, info, plttitle=plttitle, expname=expname)          
            viz.visualize()
