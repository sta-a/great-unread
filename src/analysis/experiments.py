import sys
sys.path.append("..")

from utils import DataHandler
from .mxviz import MxViz
from .nkviz import NkViz
from .topeval import TopEval
from cluster.network import NXNetwork
from cluster.combinations import CombinationsBase

import logging
logging.basicConfig(level=logging.DEBUG)



class ExpBase(DataHandler):
    def __init__(self, language, cmode):
        super().__init__(language, output_dir='analysis')
        self.cmode = cmode
        # self.mh = MetadataHandler(self.language)
        # self.metadf = self.mh.get_metadata(add_color=True)


    def run_experiments(self):
        # Default values
        maxsize = round(0.9*self.nr_texts)

        cat_evalcol = 'ARI'
        cont_evalcol = 'logreg_acc'

        if self.cmode == 'mx':
            evalcol = 'silhouette_score'
        else:
            evalcol = 'modularity'


        topkcat = {'maxsize': maxsize, 'evalcol': cat_evalcol, 'special': True, 'dfs': ['cat']}
        topkcont = {'maxsize': maxsize, 'evalcol': cont_evalcol, 'special': True, 'dfs': ['cont']}

        topcanon = {'maxsize': maxsize, 'evalcol': cont_evalcol, 'attr': ['canon'], 'special': False, 'dfs': ['cont']}

        # Internal evaluation criterion
        interesting_attrs = ['author', 'gender', 'canon', 'year']
        intfull = {'maxsize': self.nr_texts, 'attr': interesting_attrs, 'evalcol': evalcol, 'special': True, 'dfs': ['cat', 'cont']}
        intmax = {'maxsize': maxsize, 'attr': interesting_attrs, 'evalcol': evalcol, 'special': True, 'dfs': ['cat', 'cont']}

        exps = {'intfull': intfull, 'intmax': intmax, 'topkcat': topkcat, 'topkcont': topkcont, 'topcanon': topcanon}
        exps = {'topcanon': topcanon}


        for expname, expd in exps.items():
            te = TopEval(self.language, self.cmode, expname, expd)
            self.run_experiment(expname, expd, te)


class MxExp(ExpBase):
    def __init__(self, language):
        super().__init__(language, 'mx')


    def run_experiment(self, expname, expd, te):
        # topkpath = 'mxtopk.pkl'
        # if os.path.exists(topkpath):
        #     print(topkpath)
        #     with open(topkpath, 'rb') as file:
        #         topk_comb = pickle.load(file)
        #         print('loaded topk')
        # else:
        #     topk_comb = list(self.get_top_combinations(expd))
        #     with open(topkpath, 'wb') as file:
        #         pickle.dump(topk_comb, file)
        #         print('created topk')


        for topk in te.get_top_combinations():
        # for topk in topk_comb:
            info, plttitle = topk
            print(info.as_string())
            if expd['special'] == True:
                info.add('special', 'canon')
                print('added canon')

            # Get matrix
            cb = CombinationsBase(self.language, add_color=False, cmode='mx')
            mx = [mx for mx in cb.mxs if mx.name == info.mxname]
            assert len(mx) == 1
            mx = mx[0]
            info.add('order', 'olo')
            viz = MxViz(self.language, mx, info, plttitle=plttitle, expname=expname)
            viz.visualize()

        self.add_subdir(f'{self.cmode}{expname}')
        print(self.subdir)
        te.save_dfs(path=self.subdir)



class NkExp(ExpBase):
    def __init__(self, language):
        super().__init__(language, 'nk')


    def run_experiment(self, expname, expd, te):
        # topkpath = 'nktopk.pkl' #############################
        # if os.path.exists(topkpath):
        #     with open(topkpath, 'rb') as file:
        #         topk_comb = pickle.load(file)
        # else:
        #     topk_comb = list(self.get_top_combinations(expd))
        #     with open(topkpath, 'wb') as file:
        #         pickle.dump(topk_comb, file)

        for topk in te.get_top_combinations():
        # for topk in topk_comb:
            info, plttitle = topk
            print(info.as_string())
            if 'special' in expd:
                info.add('special', 'canon')
            network = NXNetwork(self.language, path=info.spmx_path)
            viz = NkViz(self.language, network, info, plttitle=plttitle, expname=expname)          
            viz.visualize()

        self.add_subdir(f'{self.cmode}{expname}')
        te.save_dfs(path=self.subdir)