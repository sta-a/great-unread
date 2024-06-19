import os
import random
random.seed(9)


import sys
sys.path.append("..")
from utils import DataHandler
from .analysis_utils import NoedgesLoader
from .viz_utils import ImageGrid
import logging
logging.basicConfig(level=logging.DEBUG)




class NkNetworkGrid(ImageGrid):
    '''
    Plot every network for an attribute. Select interesting networks by mouse click.
    '''
    def __init__(self, language, attr=None, by_author=False, select_with_gui=True):
        super().__init__(language=language, attr=attr, by_author=by_author, select_with_gui=select_with_gui)
        self.by_author = by_author
        self.nrow = 2
        self.ncol = 5
        self.name_index = 0
        self.nfields = 9
        self.add_subdir(f'nkselect_{self.attr}')


    def visualize(self, vizname='viz'):
        # self.create_overview_file()
        fndict = self.create_filenames_list()

        for figname, imglist in fndict.items():
            print(figname)
            vizname = self.get_filename(figname)


            nfields = self.nrow * self.ncol # fields per plot
            nplots = len(imglist)
            nfig = nplots // nfields 
            if nplots % nfields != 0:
                nfig += 1

            ix = 0
            for i in range(nfig):
                # If ix + nfields > len(cols), no error is raised because Python allows out-of-bound slicing
                imgs = imglist[ix: ix + (nfields)]
                ix += nfields


                if nfig == 1:
                    vizname = vizname
                else:
                    vizname = f'{vizname}{i}'
                super().visualize(vizname=vizname, imgs=imgs)



    def create_filenames_list(self):
        # Create a list of lists that contain either all matrices with the same mxname or with the same sparsification technique
        fndict = {}

        for im in self.imgs:
            name = im.split('_')[self.name_index]
            
            if name not in fndict:
                fndict[name] = []
            fndict[name].append(im)

        for name in fndict:
            fndict[name].sort()

        return fndict


    def get_filename(self, figname):
        return figname


class SparsGrid(NkNetworkGrid):
    '''
    Plot every network per sparsification technique. Attribute "canon" is highlighted.
    '''
    def __init__(self, language, attr=None, by_author=False, select_with_gui=True):
        super().__init__(language, attr, by_author, select_with_gui=select_with_gui)
        self.nrow = 6 # 58 mxs per figure
        self.ncol = 11
        self.name_index = 1
        self.nfields = 58
        self.fontsize = 6

        self.add_subdir(f'nkselect_sparsgrid_{self.attr}')

    
    def get_filename(self, figname):
        return f"gridviz_{figname}"
    


class Selector(DataHandler):
    '''
    Write names of interesting networks to file
    Interesting networks are networks where canonized texts seem to be non-randomly distributed, or which have an interesting structure, or which are not dependent on the year
    '''
    def __init__(self, language, by_author=False):
        super().__init__(language, output_dir='analysis', data_type='png')
        self.by_author = by_author
        self.imgdir = os.path.join(self.output_dir, 'nk_singleimage')
        self.nr_mxs = 58
        if self.by_author:
            self.nr_spars = 7
        else:
            self.nr_spars = 9


    def get_noedges_combs(self):
        n = NoedgesLoader()
        self.noedges = n.get_noedges_list()
    

    def get_mx_and_spars_names(self):
        mxnames = set()
        sparsnames = set()

        for filename in os.listdir(self.imgdir):
            if filename.endswith('.png'):
                # Split the filename by underscores to extract mxname and sparsname
                parts = filename.split('_')
                assert len(parts)== 3
                mxnames.add(parts[0])
                sparsnames.add(parts[1])

        mxnames = list(mxnames)
        sparsnames = list(sparsnames)
        assert len(mxnames) == self.nr_mxs
        assert len(sparsnames) == self.nr_spars
        return mxnames, sparsnames
            

    def remove_png(self, l):
        l = [x.replace('.png', '') for x in l]
        return l
    

    def read_names_from_file(self, attr, by_author=False):
        self.add_subdir(f'nkselect_{attr}')
        if by_author:
            self.subdir = self.subdir.replace('data', 'data_author')
            print(self.subdir)
        path = os.path.join(self.subdir, f'selected_{attr}.txt')
        unique_lines = []
        with open(path, 'r') as file:
            for line in file:
                line = line.strip()
                if line not in unique_lines:
                    unique_lines.append(line)
       
        unique_lines = self.remove_png(unique_lines)
        return unique_lines
    

    def remove_attr_and_duplicates(self, nklist):
        # if element has format mxname_spars_attr, remove attr
        all_mxnames, all_sparsnames = self.get_mx_and_spars_names()
        newlist = []
        for nk in nklist:
            print(nk)
            nk = nk.split('_')
            assert len(nk) == 2 or len(nk) == 3
            print(nk[0], nk[1])
            assert nk[0] in all_mxnames
            assert nk[1] in all_sparsnames
            nk = f'{nk[0]}_{nk[1]}'
            newlist.append(nk)
        return list(set(newlist))


    def get_interesting_networks(self):
        # Canon: networks were selected in class NkNetworkGrid with GUI implemented in ImageGrid
        canon = self.read_names_from_file('canon')

        # Find distances where texts are not grouped by year.
        # Text-based and author-based find approximately the same distances.
        # For eng, 'correlation' and 'sqeuclidean' are border cases -> include them
        interesting_mxnames = self.read_names_from_file('year', by_author=False) # distances where texts do not cluster according to year

        # Combine interesting distance measures and interesting sparsification methods
        interesting_mxnames = interesting_mxnames + ['full', 'both'] # embedding-based distances are also interesting
        interesting_sparsnames = ['simmel-3-10', 'simmel-5-10']

        all_mxnames, all_sparsnames = self.get_mx_and_spars_names()
        interesting_mxnames_all_spars = [elem1 + '_' + elem2 for elem1 in interesting_mxnames for elem2 in all_sparsnames]
        all_mxnames_interesting_spars = [elem1 + '_' + elem2 for elem1 in all_mxnames for elem2 in interesting_sparsnames]

        all_interesting = list(set(canon + interesting_mxnames_all_spars + all_mxnames_interesting_spars))
        all_interesting = self.remove_attr_and_duplicates(all_interesting)
        print(len(all_interesting))
        print(len(canon), len(all_mxnames_interesting_spars), len(interesting_mxnames_all_spars))

        all_interesting = [x for x in all_interesting if not x in self.noedges]
        with open(os.path.join(self.output_dir, 'interesting_networks.csv'), 'w') as f:
            f.write('\n'.join(all_interesting))