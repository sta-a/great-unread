import pandas as pd
import itertools
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pickle
from copy import deepcopy
import os
import matplotlib.gridspec as gridspec
import time
import random
import textwrap
from typing import List
from cluster.network import NXNetwork
random.seed(9)


import sys
sys.path.append("..")
from utils import DataHandler
from .analysis_utils import VizBase, NoedgesLoader
from .nkviz import NkKeyAttrViz
from cluster.cluster_utils import CombinationInfo
from cluster.combinations import InfoHandler
import logging
logging.basicConfig(level=logging.DEBUG)


import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw


class ImageGrid(DataHandler):
    def __init__(self, language, attr=None, by_author=False, output_dir='analysis', imgdir='nk_singleimage', select_with_gui=False):
        super().__init__(language, output_dir=output_dir, data_type='png')
        self.attr = attr
        self.by_author = by_author
        self.imgdir = os.path.join(self.output_dir, imgdir)
        self.select_with_gui = select_with_gui
        self.key_attrs = ['author', 'canon', 'gender', 'year']
        self.nrow = 3
        self.ncol = 3
        self.imgs = self.load_single_images()
        self.fontsize = 6

        self.img_width = 1.8 # format of single-network plots stored on file
        self.img_height = 2.1
        self.nr_mxs = 58
        self.nr_spars = 9

        # Whitespace
        self.ws_left = 0.01
        self.ws_right = 0.99
        self.ws_bottom = 0.01
        self.ws_top = 0.99
        self.ws_wspace = 0.01
        self.ws_hspace = 0.01


    def adjust_subplots(self):
        self.fig.subplots_adjust(
            left=self.ws_left,
            right=self.ws_right,
            bottom=self.ws_bottom,
            top=self.ws_top,
            wspace=self.ws_wspace,
            hspace=self.ws_hspace)  


    def get_figure(self):
        self.fig, self.axs = plt.subplots(self.nrow, self.ncol, figsize=(self.ncol*self.img_width, self.nrow*self.img_height))
        if self.nrow == 1:
            self.axs = self.axs.reshape(1, -1) # one row, infer number of cols
        if self.ncol == 1:
            self.axs = self.axs.reshape(-1, 1)
        plt.tight_layout(pad=0)



    def load_attr_images(self, file_names):
        # Select all file names where the last part before .png seperated by an underscore is equal to attr
        file_names = [fn for fn in file_names if fn.rsplit('.', 1)[0].rsplit('_', 1)[1] == self.attr]
        return file_names


    def load_single_images(self):
        file_names = [fn for fn in os.listdir(self.imgdir) if fn.endswith('.png')]
        if self.attr is not None:
            file_names = self.load_attr_images(file_names)
        return sorted(file_names)


    def select_image(self, event, imgs):
        # Select images via mouse click, write their name to file
        path = os.path.join(self.subdir, f'selected_{self.attr}.txt')
        with open(path, 'a') as f:
            for i in range(self.nrow):
                for j in range(self.ncol):
                    index = i * self.ncol + j
                    if event.inaxes == self.axs[i, j]:
                        f.write(imgs[index] + '\n')
                        break


    def get_title(self, imgname):
        return '_'.join(imgname.split('_')[:2])

        
    def visualize(self, vizname='viz', imgs=None, **kwargs):
        self.vizpath = self.get_file_path(vizname, subdir=True, **kwargs)

        if imgs is None:
            imgs = self.imgs

        if not os.path.exists(self.vizpath):
            if self.select_with_gui:
                # Create tkinter window
                root = tk.Tk()
                root.title(vizname)

            # Display images in grid layout
            self.get_figure()
            self.adjust_subplots()
            for i in range(self.nrow):
                for j in range(self.ncol):
                    index = i * self.ncol + j
                    if index < len(imgs):
                        print('Loading image:', imgs[index])
                        img = plt.imread(os.path.join(self.imgdir, imgs[index]))
                        self.axs[i, j].imshow(img)
                        self.axs[i, j].axis('off')
                        title = self.get_title(imgs[index])
                        self.axs[i, j].set_title(title, fontsize=self.fontsize)
                        self.axs[i, j].figure.canvas.mpl_connect('button_press_event', lambda event, imgs=imgs: self.select_image(event, imgs))
                    else:
                        self.axs[i, j].clear()
                        self.axs[i, j].axis('off')
            
            # bbox_inches because titles are cut off
            self.save_data(data=plt, data_type=self.data_type, file_name=None, file_path=self.vizpath, plt_kwargs={'dpi': 300, 'bbox_inches': 'tight'})
            # plt.show()
            plt.close()

            if self.select_with_gui:
                # Add matplotlib plot to tkinter window
                canvas = FigureCanvasTkAgg(self.fig, master=root)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

                # Set select_with_gui to True for selecting networks with mouse click
                # Run the tkinter event loop
                tk.mainloop()



class NkNetworkGrid(ImageGrid):
    '''
    Plot every network for an attribute.
    '''
    def __init__(self, language, attr=None, by_author=False):
        super().__init__(language=language, attr=attr, by_author=by_author, select_with_gui=True)
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

    # def create_overview_file(self):
    #     dfpath = os.path.join(self.output_dir, 'nk_visualizations.csv')
    #     if not os.path.exists(dfpath):
    #         mxs = self.load_mxnames()
    #         mxs = [mx.replace('sparsmx-', '') for mx in mxs]
    #         mxs = [mx.split('.')[0] for mx in mxs]
    #         mxs =  [mx.split('_') for mx in mxs]
    #         mxs = sorted(mxs)

    #         df = pd.DataFrame(mxs, columns=['mxname', 'sparsification'])


    #         # Add column names to DataFrame with new empty columns
    #         for col_name in ['from_sparsmethod'] + self.key_attrs:
    #             df[col_name] = ''
    #         df.to_csv(dfpath, index=False, header=True)


    def get_filename(self, figname):
        return figname


class SparsGrid(NkNetworkGrid):
    '''
    Plot every network per sparsification technique. Attribute "canon" is highlighted.
    '''
    def __init__(self, language, attr=None, by_author=False):
        super().__init__(language, attr, by_author)
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
    def __init__(self, language):
        super().__init__(language, output_dir='analysis', data_type='png')
        self.imgdir = os.path.join(self.output_dir, 'nk_singleimage')
        self.nr_mxs = 58
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
            nk = nk.split('_')
            assert len(nk) == 2 or len(nk) == 3
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