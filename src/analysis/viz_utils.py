import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import numpy as np
from copy import deepcopy
from typing import List
import pandas as pd
import matplotlib.lines as mlines
import textwrap
from copy import deepcopy
import random
random.seed(9)
import textwrap
from typing import List

import sys
sys.path.append("..")
from utils import DataHandler
from cluster.cluster_utils import Colors
import logging
logging.basicConfig(level=logging.DEBUG)

import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw




class VizBase(DataHandler):
    def __init__(self, language, output_dir='analysis', cmode='nk', info=None, plttitle=None, exp=None, by_author=False):
        super().__init__(language, output_dir=output_dir, data_type='png')
        self.cmode = cmode
        self.info = info
        self.plttitle = plttitle
        self.exp = exp
        self.by_author = by_author
        self.fontsize = 12
        self.add_custom_subdir()

        self.special_cols = ['cluster', 'clst_shape', 'gender_cluster', 'author_cluster', 'x', 'y', 'pos']     


        self.key_attrs = ['gender', 'canon', 'year', 'canon-ascat', 'year-ascat']
        if self.by_author:
            self.key_attrs = self.key_attrs + ['canon-min', 'canon-max']
        else: 
            self.key_attrs = ['author'] + self.key_attrs
            
        self.cat_attrs = ['gender', 'canon-ascat', 'year-ascat']
        if not self.by_author:
            self.cat_attrs = ['author'] + self.cat_attrs

        self.is_cat = False
        if (self.info is not None) and (self.info.attr in self.cat_attrs):
            self.is_cat = True

        self.needs_cbar = self.check_cbar()


    def add_custom_subdir(self):
        self.add_subdir(f"{self.cmode}_{self.exp['name']}")


    def get_metadf(self):
        self.df = deepcopy(self.info.metadf)


    def check_cbar(self):
        # Check if any continuous attributes are shown.
        # If yes, a cbar is necessary.
        cbar = False
        if not self.is_cat:
            cbar = True
        return cbar


    def add_text(self, ax, x=0, y=0, width=30):  
        ax.text(x=x, y=y, s=textwrap.fill(self.plttitle, width), fontsize=self.fontsize)


    def save_plot(self, plt, plt_kwargs={'dpi': 100}):
        self.save_data(data=plt, data_type=self.data_type, file_name=None, file_path=self.vizpath, plt_kwargs=plt_kwargs)


    def get_path(self, name='viz', omit: List[str]=[], data_type=None):
        if data_type is None:
            data_type = self.data_type
        file_name = f'{name}-{self.info.as_string(omit=omit)}.{data_type}'
        return self.get_file_path(file_name, subdir=True)
    

    def add_cbar(self, ax):
        # Create a color bar from a color map
        # The color map is not used in any matplotlib functions (like for a heatmap), therefore the bar has to be created manually.
        # Create a ScalarMappable with the color map
        cmap = Colors.CMAP
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Create a color bar using the ScalarMappable
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, location='left')
        cbar.ax.tick_params(axis='both', which='major', labelsize=self.fontsize)

        # Set a label for the color bar
        # cbar.set_label('Color Bar Label', rotation=270, labelpad=15)

        # Remove ticks from the color bar
        # cbar.set_ticks([])

        # Adjust the layout to make room for the color bar
        # fig_cbar.tight_layout()        


    def add_legend(self, fig_or_ax, attr, label='size', use_shapes=False, loc='upper right', boxx=1.05, boxy=1, boxwidth=0.2, boxheight=0.4, fontsize=None, markersize=8):
        bbox_to_anchor = (boxx, boxy, boxwidth, boxheight)
        if fontsize is None:
            fontsize = self.fontsize

        mapping = {}
        self.df[attr] = self.df[attr].astype(str) # Enforce that gender column is treated as string
        for unique_attr in self.df[attr].unique().tolist():
            cdf = self.df[self.df[attr] == unique_attr]
            attribute = cdf.iloc[0][f'{attr}_color']
            if use_shapes:
                shape = cdf.iloc[0]['clst_shape']
            else:
                shape = 'o'

            if len(cdf) > 1:
                mapping[unique_attr] = (attribute, shape, len(cdf))

        # Keep the 10 most frequent elements
        mapping = dict(sorted(mapping.items(), key=lambda item: item[1][2], reverse=True))
        mapping = {k: v for k, v in list(mapping.items())[:10]}


        # Create legend patches
        legend_patches = []
        for unique_attr, (attribute, shape, count) in mapping.items():
            if label == 'size':
                clabel = f'{count}'
            elif label == 'attr':
                # Underscores cannot be used in labels because they increase the space between the labels in the legend
                # This causes the two legends to not be aligned
                
                # If attr is author name, get only the first letter of the first name to keep legend short
                if '_' in unique_attr:
                    name_parts = unique_attr.split("_")
                    clabel = f'{name_parts[0]}{name_parts[1][0]} ({count})'
                else:
                    clabel = f'{unique_attr} ({count})'
                
            legend_patches.append(mlines.Line2D([], [], color=attribute, marker=shape, label=clabel, linestyle='None', markersize=markersize))

        fig_or_ax.legend(handles=legend_patches, labelspacing=0.5, loc=loc, bbox_to_anchor=bbox_to_anchor, fontsize=fontsize)


    def get_ax(self, ix):
        return self.axs[ix[0], ix[1]]


    def add_subtitles(self, attrix, clstix, shapeix, combix=None):
        self.get_ax(attrix).set_title(f'Attribute: {self.info.attr}', fontsize=self.fontsize)
        self.get_ax(clstix).set_title('Clusters', fontsize=self.fontsize)
        self.get_ax(shapeix).set_title('Attribute (color) and clusters (shapes)', fontsize=self.fontsize)
        if combix is not None:
            self.get_ax(combix).set_title('Attribute and clusters (combined)', fontsize=self.fontsize)



class ImageGrid(DataHandler):
    def __init__(self, language, attr=None, by_author=False, output_dir='analysis', imgdir='nk_singleimage', select_with_gui=False, rowmajor=True, imgs_as_paths=False, subdir=False):
        super().__init__(language, output_dir=output_dir, data_type='png', subdir=subdir)
        self.attr = attr
        self.by_author = by_author
        self.imgdir = os.path.join(self.output_dir, imgdir)
        self.select_with_gui = select_with_gui
        self.rowmajor = rowmajor # Order in which images are filled into the grid
        self.imgs_as_paths = imgs_as_paths # True if imgs contains full paths, False if it contains only image names
        self.key_attrs = ['author', 'canon', 'gender', 'year'] ###########
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
        print('vizpath', self.vizpath)

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
                    if self.rowmajor:
                        index = i * self.ncol + j
                    else:
                        index = j * self.nrow + i
                    if index < len(imgs):
                        # print('Loading image:', imgs[index])
                        if not self.imgs_as_paths:
                            img = plt.imread(os.path.join(self.imgdir, imgs[index]))
                        else:
                            img = plt.imread(imgs[index])
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

            if self.select_with_gui:
                # Add matplotlib plot to tkinter window
                canvas = FigureCanvasTkAgg(self.fig, master=root)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

                # Set select_with_gui to True for selecting networks with mouse click
                # Run the tkinter event loop
                tk.mainloop()
            plt.clf()
            self.fig.clf()
            plt.close(self.fig)
            plt.close('all')
        plt.close()
        plt.close('all')



class ClusterAuthorGrid(ImageGrid):
    '''
    Show cluster assignments with highest internal evaluation score and authors next to each other.
    '''

    def __init__(self, language, cmode, exp, te, by_author=False, output_dir='analysis', subdir=None):
        self.cmode = cmode
        self.exp = exp
        self.te = te
        if by_author == True:
            print('Not implemented for by_author=True')
        # Check if file is big enough
        self.ntop = exp['ntop']
        assert  self.ntop == 30
        if len(self.te.df) >= self.ntop:
            self.isbig = True
        else:
            self.isbig = False

        super().__init__(language=language, by_author=False, output_dir=output_dir, imgs_as_paths=True) # load_single_images is called in ImageGrid.__init__
        self.subdir = subdir
        print('subdir after init', self.subdir)
        self.nrow = 6
        self.ncol = 10


    def load_single_images(self):
        self.cluster_dir = os.path.join(self.output_dir, f'{self.cmode}_singleimage_cluster')
        self.author_dir = os.path.join(self.output_dir, f'{self.cmode}_singleimage')

        # Check if df is sorted
        assert self.te.df[self.exp['evalcol']].is_monotonic_decreasing
        imgs = []
        counter = 0
        for row in self.te.df.itertuples(index=False):
            if counter < self.ntop:
                comb, attr = row.file_info.rsplit('_', 1)
                mxname, rest = row.file_info.split('_', 1)
                imgs.append(os.path.join(self.cluster_dir, f'{comb}.{self.data_type}'))
                imgs.append(os.path.join(self.author_dir, f'{mxname}_author.{self.data_type}'))
                counter += 1
        
        if self.isbig:
            assert len(imgs) == 2*self.ntop
        return imgs
    

    def get_title(self, imgpath):
        return os.path.splitext(os.path.basename(imgpath))[0]

