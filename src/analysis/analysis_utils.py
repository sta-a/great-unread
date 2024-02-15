import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from copy import deepcopy
from typing import List

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import markers
import textwrap

import sys
sys.path.append("..")
from utils import DataHandler
from cluster.cluster_utils import Colors
import logging
logging.basicConfig(level=logging.DEBUG)


class VizBase(DataHandler):
    def __init__(self, language, cmode, info, plttitle, expname=None):
        super().__init__(language, output_dir='analysis', data_type='png')
        self.cmode = cmode
        self.info = info
        self.plttitle = plttitle
        self.expname = expname
        self.fontsize = 12
        self.add_subdir(f'{self.cmode}_{self.expname}')

        self.cat_attrs = ['gender', 'author']
        self.is_cat = False
        if self.info.attr in self.cat_attrs:
            self.is_cat = True
        self.has_special = False
        if hasattr(self.info, 'special'):
            self.has_special = True
        self.needs_cbar = self.check_cbar()
        self.is_topattr_viz = False
        self.check_expattrs()


    def check_expattrs(self):
        # Checks for extra visualisations
        self.exp_attrs = ['author', 'canon', 'gender', 'year']
        attrs_str = [f'top{x}' for x in self.exp_attrs]
        if any(x in self.expname for x in attrs_str):
            self.is_topattr_viz = True
            self.exp_attrs.remove(self.info.attr)


    def get_feature_columns(self, df):
        df = deepcopy(df)
        interesting_cols = ['canon', 'year', 'gender', 'author']
        special_cols = ['cluster', 'clst_shape', 'gender_cluster', 'author_cluster']
        # Get list of attributes in interesting order
        if self.expname == 'attrviz':
            cols = interesting_cols + [col for col in df.columns if col not in interesting_cols and col not in special_cols and ('_color' not in col)]
        else:
            cols = interesting_cols
        return cols


    def check_cbar(self):
        # Check if any continuous attributes are shown.
        # If yes, a cbar is necessary.
        cbar = False
        if not self.is_cat:
            cbar = True
        if self.has_special and (self.info.special not in self.cat_attrs):
            cbar = True
        return cbar


    def add_text(self, ax, x=0, y=0, width=30):  
        ax.text(x=x, y=y, s=textwrap.fill(self.plttitle, width), fontsize=self.fontsize)


    def save_plot(self, plt):
        self.save_data(data=plt, data_type=self.data_type, file_name=None, file_path=self.vizpath, plt_kwargs={'dpi': 600})

        # if self.cmode == 'nk':
            # Save graphml
            # path = self.get_path(data_type='graphml', omit=always_omit+omit, name=vizname)
            # graph = self.save_graphml()
            # self.save_data(data=graph, data_type='graphml', file_name=None, file_path=path)


    def get_path(self, name='viz', omit: List[str]=[], data_type=None):
        # always_omit has to be added because 'special' it is not included in omit_default of the CombinationInfo objects on file
        always_omit = ['special']
        if data_type is None:
            data_type = self.data_type
        file_name = f'{name}-{self.info.as_string(omit=always_omit+omit)}.{data_type}'
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


    def add_subtitles(self, attrix, clstix, shapeix, combix=None, specix=None):
        self.get_ax(attrix).set_title('Attribute', fontsize=self.fontsize)
        self.get_ax(clstix).set_title('Cluster', fontsize=self.fontsize)
        self.get_ax(shapeix).set_title('Attributes and clusters (shapes)', fontsize=self.fontsize)
        if combix is not None:
            self.get_ax(combix).set_title('Attributes and clusters (combined)', fontsize=self.fontsize)
        if specix is not None:
            self.get_ax(specix).set_title(f'{self.info.special.capitalize()}', fontsize=self.fontsize)


