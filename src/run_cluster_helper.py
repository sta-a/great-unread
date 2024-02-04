# %%

%load_ext autoreload
%autoreload 2
# Don't display plots
# %matplotlib agg

import sys
sys.path.append("..")
from cluster.experiments import NkExp, MxExp
from helpers import remove_directories, delete_png_files

import logging
logging.basicConfig(level=logging.DEBUG)


delete_png_files(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/mxtop'])
mxex = MxExp(language='eng')
mxex.run_experiments()




# %%

%load_ext autoreload
%autoreload 2
# Don't display plots
# %matplotlib agg

import sys
sys.path.append("..")
import os
import argparse

from cluster.combinations import MxCombinations, NkCombinations
from cluster.experiments import NkExp, MxExp
from helpers import remove_directories, delete_png_files

import logging
logging.basicConfig(level=logging.DEBUG)


delete_png_files(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/nktop'])
nkex = NkExp(language='eng')
nkex.run_experiments()

# %%
# 'anova_pval', 'logreg_acc',  mit neuen resultaten ####################