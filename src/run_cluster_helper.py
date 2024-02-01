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


delete_png_files(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/mxtop'])
nkc = MxExp(language='eng')
# nkc.log_combinations()
nkc.viz_topk()















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


# delete_png_files(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/nktop'])
nkc = NkExp(language='eng')
# nkc.log_combinations()
nkc.viz_topk()

# %%


# add arrows to nk plots
