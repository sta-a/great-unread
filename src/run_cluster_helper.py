# %%

%load_ext autoreload
%autoreload 2
# Don't display plots
# %matplotlib agg

import sys
sys.path.append("..")
from analysis.experiments import NkExp, MxExp
from helpers import remove_directories, delete_png_files

import logging
logging.basicConfig(level=logging.DEBUG)


# p = ['/home/annina/scripts/great_unread_nlp/data/analysis']
# remove_directories(p)
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

from analysis.experiments import NkExp, MxExp
from helpers import remove_directories, delete_png_files

import logging
logging.basicConfig(level=logging.DEBUG)


# p = ['/home/annina/scripts/great_unread_nlp/data/analysis']
# remove_directories(p)
nkex = NkExp(language='eng')
nkex.run_experiments()

# %%
