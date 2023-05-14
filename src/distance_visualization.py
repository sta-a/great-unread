import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import get_texts_by_author
import time
import networkx as nx
import os
from distance_analysis import get_mx_triangular, check_symmetric
import colorcet as cc


def get_colors_continuous(cluster_assignments):
    colors = iter(plt.cm.gist_ncar(np.linspace(0, 1, len(cluster_assignments)))) #hsv
    return colors


def get_colors_discrete():
    colors = iter(cc.glasbey_hv) #hsv
    return colors


def map_colors_authors(list_of_filenames):
    '''
    Map each filename to a color so that texts by the same author have the same color.
    '''

    # Map each author to a color.
    author_filename_mapping, _ = get_texts_by_author(list_of_filenames)
    colors = get_colors_discrete()
    author_color_mapping = {author: next(colors) for author, _ in author_filename_mapping.items()}

    fn_color_mapping = {}
    for author, list_of_filenames in author_filename_mapping.items():
        for fn in list_of_filenames:
            fn_color_mapping[fn] = author_color_mapping[author]
    return fn_color_mapping


def get_color_map(cluster_type, cluster_assignments):
    '''
    cluster_assignment: list of filenames or list of lists
    '''
    if cluster_type == 'author':
        cluster_color_mapping = map_colors_authors(cluster_assignments)
    else:
        colors = get_colors_discrete()
        cluster_color_mapping = {}
        for cluster in cluster_assignments:
            color = next(colors)
            for fn in cluster:
                cluster_color_mapping[fn] = color
    return cluster_color_mapping


def plot_distance_distribution(mx, mx_type, language, filename, data_dir):
    start = time.time()
    vals = get_mx_triangular(mx) # Get triangular matrix, ignore diagonal
    assert not np.any(np.isnan(vals)) # Test whether any element is nan

    # Find most common frequency
    _, counts = np.unique(vals, return_counts=True)
    ind = np.argmax(counts)

    print(f'Minimum {mx_type}: {min(vals)}. Maximum {mx_type}: {max(vals)}. Most common {mx_type}: {vals[ind]}.')

    fig = plt.figure(figsize=(20,6), dpi=300)
    ax = fig.add_subplot(111)
    if mx_type == 'distance':
        binsize = 0.001
        xtick_step = 1
    else:
        binsize = 0.1
        xtick_step = 5
    ax.hist(vals, bins = np.arange(0,max(vals) + 0.1, binsize), log=True, ec='black', color='black') #kwargs set edge color
    ax.set_xlabel(f'{mx_type.capitalize()}')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{mx_type.capitalize()} distribution {filename}')
    plt.xticks(np.arange(0, max(vals) + 0.1, step=xtick_step))
    plt.xticks(rotation=90)
    ax.grid(False)

    print(os.path.join(data_dir, 'distances', language, f'distribution_{mx_type}_{filename}.png'))
    plt.savefig(os.path.join(data_dir, 'distances', language, f'distribution_{mx_type}_{filename}.png'), format="png")
    plt.show()
    print(f'{time.time()-start}s to create {mx_type} distribution plot.')
