
# %%
import sys
sys.path.append("..")
import pandas as pd
import os
from utils import copy_imgs_from_harddrive
import pandas as pd
import os
import re

import warnings

# Suppress the specific warning
warnings.filterwarnings("ignore", category=FutureWarning, message="In future versions `DataFrame.to_latex`")
# Display the table in a format that can be copied into latex

def make_latex_table(df, caption, label, outf, language, sparsification, print_sparsification=False):
    dfcols = df.columns
    # Add language in bold face
    new_row = pd.DataFrame([['' for _ in range(len(dfcols))]], columns=dfcols)
    if language == 'eng':
        new_row.loc[0, 'Dist'] = r'\textbf{English}'
    else:
        new_row.loc[0, 'Dist'] = r'\textbf{German}'
    print(new_row)
    # df = pd.concat([new_row, df]).reset_index(drop=True)

    df = df[dfcols]
    print(df)

    if 'Spars' in dfcols:
        column_format = 'lllccccc'
    else:
        column_format = 'llccccc'



    # print('\n\n\\begin{table}[H]')
    # print('\\centering')
    # print('\\scriptsize')
    # print(f'\\caption{{{caption}}}')
    # print('%\\textbf{English}')
    # print(df.to_latex(index=False, column_format=column_format))
    # print(f'\\label{{{label}}}')
    # print('\\end{table}')
    if print_sparsification and sparsification is not None and language == 'eng':
        if 'threshold' in sparsification:
            sparsification = sparsification.replace('threshold-', 'Threshold ')
            sparsification = sparsification.replace('%', '.')
        outf.write(f'\n\n\\subsection{{{sparsification}}}\n')
    outf.write('\\begin{table}[H]\n')
    outf.write('\\centering\n')
    outf.write('\\scriptsize\n')
    outf.write(f'\\caption{{{caption}}}\n')
    outf.write(f'\\label{{{label}}}\n')
    outf.write(f'\\adjustbox{{margin=-2cm 0cm}}{{ % Shifts the table 2cm to the left\n')
    outf.write(df.to_latex(index=False, column_format=column_format, escape=False))
    outf.write('}\n')
    outf.write('\\end{table}\n')


def extract_distance_and_sparsification(mxname):
    parts = mxname.split('_') # mxname, spars, ...
    # Extract the first part for Distance and the second part for Sparsification
    return parts[0], parts[1]


# Function to update the values
def update_clst_alg_param(value):
    value = re.sub(r'-nclust-\d+', '', value)
    parts = value.split('-')  # Split the string by '-'
    if 'hierarchical' in value:
        value = f'{parts[0]}, {parts[-1]}'  # Join the first and last parts
    elif 'louvain' in value:
        value = f'{parts[0]}, res {parts[-1]}'
    elif 'alpa' in value:
        value = 'alp'
    elif 'dbscan' in value:
        value = f'{parts[0]}, {parts[1]} {parts[2]}, {parts[3]} {parts[4]}'
    return value  # If not in the right format, return the value unchanged

# Apply the function to the column



def prepare_df(path, eval_metric, int_eval_metric, topn=None, sort_eval_ascending=False):
    if not 'csv' in path:
        path = os.path.join(path, 'df.csv')
    df = pd.read_csv(path)
    # Filter out invalid linkage functions
    mask = ~df['clst_alg_params'].str.contains('centroid|median|ward', case=False, na=False)
    df = df[mask]

    if eval_metric == 'vmeasure':
        collist = ['mxname', 'clst_alg_params', 'nclust', 'homogeneity', 'completeness', eval_metric, int_eval_metric]
    else:
        collist = ['mxname', 'clst_alg_params', 'nclust',  eval_metric, int_eval_metric]
   

    if df['mxname'].str.contains('dimensions').any():
        df[['mxname', 'sparsmode']] = df['mxname'].apply(lambda x: pd.Series(extract_distance_and_sparsification(x)))
        # df = df[['Distance', 'sparsmode', 'clst_alg_params', eval_metric, int_eval_metric]]

    if 'sparsmode' in df.columns:
        collist.insert(1, 'sparsmode')

    df['mxname'] = df['mxname'].str.replace('argamonquadratic', 'argamonquad') # shorted so that tables don't become too broad

    df = df[collist]
    df = df.sort_values(by=eval_metric, ascending=sort_eval_ascending)
    if topn is None:
        topn = len(df)
    df = df.head(topn)
    df['clst_alg_params'] = df['clst_alg_params'].str.replace('%', '.')
    df['clst_alg_params'] = df['clst_alg_params'].apply(update_clst_alg_param)
    if 'sparsmode' in df.columns:
        df['sparsmode'] = df['sparsmode'].str.replace('threshold-0%', 'threshold 0.')


    df = df.rename({
        'mxname': 'Dist', 
        'sparsmode': 'Spars', 
        'clst_alg_params': 'Alg + Params',
        'modularity': 'Modularity',
        'nclust': 'NrClst',
        'vmeasure': 'V-measure',
        'homogeneity': 'Homogeneity',
        'completeness': 'Completeness',
        'silhouette_score': 'Silhouette',
        'weighted_avg_variance': 'Weighted Av Var',
        'avg_variance': 'Av Variance',
        'smallest_variance': 'Smallest Var',
        'ext_wcss': 'WCSS'
        }, inplace=False, axis='columns')
    return df


def build_caption(attr, level, datadir, language, eval_metric, output_dir):
    if level == 'mx':
        mx_or_nk = 'unsparsified matrices'
    else:
        mx_or_nk = 'networks'
    if datadir == 'data':
        by_author = 'text-based'
        is_by_author = False
    else:
        by_author = 'author-based'
        is_by_author = True 


    tab_label = f'tab:{level}-{attr}-{eval_metric}-{language}-isbyauthor-{is_by_author}'

    if language == 'eng':
        lang_string = 'English'
    else:
        lang_string = 'German'
    if eval_metric == 'weighted_avg_variance':
        eval_metric = 'weighted average variance'
    if eval_metric == 'avg_variance':
        eval_metric = 'average variance'
    if eval_metric == 'vmeasure':
        eval_metric = 'V-measure'
    if eval_metric == 'smallest_variance':
        eval_metric = 'smallest variance'
    if eval_metric == 'ext_wcss':
        eval_metric = 'WCSS'        
    if output_dir == 'analysis':
        tab_caption = f'Top combinations for \\inquotes{{{attr}}} on {mx_or_nk} using {eval_metric}, {by_author}, {lang_string}'
    else:
        tab_caption = f'Top combinations for \\inquotes{{{attr}}} using {eval_metric}, {by_author}, {lang_string}'
        tab_label = f'{tab_label}_s2v'
    return tab_caption, tab_label


def write_latex_figure_mx2d3d(outf, nk_cluster_path, nk_attr_path, mx_cluster_path, mx_attr_path, caption, fig_label, attr):
    # LaTeX template for the figure
    outf.write(f"""\n\n\\begin{{figure}}[H]
        \\centering
        \\begin{{tabular}}{{cc}}
            \\begin{{subfigure}}{{0.45\\textwidth}}
                \\centering
                \\includegraphics[width=\\linewidth]{{{nk_cluster_path}}} 
            \\end{{subfigure}} &
            \\begin{{subfigure}}{{0.45\\textwidth}}
                \\centering
                \\includegraphics[width=\\linewidth]{{{nk_attr_path}}}
            \\end{{subfigure}} \\\\
            \\begin{{subfigure}}{{0.45\\textwidth}}
                \\centering
                \\includegraphics[width=\\linewidth]{{{mx_cluster_path}}}
                \\caption{{Clusters}}
            \\end{{subfigure}} &
            \\begin{{subfigure}}{{0.45\\textwidth}}
                \\centering
                \\includegraphics[width=\\linewidth]{{{mx_attr_path}}}
                \\caption{{{attr.capitalize()}}}
            \\end{{subfigure}} \\\\
        \\end{{tabular}}
        \\caption{{{caption}}}
        \\label{{{fig_label}}}
    \\end{{figure}}\n
    """)


def write_latex_figure_nk(outf, clst_path_eng, attr_path_eng, clst_path_ger, attr_path_ger, caption_eng, caption_ger, caption_fullfig, label, attr):
    
    outf.write(f"""\n\n\\begin{{figure}}[h!]
    \\centering
    \\captionsetup[subfigure]{{labelformat=empty}} % Suppress automatic subfigure labeling for this figure only
    % Top row
    \\begin{{subfigure}}[t]{{\\textwidth}}
        \\centering
        \\begin{{subfigure}}[t]{{0.45\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{clst_path_eng}}}
            \\caption{{Clusters}}
        \\end{{subfigure}}
        \\hfill
        \\begin{{subfigure}}[t]{{0.45\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{attr_path_eng}}}
            \\caption{{{attr.capitalize()}}}
        \\end{{subfigure}}
        \\caption{{{caption_eng}}}
    \\end{{subfigure}}
    
    \\vspace{{1em}} % Adjust the space between top and bottom rows
    
    % Bottom row
    \\begin{{subfigure}}[t]{{\\textwidth}}
        \\centering
        \\begin{{subfigure}}[t]{{0.45\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{clst_path_ger}}}
            \\caption{{Clusters}}
        \\end{{subfigure}}
        \\hfill
        \\begin{{subfigure}}[t]{{0.45\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{attr_path_ger}}}
            \\caption{{{attr.capitalize()}}}
        \\end{{subfigure}}
        \\caption{{{caption_ger}}}
    \\end{{subfigure}}
    
    % Overall caption for the figure
    \\caption{{{caption_fullfig}}}
    \\label{{{label}}}
\\end{{figure}}\n
""")
    


def write_latex_figure_mx2d3d_all_attr(outf, 
                              nk_cluster_path, nk_gender_path, nk_year_path, nk_canon_path, 
                              mx_cluster_path, mx_gender_path, mx_year_path, mx_canon_path, 
                              caption, fig_label):
    # LaTeX template for the figure with 2 rows and 4 images per row
    outf.write(f"""\n\n\\begin{{figure}}[H]
        \\centering
        \\begin{{tabular}}{{cccc}}
            \\begin{{subfigure}}{{0.24\\textwidth}}
                \\centering
                \\includegraphics[width=\\linewidth]{{{nk_cluster_path}}}
            \\end{{subfigure}} &
            \\begin{{subfigure}}{{0.24\\textwidth}}
                \\centering
                \\includegraphics[width=\\linewidth]{{{nk_gender_path}}}
            \\end{{subfigure}} &
            \\begin{{subfigure}}{{0.24\\textwidth}}
                \\centering
                \\includegraphics[width=\\linewidth]{{{nk_year_path}}}
            \\end{{subfigure}} &
            \\begin{{subfigure}}{{0.24\\textwidth}}
                \\centering
                \\includegraphics[width=\\linewidth]{{{nk_canon_path}}}
            \\end{{subfigure}} \\\\
            
            \\begin{{subfigure}}{{0.24\\textwidth}}
                \\centering
                \\includegraphics[width=\\linewidth]{{{mx_cluster_path}}}
                \\caption{{Clusters}}
            \\end{{subfigure}} &
            \\begin{{subfigure}}{{0.24\\textwidth}}
                \\centering
                \\includegraphics[width=\\linewidth]{{{mx_gender_path}}}
                \\caption{{Gender}}
            \\end{{subfigure}} &
            \\begin{{subfigure}}{{0.24\\textwidth}}
                \\centering
                \\includegraphics[width=\\linewidth]{{{mx_year_path}}}
                \\caption{{Year}}
            \\end{{subfigure}} &
            \\begin{{subfigure}}{{0.24\\textwidth}}
                \\centering
                \\includegraphics[width=\\linewidth]{{{mx_canon_path}}}
                \\caption{{Canon}}
            \\end{{subfigure}} \\\\
        \\end{{tabular}}
        \\caption{{{caption}}}
        \\label{{{fig_label}}}
    \\end{{figure}}\n
    """)



def prepare_latex_figure_mx(outf, attr, datadir, eval_metric, first_row, first_row_latex, level, language, sparsification, include_all_attr=False):
    if datadir == 'data':
        by_author = 'text-based'
        is_by_author = False
    else:
        by_author = 'author-based'
        is_by_author = True 

    if language == 'eng':
        langstr = 'English'
    else:
        langstr = 'German'

    mxnamestr, sparsstr = extract_distance_and_sparsification(first_row['mxname'])
    if '%' in sparsstr:
        sparsstr = sparsstr.replace('%', '.')

    keyattrs = ['gender', 'year', 'canon']

    paths = {}
    paths['nk_clst'] = f"/media/annina/elements/back-to-computer-240615/{datadir}/analysis_s2v/{language}/nk_singleimage_s2v/{first_row['mxname']}_{first_row['clst_alg_params']}_cluster.png"
    paths['mx_clst'] = f"/media/annina/elements/back-to-computer-240615/{datadir}/analysis_s2v/{language}/mx_singleimage_s2v/{first_row['mxname']}_{first_row['clst_alg_params']}.png"

    for keyattr in keyattrs:
        paths[f'nk_{keyattr}'] = f"/media/annina/elements/back-to-computer-240615/{datadir}/analysis_s2v/{language}/nk_singleimage_s2v/{first_row['mxname']}_{keyattr}.png"
        paths[f'mx_{keyattr}'] = f"/media/annina/elements/back-to-computer-240615/{datadir}/analysis_s2v/{language}/mx_singleimage_s2v/s2v-{first_row['mxname']}_{keyattr}.png"
    caption = f"Best combination for \\inquotes{{{attr}}}. Distance \\dist{{{mxnamestr}}}, sparsified with \\sparsname{{{sparsstr}}}, {first_row_latex['Alg + Params']}, {eval_metric} = {first_row[eval_metric]}, {langstr}. Groups of size 1 are colored gray."
    fig_label = f'fig:s2v-{level}-{attr}-{eval_metric}-{language}-isbyauthor-{is_by_author}'

    for pathname, source in paths.items():
        target = copy_imgs_from_harddrive(source, copy=True)
        paths[pathname] = target

    if 'louvain' in caption:
        caption = caption.replace('louvain, res', 'Louvain with resolution = ')
    if 'weighted_avg_variance' in caption:
        caption = caption.replace('weighted_avg_variance', 'weighted average variance')
    elif 'avg_variance' in caption:
        caption = caption.replace('avg_variance', 'average variance')
    elif 'smallest_variance' in caption:
        caption = caption.replace('smallest_variance', 'smallest variance')
    elif 'ext_wcss' in caption:
        caption = caption.replace('ext_wcss', 'WCSS')

    if sparsification is not None:
        fig_label = f"{fig_label}_{sparsification.replace('%', '.')}"

    if not include_all_attr:
        write_latex_figure_mx2d3d(outf, paths['nk_clst'], paths[f'nk_{attr}'], paths['mx_clst'], paths[f'mx_{attr}'], caption, fig_label, attr)
    else:
        write_latex_figure_mx2d3d_all_attr(outf, paths['nk_clst'], paths['nk_gender'], paths['nk_year'], paths['nk_canon'], 
                                        paths['mx_clst'], paths['mx_gender'], paths['mx_year'], paths['mx_canon'], 
                                        caption, fig_label)



def prepare_latex_figure_networks(outf, attr, datadir, eval_metric, first_rows, first_rows_latex, level, sparsification): 
    if datadir == 'data':
        by_author = 'text-based'
        is_by_author = False
    else:
        by_author = 'author-based'
        is_by_author = True 
    
    sparsstr_eng = first_rows['eng']['sparsmode']
    if '%' in sparsstr_eng:
        sparsstr_eng = sparsstr_eng.replace('%', '.')
    sparsstr_ger = first_rows['ger']['sparsmode']
    if '%' in sparsstr_ger:
        sparsstr_ger = sparsstr_ger.replace('%', '.')

    paths = {}
    captions = {}
    paths['clst_eng'] = f"/media/annina/elements/back-to-computer-240615/{datadir}/analysis/eng/nk_singleimage/{first_rows['eng']['mxname']}_{first_rows['eng']['sparsmode']}_{first_rows['eng']['clst_alg_params']}_cluster.png"
    paths['attr_eng'] = f"/media/annina/elements/back-to-computer-240615/{datadir}/analysis/eng/nk_singleimage/{first_rows['eng']['mxname']}_{first_rows['eng']['sparsmode']}_{attr}.png"
    paths['clst_ger'] = f"/media/annina/elements/back-to-computer-240615/{datadir}/analysis/ger/nk_singleimage/{first_rows['ger']['mxname']}_{first_rows['ger']['sparsmode']}_{first_rows['ger']['clst_alg_params']}_cluster.png"
    paths['attr_ger'] = f"/media/annina/elements/back-to-computer-240615/{datadir}/analysis/ger/nk_singleimage/{first_rows['ger']['mxname']}_{first_rows['ger']['sparsmode']}_{attr}.png"
    captions['eng'] = f"Distance \\dist{{{first_rows['eng']['mxname']}}}, sparsified with \\sparsname{{{sparsstr_eng}}}, {first_rows_latex['eng']['Alg + Params']}, {eval_metric} = {first_rows['eng'][eval_metric]}, English"
    captions['ger'] = f"Distance \\dist{{{first_rows['ger']['mxname']}}}, sparsified with \\sparsname{{{sparsstr_ger}}}, {first_rows_latex['ger']['Alg + Params']}, {eval_metric} = {first_rows['ger'][eval_metric]}, German"
    caption_fullfig = f'Best combination for \\inquotes{{{attr}}} on networks. Groups of size 1 are colored gray.'
    fig_label = f'fig:{level}-{attr}-{eval_metric}-isbyauthor-{is_by_author}'
    if sparsification is not None:
        fig_label = f"{fig_label}_{sparsification.replace('%', '.')}"

    for pathname, source in paths.items():
        target = copy_imgs_from_harddrive(source, copy=True)
        paths[pathname] = target

    for lang, caption in captions.items():
        if 'louvain' in caption:
            caption = caption.replace('louvain, res', 'Louvain with resolution = ')
        if 'weighted_avg_variance' in caption:
            caption = caption.replace('weighted_avg_variance', 'weighted average variance')
            print('replaced weighed av')
        elif 'avg_variance' in caption:
            caption = caption.replace('avg_variance', 'average variance')
        captions[lang] = caption

    write_latex_figure_nk(outf, paths['clst_eng'], paths['attr_eng'], paths['clst_ger'], paths['attr_ger'], captions['eng'], captions['ger'], caption_fullfig, fig_label, attr)
    return fig_label



import os
import pandas as pd
with open('latex/tables_for_latex.txt', 'w') as outf:
    # for attr in ['author', 'gender']:
    for attr in ['canon']:
        # for level in ['mx', 'nk']:
        for level in ['mx']:
            for datadir in ['data_author']:
                for output_dir in ['analysis_s2v']: # 'analysis', 
                        for sparsification in ['threshold-0%8', 'threshold-0%90', 'threshold-0%95', 'authormin', 'authormax', 'simmel-3-10', 'simmel-4-6', 'simmel-5-10', 'simmel-7-10']:
                            first_rows = {}
                            first_rows_latex = {}
                            tab_labels = {}
                            no_df_empty = True
                            outf.write('%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
                            outf.write('%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
                            for language in ['eng', 'ger']:
                                make_latex_figure = False
                                if attr == 'author' and datadir == 'data_author':
                                    print('invalid combination')
                                    continue
                                elif attr == 'gender' and datadir == 'data':
                                    print('invalid combination')
                                    continue
                                elif attr == 'year' and datadir == 'data':
                                    print('invalid combination')
                                elif attr == 'canon' and datadir == 'data':
                                    print('invalid combination')
                                    continue
                                if level == 'nk' and output_dir == 'analysis_s2v':
                                    print('invalid combination')
                                    continue
                                if output_dir == 'analysis_s2v' and sparsification == 'authormin' or output_dir == 'analysis_s2v' and sparsification == 'authormax':
                                    print('invalid combination')
                                    continue
                                else:
                                    make_latex_figure = True


                                if attr in ['author', 'gender']:
                                    eval_metric = 'vmeasure' ########################### vmeasure
                                    sort_eval_ascending = False
                                else:
                                    eval_metric = 'avg_variance' ###################3
                                    sort_eval_ascending = True
                                


                                extra_path_str = ''
                                # if attr == 'author' and output_dir == 'analysis_s2v':
                                #     extra_path_str = '_nclust-50-50'
                                if attr == 'gender' and eval_metric == 'ARI':
                                    extra_path_str = '_nclust-2-2'
                                # if attr == 'gender' and eval_metric == 'vmeasure':
                                #     extra_path_str = '_nclust-2-10'
                                if sparsification is None:
                                    extra_path_str = ''
                                else:
                                    extra_path_str = f'_{sparsification}'

                                # path = f'/media/annina/elements/back-to-computer-240615/data_author/analysis/{language}/nk_topgender_ARI_nclust-2-2'
                                path = f'/media/annina/elements/back-to-computer-240615/{datadir}/{output_dir}/{language}/{level}_top{attr}_{eval_metric}_nclust-2-5{extra_path_str}'
                                path = '/media/annina/elements/back-to-computer-240615/data_author/analysis_s2v/eng/mx_topcanon_avg_variance_nclust-2-5_threshold-0%95'
                                print(path)
                                df = pd.read_csv(os.path.join(path, 'df.csv'))
                                if df.empty:
                                    outf.write(f"% Empty df: {os.path.join(path, 'df.csv')}\n\n")
                                    no_df_empty = False
                                    continue

                                if level == 'mx':
                                    int_eval_metric = 'silhouette_score'
                                else:
                                    int_eval_metric = 'modularity'
                                # eval_metric = 'vmeasure'
                                
                                full_topn_rows = True # if true, topn = 20 regardless of eval metric
                                highest_eval_value = df.loc[0, eval_metric]
                                if highest_eval_value >= 0.3 or full_topn_rows:
                                    topn = 20
                                else:
                                    topn = 3

                                print('------------------------------------------------------------------------')
                                print(attr, level, datadir, language, eval_metric)
                                print(path)
                                print('------------------------------------------------------------------------')

                                outf.write('%------------------------------------------------------------------------\n%')
                                outf.write(f'{attr} {level} {datadir} {language} {eval_metric}\n%')
                                outf.write(f'{path}\n%')
                                outf.write('------------------------------------------------------------------------\n')

                                first_row = df.iloc[0].to_dict()
                                first_rows[language] = first_row

                                tab_caption, tab_label = build_caption(attr, level, datadir, language, eval_metric, output_dir)
                                if sparsification is not None:
                                    tab_label = f"{tab_label}_{sparsification.replace('%', '.')}"
                                tab_labels[language] = tab_label
                                df = prepare_df(path=path, eval_metric=eval_metric, int_eval_metric=int_eval_metric, topn=topn, sort_eval_ascending=sort_eval_ascending)
                                print(df)

                                first_row_latex = df.iloc[0].to_dict()
                                first_rows_latex[language] = first_row_latex
                                outf.write('%------------------------------------------------------------------------\n')
                                make_latex_table(df,tab_caption, tab_label, outf, language, sparsification, print_sparsification=True)

                                if output_dir == 'analysis_s2v' and make_latex_figure:
                                    if attr == 'gender' or attr == 'canon':
                                        prepare_latex_figure_mx(outf, attr, datadir, eval_metric, m, first_row_latex, level, language, sparsification, include_all_attr=True)
                                    else:
                                        prepare_latex_figure_mx(outf, attr, datadir, eval_metric, first_row, first_row_latex, level, language, sparsification)



                                
                            if make_latex_figure and level == 'nk' and no_df_empty:
                                fig_label = prepare_latex_figure_networks(outf, attr, datadir, eval_metric, first_rows, first_rows_latex, level, sparsification)
                                outf.write(f"\n\n The top combinations are shown in Tables \\ref{{{tab_labels['eng']}}} and \\ref{{{tab_labels['ger']}}}, and Figure \\ref{{{fig_label}}} shows the single best combination for each language.\n\n")


# year: /media/annina/elements/back-to-computer-240615/data/analysis/eng/mx_topyear_ext_calinski_harabasz interesting



# %%
