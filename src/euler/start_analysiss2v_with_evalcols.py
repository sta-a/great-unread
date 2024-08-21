import os
evalcols_list = [
    'ext_silhouette',
    'ext_davies_bouldin',
    'ext_calinski_harabasz',
    'avg_variance',
    'weighted_avg_variance',
    'smallest_variance',
    'ext_wcss',
    'ARI',
    'nmi',
    'fmi',
    'mean_purity',
    'homogeneity',
    'completeness',
    'vmeasure',
    'ad_nmi'
]
# don't use 'singleimage' because it uses different viz class, making different images than those for clusters
# Define the combinations
languages = ['eng', 'ger']
by_author_list = [True, False]
outputdir_list = ['analysis_s2v', 'analysis']
cmode_list = ['mx', 'nk']

cmode_list = ['mx'] ##################################################
outputdir_list = ['analysis_s2v'] ###########################

errorfiles_list = []
output_file_list = []

# Define the base command components
conda_env = 'nlpplus_updated'
script_path = '/cluster/scratch/stahla/src_evalcol/run_experiments.py'
output_template = 'output-exp_{language}_{output_dir}_{cmode}_{evalcol}_byauthor-{by_author}.txt'
error_template = 'error-exp_{language}_{output_dir}_{cmode}_{evalcol}_byauthor-{by_author}.txt'


print('-------------------conda nlpplus_updated --------')
# Iterate over all combinations
for language in languages:
    for by_author in by_author_list:
        for output_dir in outputdir_list:
            for cmode in cmode_list:
                if output_dir == 'analysis_s2v' and cmode =='nk':
                    continue
                for evalcol in evalcols_list:
                    output_file = output_template.format(language=language, output_dir=output_dir, cmode=cmode, evalcol=evalcol, by_author=by_author)
                    error_file = error_template.format(language=language, output_dir=output_dir, cmode=cmode, evalcol=evalcol, by_author=by_author)
                    # if error_file not in errorfiles_list: ########################
                        # continue
                    # if output_file in output_file_list:
                        # continue


                    mem = 90000
                    if 'smallest_variance' in error_file:
                        mem = 900000
                    command = (
                        f'sbatch --time=22:00:00 --mem-per-cpu={mem} '
                        f'--output="{output_file}" --error="{error_file}" '
                        f'--wrap="python {script_path} --language {language} --output_dir {output_dir} --cmode {cmode} --evalcol {evalcol}"'
                    )
                    if by_author == True:
                        command = command[:-1] + ' --by_author"'
                    print(f'Executing: {command}')
                    # os.system(command)
