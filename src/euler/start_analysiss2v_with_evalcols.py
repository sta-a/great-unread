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

# Define the combinations
languages = ['eng', 'ger']
by_author_list = [True, False]





# Define the base command components
conda_env = 'nlpplus_updated'
script_path = '/cluster/scratch/stahla/src_evalcol/run_embeddings.py'
output_template = 'output-evalcol_{language}_{evalcol}_byauthor-{by_author}.txt'
error_template = 'error-evalcol_{language}_{evalcol}_byauthor-{by_author}.txt'


print('-------------------conda nlpplus_updated--------')
# Iterate over all combinations
for language in languages:
    for by_author in by_author_list:
        for evalcol in evalcols_list:
            output_file = output_template.format(language=language, evalcol=evalcol, by_author=by_author)
            error_file = error_template.format(language=language, evalcol=evalcol, by_author=by_author)
            command = (
                f'sbatch --time=96:00:00 --mem-per-cpu=10000 '
                f'--output="{output_file}" --error="{error_file}" '
                f'--wrap="python {script_path} --language {language} --evalcol {evalcol}"'
            )
            if by_author == True:
                command = command[:-1] + ' --by_author"'
            print(f'Executing: {command}')
            # os.system(command)
