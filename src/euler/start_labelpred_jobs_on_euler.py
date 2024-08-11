# %%
import os

# Define the combinations
languages = ['eng', 'ger']
attrs = ['author', 'gender', 'canon', 'year']
by_author_list = [True, False]

# Define the base command components
conda_env = 'nlpplus'
script_path = '/cluster/scratch/stahla/src_label/run_label_prediction.py'
output_template = 'output-label_{language}_{attr}_byauthor-{by_author}.txt'
error_template = 'error-label_{language}_{attr}_byauthor-{by_author}.txt'


print('-------------------conda nlpplus--------')
# Iterate over all combinations
for language in languages:
    for attr in attrs:
        for by_author in by_author_list:
            if by_author == True and attr == 'author':
                continue
            output_file = output_template.format(language=language, attr=attr, by_author=by_author)
            error_file = error_template.format(language=language, attr=attr, by_author=by_author)
            command = (
                f'sbatch --ntasks=50 --time=96:00:00 --mem-per-cpu=10000 '
                f'--output="{output_file}" --error="{error_file}" '
                f'--wrap="python {script_path} --language {language} --attr {attr}"'
            )
            if by_author == True:
                command = command[:-1] + ' --by_author"'
            print(f'Executing: {command}')
            # os.system(command)


# %%
