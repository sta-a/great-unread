import os

# Define the combinations
languages = ['eng', 'ger']
features = ['cacb', 'chunk', 'book', 'baac']
tasks = ['regression-canon'] #, 'regression-senti']
folds = [0, 1, 2, 3, 4]

# Define the base command components
conda_env = 'nlpplus_updated'
script_path = '/cluster/scratch/stahla/src_byauthor/run_prediction.py'
data_dir = '/cluster/scratch/stahla/data_author'
output_template = 'output_{language}_{task}_{features}_fold{fold}_byauthor.txt'
error_template = 'errors_{language}_{task}_{features}_fold{fold}_byauthor.txt'


print('-------------------conda nlpplus--------')
# Iterate over all combinations
for language in languages:
    for feature in features:
        for task in tasks:
            for fold in folds:
                output_file = output_template.format(language=language, task=task, features=feature, fold=fold)
                error_file = error_template.format(language=language, task=task, features=feature, fold=fold)
                command = (
                    f'sbatch --ntasks=50 --time=96:00:00 --mem-per-cpu=10000 '
                    f'--output="{output_file}" --error="{error_file}" '
                    f'--wrap="python {script_path} {language} {data_dir} {task} --features {feature} --fold {fold}"'
                )
                print(f'Executing: {command}')
                os.system(command)

