# Environment
The conda environment is provided in `src/environments/environment.yml`.

# Data
The texts and metadata can be downloaded from [here](https://figshare.com/articles/dataset/JCLS2022_-_Modeling_and_Predicting_Literary_Reception/19672410/1).
The expected directory structure is as follows:
```
great-unread
├── data/
│   ├── metadata/
│   │   ├── eng/
│   │   ├── ger/
│   ├── text_raw/
│   │   ├── eng/
│   │   ├── ger/
│   ├── preprocess/
├── src/
```

# Scripts
## Project 1:  Prediction of canonization scores from text features
`run_features.py` extracts the text features.
`prepare_features.py` can be run before `run_features.py` to extract basic features that are slow to process before calculating the more complex features. This is not necessary, since all data preparation steps are also called from within `run_features.py`, but it is convenient.

`run_prediction.py` runs the cross-validation to find the best model to predict the canonization scores from the text features. `run_prediction_evaluation.py` evaluated the results of the cross-validation and delivers the results for the prediction.

## Project 2: Stylometric networks
`run_cluster.py` creates similarity matrixes for all combinations of text distance measure and network sparsification techniques and runs and evaluates different clustering algorithms on them.

## Project 3: Network positions
`run_embeddings.py` creates struc2vec embedding for different parameter combinations, creates visualiziations, and runs the mirror graph experiment.
Then it runs the same clustering and evaluation pipeline as in step 2 on the similarity matrices obtained from these embeddings.

## Author-based analysis
For author-based analysis, all texts by the same author are combined into one. The script for combining the texts is `run_author.py`.
