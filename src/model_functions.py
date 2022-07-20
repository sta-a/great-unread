import pandas as pd
import statistics
import os
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def drop_columns_transformer(X, drop_columns=None):
    def _drop_column(column):
        for string in drop_columns:
            if string in column:
                return True
        return False
    columns_before_drop = set(X.columns)
    if drop_columns != None:
        #print([_drop_column(column) for column in X.columns])
        X = X[[column for column in X.columns if not _drop_column(column)]].reset_index(drop=True)
    print('X shape after dropping columns: ', X.shape)
    columns_after_drop = set(X.columns)
    print(f'Dropped {len(columns_before_drop - columns_after_drop)} columns.') 
    return X

def custom_pca(X):
    X = StandardScaler().fit_transform(X)
    print('################################3 X before pca: ', X.shape)
    for i in range(5, X.shape[1], int((X.shape[1] - 5) / 10)):
        pca = PCA(n_components=i)
        X_trans = pca.fit_transform(X)
        if pca.explained_variance_ratio_.sum() >= 0.95:
            break
    print('##################################Nr dim after pca: ',X_trans.shape)
    return X_trans

# from sklearn.impute import KNNImputer
# def _impute(X):
#     imputer = KNNImputer()
#     X = imputer.fit_transform(X)
#     return X

def get_labels(label_type, language, canonscores_dir=None, sentiscores_dir=None, metadata_dir=None):
    if label_type == 'canon':
        canon_file = '210907_regression_predict_02_setp3_FINAL.csv'
        labels = pd.read_csv(os.path.join(canonscores_dir, canon_file), sep=';')[['file_name', 'm3']]
        labels = labels.rename(columns={'m3': 'y'})
        labels = labels.sort_values(by='file_name', axis=0, ascending=True, na_position='first')
        
    elif label_type == 'library':
        if language == 'eng':
            library_file = 'ENG_texts_circulating-libs.csv'
        else:
            library_file = 'GER_texts_circulating-libs.csv'
        labels = pd.read_csv(os.path.join(metadata_dir, library_file), sep=';', header=0)[['file_name', 'sum_libraries']]
        labels = labels.rename(columns={'sum_libraries': 'classified'})
        labels['classified'] = labels['classified'].apply(lambda x: 1 if x!=0 else 0)

    else:
        # Combine all labels and filenames in one file
        # File with sentiment scores for both tools
        if language == 'eng':
            scores_file = 'ENG_reviews_senti_FINAL.csv'
        else:
            scores_file = 'GER_reviews_senti_FINAL.csv'
        score_labels = pd.read_csv(os.path.join(sentiscores_dir, scores_file), sep=';', header=0)

        # File with class labels
        if language == 'eng':
            class_file = 'ENG_reviews_senti_classified.csv'
        else:
            class_file = 'GER_reviews_senti_classified.csv'
        class_labels = pd.read_csv(os.path.join(sentiscores_dir, class_file), sep=';', header=0)[['textfile', 'journal', 'file_name', 'classified']]

        labels = score_labels.merge(right=class_labels, 
                                    on=['textfile', 'journal', 'file_name'], 
                                    how='outer', 
                                    suffixes=(None, '_classlabels'), 
                                    validate='one_to_one')
        labels = labels[['sentiscore_average', 'sentiment_Textblob', 'file_name', 'classified']]

        # Sentiment Scores
        if (label_type == 'textblob') or (label_type == 'sentiart'):
            def _average_scores(group):
                textblob_value = group['sentiment_Textblob'].mean()
                sentiart_value = group['sentiscore_average'].mean()
                group['sentiment_Textblob'] = textblob_value
                group['sentiscore_average'] = sentiart_value
                return group        
            # Get one label per book
            labels = labels.groupby('file_name').apply(_average_scores)

        elif label_type == 'combined':
            def _aggregate_scores(row):
                if row['classified'] == 'positive':
                    score = row['sentiment_Textblob']
                elif row['classified'] == 'negative':
                    score = row['sentiscore_average']
                elif row['classified'] == 'not_classified':
                    score = statistics.mean([row['sentiment_Textblob'], row['sentiscore_average']]) 
                return score
            labels['combined'] = labels.apply(lambda row: _aggregate_scores(row), axis=1)
            #labels['combined'].sort_values().plot(kind='bar', figsize=(10, 5))

        elif (label_type == 'multiclass') or (label_type == 'binary'):
            #Assign one class label per work
            grouped_docs = labels.groupby('file_name')
            single_review = grouped_docs.filter(lambda x: len(x)==1)
            # If work has multiple reviews, keep labels only if reviews are not opposing (positive and negative)
            multiple_reviews = grouped_docs.filter(lambda x: len(x)>1 and not('negative' in x['classified'].values and 'positive' in x['classified'].values))
            #opposed_reviews = grouped_docs.filter(lambda x: len(x)>1 and ('negative' in x['classified'].values and 'positive' in x['classified'].values))  
            def _select_label(group):
                # Keep label with higher count, keep more extreme label if counts are equal
                count = group['classified'].value_counts().reset_index().rename(columns={'index': 'classified', 'classified': 'count'})
                if count.shape[0]>1:
                    if count.iloc[0,1] == count.iloc[1,1]:
                        grouplabel = count['classified'].max()
                    else:
                        grouplabel = count.iloc[0,0]
                    group['classified'] = grouplabel
                return group
            multiple_reviews = multiple_reviews.groupby('file_name').apply(_select_label)
            labels = pd.concat([single_review, multiple_reviews])
            labels['classified'] = labels['classified'].replace(to_replace={'positive': 3, 'not_classified': 2, 'negative': 1})

            if label_type =='binary':
                # Create label reviewed/not reviewed
                labels['classified'] = 1
            
    labels = labels.sort_values(by='file_name', axis=0, ascending=True, na_position='first')
    labels = labels.drop_duplicates(subset='file_name')
    if label_type == 'canon':
        labels = labels.rename(columns={'m3': 'y'})
    if label_type == 'textblob':
        labels = labels[['file_name', 'sentiment_Textblob']].rename(columns={'sentiment_Textblob': 'y'})
    elif label_type == 'sentiart':
        labels = labels[['file_name', 'sentiscore_average']].rename(columns={'sentiscore_average': 'y'})
    elif (label_type == 'multiclass') or (label_type == 'binary'):
        labels = labels[['file_name', 'classified']].rename(columns={'classified': 'y'})
    elif label_type == 'combined':
        labels = labels[['file_name', 'combined']].rename(columns={'combined': 'y'})
    elif label_type == 'library':
        labels = labels[['file_name', 'classified']].rename(columns={'classified': 'y'})
    return labels


def get_data(task_type, language, features_type, label_type, features_dir, canonscores_dir, sentiscores_dir, metadata_dir): 
    X = pd.read_csv(os.path.join(features_dir, f'{features_type}_features.csv'), index_col='file_name')
    y = get_labels(label_type, language, canonscores_dir, sentiscores_dir, metadata_dir).set_index('file_name')

    if task_type == 'regression':
        df = X.merge(right=y, left_index=True, right_index=True, how='inner', validate='many_to_one') ####################33'inner'
    else:
        # Select books written after year of first review
        # Keep only those texts that that were published after the first review had appeared
        df = X.merge(right=y, left_index=True, right_index=True, how='left', validate='many_to_one')
        df['y'] = df['y'].fillna(value=0)
        book_year = df['file_name'].str.replace('-', '_').str.split('_').str[-1].astype('int64')
        review_year = book_year[df['y'] != 0]
        df = df.loc[book_year>=min(review_year)]
    X = df.drop(labels=['y'], axis=1, inplace=False)
    y = df[['y']]

    return X, y
