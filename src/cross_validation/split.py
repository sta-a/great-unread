import heapq
from collections import Counter
import random
random.seed(56)

class AuthorCV():
    '''
    Split book names into n folds.
    All works of an author are put into the same fold.
    Adapted from https://www.titanwolf.org/Network/q/b7ee732a-7c92-4416-bc80-a2bd2ed136f1/y
    '''
    def __init__(self, df, n_folds, seed, stratified=False, return_indices=False):
        self.df = df
        self.n_folds = n_folds
        self.stratified = stratified
        self.return_indices = return_indices
        self.file_names = df['file_name']
        self.author_bookname_mapping, self.works_per_author = self.get_author_books()
        random.seed(seed)

    def get_author_books(self):
        authors = []
        author_bookname_mapping = {}
        #Get books per authors
        for file_name in self.file_names:
            author = '_'.join(file_name.split('_')[:2])
            authors.append(author)
            if author in author_bookname_mapping:
                author_bookname_mapping[author].append(file_name)
            else:
                author_bookname_mapping[author] = []
                author_bookname_mapping[author].append(file_name)
                
        # Aggregate if author has collaborations with others
            agg_dict = {'Hoffmansthal_Hugo': ['Hoffmansthal_Hugo-von'], 
                        'Schlaf_Johannes': ['Holz-Schlaf_Arno-Johannes'],
                         'Arnim_Bettina': ['Arnim-Arnim_Bettina-Gisela'],
                         'Stevenson_Robert-Louis': ['Stevenson-Grift_Robert-Louis-Fanny-van-de', 
                                                   'Stevenson-Osbourne_Robert-Louis-Lloyde']}
            
        for author, aliases in agg_dict.items():
            if author in authors:
                for alias in aliases:
                    if alias in authors:
                        author_bookname_mapping[author].extend(author_bookname_mapping[alias]) 
                        del author_bookname_mapping[alias]
                        authors = [author for author in authors if author != alias]
        
        works_per_author = Counter(authors)
        return author_bookname_mapping, works_per_author
    
    def get_folds(self):
        splits = [[] for _ in range(0,self.n_folds)]

        if self.stratified == True:
            rare_labels = sorted(self.df['y'].unique().tolist())[1:]
            splits_counter = [0 for _ in range(0, self.n_folds)]
            for rare_label in rare_labels:
                # If stratified, first distribute authors that have rarest label over split so that the author is assigned to the split with the smallest number or rarest labels
                counts = [(0,i) for i in range (0, self.n_folds)]
                # heapify based on first element of tuple, inplace
                heapq.heapify(counts)
                for author in list(self.works_per_author.keys()):
                    rare_label_counter = 0
                    for curr_file_name in self.author_bookname_mapping[author]:
                        if self.df.loc[self.df['file_name'] == curr_file_name].squeeze().at['y'] == rare_label:
                            rare_label_counter += 1
                    if rare_label_counter != 0:
                        author_workcount = self.works_per_author.pop(author)
                        count, index = heapq.heappop(counts)
                        splits[index].append(author)
                        splits_counter[index] += author_workcount
                        heapq.heappush(counts, (count + rare_label_counter, index))
            totals = [(splits_counter[i],i) for i in range(0,len(splits_counter))]
        else:
            totals = [(0,i) for i in range (0, self.n_folds)]
        # heapify based on first element of tuple, inplace
        heapq.heapify(totals)
        while bool(self.works_per_author):
            author = random.choice(list(self.works_per_author.keys()))
            author_workcount = self.works_per_author.pop(author)
            # find split with smallest number of books
            total, index = heapq.heappop(totals)
            splits[index].append(author)
            heapq.heappush(totals, (total + author_workcount, index))

        if not self.return_indices:
            # Return file_names in splits
            #Map author splits to book names
            map_splits = []
            for split in splits:
                new = []
                for author in split:
                    new.extend(self.author_bookname_mapping[author])
                map_splits.append(new)

            if self.stratified == True:
                for split in map_splits:
                    split_df = self.df[self.df['file_name'].isin(split)]
        else:
            # Return indices of file_names in split
            file_name_idx_mapping = dict((file_name, index) for index, file_name in enumerate(self.file_names))
            map_splits = []
            for split in splits:
                test_split = []
                for author in split:
                    # Get all indices from file_names from the same author
                    test_split.extend([file_name_idx_mapping[file_name] for file_name in  self.author_bookname_mapping[author]])
                # Indices of all file_names that are not in split
                train_split = list(set(file_name_idx_mapping.values()) - set(test_split))
                map_splits.append((train_split, test_split))
        return map_splits
