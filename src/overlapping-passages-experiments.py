# %%
%load_ext autoreload
%autoreload 2
import os
import time
import re
import hashlib
import multiprocessing

import difflib
from utils import DataHandler, get_filename_from_path
path = '/home/annina/Downloads/JCLS2022_Modeling-and-Predicting-Lit-Reception/corpora/ENG/Forrester_Andrew_The-Female-Detective_1864.txt'
path = '/home/annina/scripts/great_unread_nlp/data/corpus_corrections/manually_corrected_texts/eng/Forrester_Andrew_The-Female-Detective_1864.txt'

class DupSentencesFinder(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='duplicated_sentences', data_type='txt')


    # Function to clean and tokenize a sentence
    def tokenize(self, sentence):
        # Remove punctuation and convert to lowercase
        sentence = re.sub(r'[^\w\s]', '', sentence).lower()
        # Tokenize the sentence into words
        return sentence.split()
    

    # Find sentences that are similar, too slow for long texts
    def find_similar_sentences_fuzzy(self, file_path, similarity_threshold=0.8):
        # Read the file and split it into sentences
        with open(file_path, 'r') as f:
            sentences = [line.strip() for line in f]

        sentences = [self.tokenize(s) for s in sentences]

        print(f'{len(sentences)} sentences.')

        similar_sentences = []

        # Compare each pair of sentences
        for i in range(len(sentences)):                
            sentence1 = sentences[i]
            if i%1000 == 0:
                print(os.path.basename(file_path), 'sentence', i)
            for j in range(i + 1, len(sentences)):
                sentence2 = sentences[j]

                if len(sentence1) > 6 and len(sentence2)>6:
                    # Calculate the similarity ratio using SequenceMatcher
                    similarity_ratio = difflib.SequenceMatcher(None, sentence1, sentence2).ratio()

                    # If the similarity ratio is above the threshold, consider them similar
                    if similarity_ratio >= similarity_threshold:
                        similar_sentences.append([str(i), str(j), ' '.join(sentence1), ' '.join(sentence2), str(similarity_ratio), '\n---------------------------------------------\n'])

        return similar_sentences
    

    # Find sentences that are exactly the same
    def find_similar_sentences_exact(self, file_path, similarity_threshold=0.8):
        def hash_lines(lines):
            return [hashlib.md5(line.encode()).hexdigest() for line in lines]

        # Read the file and split it into sentences
        with open(file_path, 'r') as f:
            sentences = [line.strip() for line in f]
            sentences = [re.sub(r'[^\w\s]', '', sentence).lower() for sentence in sentences]
            sentences = [s for s in sentences if len(s.split()) > 6]

        line_hashes = hash_lines(sentences)
        similar_sentences = []

        for i, line_hash in enumerate(line_hashes):
            for j, other_line_hash in enumerate(line_hashes[i+1:], start=i+1):
                if line_hash == other_line_hash:
                        similar_sentences.append([str(i), str(j), sentences[i], sentences[j], '\n---------------------------------------------\n'])

        return similar_sentences

    
    def create_data(self, **kwargs):
        doc_path = kwargs['doc_path']
        path = '/home/annina/scripts/great_unread_nlp/data/text_tokenized'
        # 'Wieland_Christoph-Martin_Geschichte-des-weisen-Danischmed_1795.txt' ###########################
        path = os.path.join(path, self.language)

        filename = os.path.basename(doc_path)
        file_path = os.path.join(path, filename)
        self.logger.info(f'Comparing sentences for {filename}.')
        similarity_threshold = 0.8 

        similar_sentences = self.find_similar_sentences_exact(file_path, similarity_threshold)

        self.save_data(data=similar_sentences, file_name=get_filename_from_path(file_path), txt_sep='\n')

    # def create_all_data(self):
    #     for doc_path in self.doc_paths:
    #         print(get_filename_from_path(doc_path))
    #         startc = time.time()
    #         _ = self.load_data(load=False, file_name=get_filename_from_path(doc_path), doc_path=doc_path)
    #         print(f'{time.time()-startc}s to process {get_filename_from_path(doc_path)}')

    def create_all_data(self):
        # Use multiprocessing to process multiple documents in parallel
        num_processes = multiprocessing.cpu_count()-1
        pool = multiprocessing.Pool(num_processes)
        
        for doc_path in self.doc_paths:
            kwargs = {
                'load': False,
                'file_name': get_filename_from_path(doc_path),
                'doc_path': doc_path
            }
            pool.apply_async(self.load_data, kwds=kwargs)
            

        pool.close()
        pool.join()

for language in ['ger']:
    d = DupSentencesFinder(language)
    d.create_all_data()


# %%
%load_ext autoreload
%autoreload 2
import os
import matplotlib.pyplot as plt
from utils import DataHandler, get_filename_from_path

class LongestLinesFinder(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='duplicated_sentences', data_type='txt')


    def count_words(self, line):
        words = line.split()
        return len(words)

    def process_files(self):
        # Dictionary to store results
        result_dict = {}

        # Iterate through each file in the directory
        for filename in os.listdir(self.text_raw_dir):
            # Check if the file is a text file (you can customize this check)
            if filename.endswith('.txt'):
                file_path = os.path.join(self.text_raw_dir, filename)
                
                # Open the file and count words per line
                with open(file_path, 'r') as file:
                    for i, line in enumerate(file):
                        word_count = self.count_words(line)
                        result_dict[f'{filename}_{i}'] = word_count

        result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))

        # Keep only the entries with the 300 biggest values
        result_dict = {k: v for i, (k, v) in enumerate(result_dict.items()) if i < 200}
        result_dict = {key: result_dict[key] for key in sorted(result_dict.keys())}
        print(len(result_dict))

        return result_dict
    

    def plot(self):
            result = self.process_files()
            # Extract the filenames and shortest longest sentence lengths from sorted_result
            filenames = list(result.keys())
            lengths = list(result.values())

            # Identify the 20 longest lines and their corresponding indices
            top_20_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)[:20]

            highlighted_keys = [filenames[i] for i in top_20_indices]
            # highlighted_keys.sort()
            print("Keys of Highlighted Files:")
            for key in highlighted_keys:
                print(key)

            # Create a list of colors where the top 20 lines are highlighted
            colors = ['skyblue' if i not in top_20_indices else 'orange' for i in range(len(lengths))]

            # Create a bar plot
            plt.figure(figsize=(10, 30))
            bars = plt.barh(filenames, lengths, color=colors)
            plt.xlabel('Shortest Longest Sentence Length')
            plt.ylabel('Files')
            plt.title('Longest lines in corpus')
            plt.gca().invert_yaxis()  # Invert the y-axis to have the shortest at the top

            # Add a legend for the highlighted lines
            legend = plt.legend(handles=[bars[top_20_indices[0]]], labels=['Top 20 Longest Lines'], loc='upper right')
            legend.set_in_layout(False)

            plt.tight_layout()

            # Show the plot
            plt.show()

lf = LongestLinesFinder('eng')
lf.plot()
# %%
