# %%
%load_ext autoreload
%autoreload 2
import os
import time
import re
import difflib
from difflib import SequenceMatcher
from utils import DataHandler, get_filename_from_path
path = '/home/annina/Downloads/JCLS2022_Modeling-and-Predicting-Lit-Reception/corpora/ENG/Forrester_Andrew_The-Female-Detective_1864.txt'
path = '/home/annina/scripts/great_unread_nlp/data/corpus_corrections/manually_corrected_texts/eng/Forrester_Andrew_The-Female-Detective_1864.txt'

class DupSentencesFinder(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='duplicated_sentences', data_type='txt')


    # Function to clean and tokenize a sentence
    def clean_and_tokenize(self, sentence):
        # Remove punctuation and convert to lowercase
        sentence = re.sub(r'[^\w\s]', '', sentence).lower()
        # Tokenize the sentence into words
        return sentence.split()
    

    # Function to find similar sentences using difflib's SequenceMatcher
    def find_similar_sentences(self, file_path, similarity_threshold=0.8):
        # Read the file and split it into sentences
        with open(file_path, 'r') as f:
            sentences = [line.strip() for line in f]

        print(f'{len(sentences)} sentences.')

        similar_sentences = []

        # Compare each pair of sentences
        for i in range(len(sentences)):                
            sentence1 = sentences[i]
            tokens1 = self.clean_and_tokenize(sentence1)
            if i%100 == 0:
                print('sentence', i)
            for j in range(i + 1, len(sentences)):
                sentence2 = sentences[j]
                tokens2 = self.clean_and_tokenize(sentence2)

                if len(tokens1) > 5 and len(tokens2)>2:
                    # Calculate the similarity ratio using SequenceMatcher
                    similarity_ratio = difflib.SequenceMatcher(None, tokens1, tokens2).ratio()

                    # If the similarity ratio is above the threshold, consider them similar
                    if similarity_ratio >= similarity_threshold:
                        similar_sentences.append([sentence1, sentence2, str(similarity_ratio), '\n---------------------------------------------\n'])

        return similar_sentences

    
    def create_data(self, **kwargs):
        doc_path = kwargs['doc_path']
        path = '/home/annina/scripts/great_unread_nlp/data/text_tokenized'
        path = os.path.join(path, self.language)

        filename = os.path.basename(doc_path)
        file_path = os.path.join(path, filename)
        self.logger.info(f'{self.__class__.__name__}: Comparing sentences for {filename}.')
        similarity_threshold = 0.8 

        similar_sentences = self.find_similar_sentences(file_path, similarity_threshold)

        if similar_sentences:
            print("Similar sentences found:")
            for sentence1, sentence2, similarity_ratio, _ in similar_sentences:
                print(f"{sentence1} and {sentence2} are similar with a similarity ratio of {similarity_ratio}.")
        else:
            print("No similar sentences found in the file.")
        self.save_data(data=similar_sentences, file_name=get_filename_from_path(file_path), txt_sep='\n')

    def create_all_data(self):
        for doc_path in self.doc_paths:
            startc = time.time()
            _ = self.load_data(load=False, file_name=get_filename_from_path(doc_path), doc_path=doc_path)
            print(f'{time.time()-startc}s to process {get_filename_from_path(doc_path)}')

for language in ['eng', 'ger']:
    d = DupSentencesFinder(language)
    d.create_all_data()







# %%
class PassagesFinder(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='overlapping_passages', data_type='svg')

    def split_text_by_words(self, text, words_per_chunk, window_size):
        words = text.split()
        chunks = [words[i:i + words_per_chunk] for i in range(0, len(words), window_size)]
        return [' '.join(chunk) for chunk in chunks]

    def find_repeated_passages(self, text, words_per_chunk, window_size, similarity_threshold):
        passages = []
        chunks = self.split_text_by_words(text, words_per_chunk, window_size)
        num_chunks = len(chunks)

        for i in range(num_chunks - 1):
            window = chunks[i]

            for j in range(i + 1, num_chunks):
                candidate = chunks[j]

                similarity = SequenceMatcher(None, window, candidate).ratio()

                if similarity >= similarity_threshold:
                    print(similarity, '\n', window, '\n', candidate, '='*100)
                    passages.append((window, candidate, similarity))

        return passages

    def run(self):
        words_per_chunk = 5  # Adjust the words per chunk as needed
        window_size = int(words_per_chunk/2)
        similarity_threshold = 0.55  # Adjust the similarity threshold as needed

        for doc_path in self.doc_paths:
            with open(doc_path, 'r') as file:
                text = file.read()
        text = """This is a short example text to demonstrate how the code works. It contains repeated passages to test the passage finding algorithm. example text to demonstrate how the code works. The goal is to identify similar passages with a specified similarity threshold."""

        passages = self.find_repeated_passages(text, words_per_chunk, window_size, similarity_threshold)

        for passage in passages:
            print(passage)

for language in ['eng', 'ger']:
    pf = PassagesFinder(language)

# %%