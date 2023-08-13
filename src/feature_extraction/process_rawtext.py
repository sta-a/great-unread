# %%
import os
import spacy
import time
import string
import random
import sys
import logging
from collections import Counter
logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import DataHandler, check_equal_line_count, check_equal_files,  get_filename_from_path, get_doc_paths
from .preprocessor import Preprocessor

class SentenceTokenizer(DataHandler):
    def __init__(self, language):
        super().__init__(language, 'text_sentences', data_type='txt')
        if self.language == 'eng':
            self.model_name = 'en_core_web_sm'
        else:
            self.model_name = 'de_core_news_sm'
        self.nlp = spacy.load(self.model_name)

    def create_data(self, doc_path=None, text=None):
        print(f'doc path: {doc_path}, \ntext: {text}')
        self.logger.info(f'Tokenizing sentences: {doc_path}')
        start = time.time()
        if doc_path is None and text is None:
            raise ValueError("Either doc_path or text must not be None.")
        if doc_path is not None and text is not None:
            raise ValueError("Only one of doc_path or text should be provided, not both.")
        
        if doc_path is not None:
            with open(doc_path, 'r') as reader:
                text = reader.read().strip()

            # Preprocess full text
            pp = Preprocessor(self.language, doc_path)
            text = pp.preprocess_text(text)

        i = 0
        all_sentences = []
        while True:
            # Process in chunks because of Spacy character limit
            current_text = text[i:i+500000]
            current_sentences = [(sent.text, sent.text_with_ws) for sent in self.nlp(current_text).sents]

            # Ensures that complete sentences are extracted 
            # Check if there's only one sentence in the current chunk
            if len(current_sentences) == 1:
                # Add the single sentence to the list of all sentences
                all_sentences.extend([sents[0].strip() for sents in current_sentences])
                break  # Exit the loop since the current chunk is fully processed
            else:
                # Add all sentences except the last one to the list of all sentences
                all_sentences.extend([sents[0].strip() for sents in current_sentences[:-1]])
                # Update the index 'i' to skip the characters corresponding to the stripped whitespace
                i += len(''.join([sents[1] for sents in current_sentences[:-1]]))

        self.save_data(data=all_sentences, file_name=get_filename_from_path(doc_path))
        print(f'Time for sentence tokenization: {time.time()-start}')
        return all_sentences
    
    def tests_tokenization(self, doc_path):
        start = time.time()
        all_sentences = self.create_data(doc_path=doc_path)
        text = ' '.join(all_sentences)
        new_all_sentences = self.create_data(text=text)
        if not all_sentences == new_all_sentences:
            print(len(all_sentences), len(new_all_sentences))
            print('Sentence tokenization is not always the same.')
        print(f'Time for calculating tokenization test: {time.time()-start}.')
        with open('test_sentence_tokenization.txt', "w") as file:
            for item1, item2 in zip(all_sentences, new_all_sentences):
                file.write(item1 + '\n')
                file.write(item2 + '\n')


class TextChunker(DataHandler):
    def __init__(self, language, tokens_per_chunk):
        super().__init__(language=language, output_dir='text_chunks', data_type='txt', tokens_per_chunk=tokens_per_chunk)
        self.tokens_per_chunk = tokens_per_chunk
        self.tpc_increase = int(self.tokens_per_chunk/100)
        self.tolerance = 0.2 # How much the nr tokens per chunk can deviate from the predefined value 
        self.tpc_window = self.get_tpc_window()
        print(self.tpc_window)
        self.sentencetokenizer = SentenceTokenizer(self.language)

    def get_tpc_window(self):
        nr_steps = int((self.tokens_per_chunk * self.tolerance) / self.tpc_increase)
        # print(nr_steps)
        # for i in range(nr_steps):
        #     print(i)
        # return [self.tokens_per_chunk + (i * self.tpc_increase * (-1)**i) for i in range(nr_steps)]
        l = [(self.tokens_per_chunk + i * self.tpc_increase, self.tokens_per_chunk - i * self.tpc_increase) for i in range(1, nr_steps)]
        l = [self.tokens_per_chunk] + [item for tup in l for item in tup]
        print(l)
        return l

    def distribute_sents_to_chunks(self, tpc):
        chunks = []
        next_chunk = []
        token_count = 0
        for sentence in self.sentences:
            sentence_tokens = sentence.split()
            if len(sentence_tokens) < tpc:
                # print('nr tokens in sent', len(sentence_tokens))
                if token_count < tpc:
                    next_chunk.extend(sentence_tokens)
                    token_count += len(sentence_tokens)
                else:
                    chunks.append(' '.join(next_chunk))
                    next_chunk = sentence_tokens
                    token_count = len(sentence_tokens)
            else:
                for token in sentence_tokens:
                    if token_count  < tpc:
                        next_chunk.extend([token])
                        token_count += 1
                    else:
                        chunks.append(' '.join(next_chunk))
                        next_chunk = [token]
                        token_count = 1

        return chunks, next_chunk
    
    def create_data(self, doc_path):
        self.logger.info(f'Creating chunks: {doc_path}')

        print(get_filename_from_path(doc_path))
        self.sentences = self.sentencetokenizer.load_data(file_name=get_filename_from_path(doc_path), doc_path=doc_path)

        self.logger.info(f'Splitting text into chunks.')
        chunks, next_chunk = self.distribute_sents_to_chunks(self.tokens_per_chunk)
        # Deal with remaining tokens at the end of the texts that are shorter thant tokens_per_chunk
        i = 0
        while len(next_chunk) > self.tolerance*self.tokens_per_chunk:
            chunks, next_chunk = self.distribute_sents_to_chunks(tpc=self.tpc_window[i])
            print('len next chunk', len(next_chunk))
            i += 1
            
        # If nr of remaining tokens is below tolerance, add them to previous chunk
        assert len(next_chunk) <= self.tolerance*self.tokens_per_chunk, 'Nr of tokens at the end of the text that belong to no chunk are above tolerance.'
        assert all(len(string.split()) >= self.tokens_per_chunk for string in chunks), "Some strings don't meet the token count requirement."
        print('next_chunk', len(next_chunk), 'chunks', len(chunks))
        chunks[-1] += ' '.join(next_chunk)
        self.logger.info(f'Finished splitting text into chunks. Chunk length = {self.tpc_window[i-1]}.')

        self.save_data(data=chunks, file_name=get_filename_from_path(doc_path))


class Tokenizer(DataHandler):
    def __init__(self, language=None, doc_paths=None, tokens_per_chunk=500, data_type='txt'):
        super().__init__(language=language, output_dir='text_tokenized', tokens_per_chunk=tokens_per_chunk, data_type=data_type)
        self.doc_paths = doc_paths
        self.nlp = self.load_spacy_model()
        self.textchunker = TextChunker(language=self.language, tokens_per_chunk=self.tokens_per_chunk)

    def load_spacy_model(self):
        if self.language == 'eng':
            model_name = 'en_core_web_sm'
        elif self.language == 'ger':
            model_name = 'de_core_news_sm'
        else:
            raise Exception(f'Not a valid language {self.language}')

        try:
            nlp = spacy.load(model_name)
            return nlp
        except OSError:
            print(f'The model {model_name} for Spacy is missing.')
    

    def tokenize_words(self, chunks):
        start = time.time()
        new_chunks = []
        for chunk in chunks:
            chunk = [token.text for token in self.nlp(chunk)] 
            chunk = ' '.join(chunk)
            new_chunks.append(chunk)
        print(f'Time for word tokenization: {time.time()-start}')
        return new_chunks


    # def tokenize_words(self, chunks):
    #     # Helper function for testing mode
    #     # Return list to be stored
    #     new_chunks = chunks.split('.')
    #     return new_chunks
    

    def create_data(self, doc_path, remove_punct=False, lower=False, as_chunk=True):
        self.logger.info(f'Tokenizing {doc_path}')

        chunks = self.textchunker.load_data(file_name=get_filename_from_path(doc_path), doc_path=doc_path) # load chunks as a list of chunks
        # with open(doc_path, 'r') as reader: # Helper for testing
        #     text = reader.read().strip()
        # pp = Preprocessor(self.language, doc_path)
        # chunks = pp.preprocess_text(text)

        chunks = self.tokenize_words(chunks) 
        self.save_data(file_name=get_filename_from_path(doc_path), data=chunks)

        chunks = Postprocessor(remove_punct=remove_punct, lower=lower).postprocess_chunks(chunks)

        if as_chunk == False:
            chunks = ' '.join(chunks)
        #     self.logger.info('Returning tokenized words as one string.')
        # else:
        #     self.logger.info('Returning tokenized chunks as list of strings.')
        self.logger.info(f'Finished tokenizing: {doc_path}.\n----------------------------')
        return chunks
    
    def create_all_data(self):
        start = time.time()
        for i, doc_path in enumerate(self.doc_paths):
            _ = self.load_data(load=False, file_name=get_filename_from_path(doc_path), doc_path=doc_path)
        # print(f'{time.time()-start}s to tokenize all texts')
    
    def check_data(self):
        assert check_equal_line_count(self.output_dir, self.textchunker.output_dir, self.data_type)
        assert check_equal_files(self.output_dir.replace('/text_tokenized', '/text_raw'), self.output_dir)
        dc = DataChecker(self.language)
        dc.count_chunks_per_doc()
        dc.count_tokens_per_chunk()



class Postprocessor():
    def __init__(self, remove_punct=False, lower=False):
        self.remove_punct = remove_punct
        self.lower = lower

    def postprocess_text(self, text):
        if self.remove_punct:
            punctuation = set(string.punctuation + '’' + '‘' + '—' + '“' + '”' + '–')
            text = text.split()
            # Remove list items that consist only of puncutation as well as punctuation that belongs to other tokens.
            # "'" is ignored if it is part of a word (Mary's -> Mary + 's)
            text = [''.join(char for char in item
                        if char not in punctuation)
                        for item in text if item != '']
            text = ' '.join(text)

        if self.lower:
            text = text.lower()
        return text

    def postprocess_chunks(self, chunks):
        new_chunks = []
        for chunk in chunks:
            new_chunk = self.postprocess_text(chunk)
            new_chunks.append(new_chunk)
        return new_chunks

class DataChecker(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='text_statistics', data_type='png')
        self.text_raw_dir = os.path.join(self.data_dir, 'text_raw', language)
        self.text_tokenized_dir = self.text_raw_dir.replace('/text_raw', '/text_tokenized') 
        self.doc_paths = get_doc_paths(self.text_raw_dir)


    def count_chunks_per_doc(self):
        # Count chunks
        nr_chunks_per_doc = {}

        # Nr chunks per doc
        for doc_path in self.doc_paths:
            text_tokenized_path = doc_path.replace('/text_raw', '/text_tokenized') 
            with open(text_tokenized_path, 'r') as f:
                nr_chunks = sum(1 for _ in f)
                nr_chunks_per_doc[get_filename_from_path(doc_path)] = nr_chunks
        print("Number of Chunks per Document:", nr_chunks_per_doc)

        # Calculate total number of chunks
        total_nr_chunks = sum(nr_chunks_per_doc.values())
        print('Total nr chunks:', total_nr_chunks)

        # Chunk count distribution
        chunk_count_freq = Counter(nr_chunks_per_doc.values())
        print("Chunk count freq:", chunk_count_freq)

        min_chunks_doc = min(nr_chunks_per_doc, key=nr_chunks_per_doc.get)
        max_chunks_doc = max(nr_chunks_per_doc, key=nr_chunks_per_doc.get)
        print("Document with minimum chunks:", min_chunks_doc, "Chunks:", nr_chunks_per_doc[min_chunks_doc])
        print("Document with maximum chunks:", max_chunks_doc, "Chunks:", nr_chunks_per_doc[max_chunks_doc])

        title = (
            f"Total nr chunks: {total_nr_chunks}\n"
            f"Document with minimum chunks: {min_chunks_doc}, Chunks: {nr_chunks_per_doc[min_chunks_doc]}\n"
            f"Document with maximum chunks: {max_chunks_doc}, Chunks: {nr_chunks_per_doc[max_chunks_doc]}"
        )

        self.plot_chunks_per_doc(chunk_count_freq, title)


    def plot_chunks_per_doc(self, chunk_count_freq, title):
        #chunk_count_freq = dict(sorted(list(chunk_count_freq.items()))[:10])
        plt.figure(figsize=(20, 6))
        
        # Calculate the bar width to avoid overlapping
        num_categories = len(chunk_count_freq)
        max_xrange = max(chunk_count_freq.keys()) - min(chunk_count_freq.keys())
        bar_width = max_xrange / (num_categories * 2)  # Adjust the divisor as needed
        

        plt.bar(chunk_count_freq.keys(), chunk_count_freq.values(), color='red', width=bar_width)
        plt.xlabel('Chunk Count')
        plt.ylabel('Frequency')
        plt.title('Chunk Frequency Distribution')
        plt.xticks(rotation=45)

        # Set y-axis ticks to integers
        max_freq = max(chunk_count_freq.values())
        print('max freq', max_freq)
        plt.yticks(range(max_freq + 1))
        # plt.tight_layout()
        # plt.show()
        # Set the title string into a box at the top right corner of the plot
        plt.text(0.95, 0.95, title, transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
                va='top', ha='right')

        self.save_data(data=plt, file_name='chunks-per-doc')
        plt.close()

    def count_tokens_per_chunk(self):
        textchunker = TextChunker(language=self.language, tokens_per_chunk=self.tokens_per_chunk)

        counts = {}
        for doc_path in self.doc_paths:
            chunks = textchunker.load_data(file_name=get_filename_from_path(doc_path), doc_path=doc_path) # load chunks as a list of chunks
            chunks = ' '.join(chunks)
            l = len(chunks.split())
            counts[get_filename_from_path(doc_path)] = l

        # self.save_data(data=plt, file_name='tokens-per-chunk')
        self.plot_tokens_per_chunk(counts)


    def plot_tokens_per_chunk(self, counts):
        plt.figure(figsize=(8, 6))
        plt.bar(counts.keys(), counts.values())
        plt.xlabel('Text')
        plt.ylabel('Tokens per chunk')
        plt.title('Tokens per chunk')
        plt.xticks(rotation=45)
        self.save_data(data=plt, file_name='tokens-per-chunk')
        plt.close()
 