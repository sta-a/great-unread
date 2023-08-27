# %%
import os
import spacy
import time
import string
import random
import numpy as np
import sys
import statistics
import logging
from collections import Counter
logging.basicConfig(level=logging.DEBUG)
import matplotlib
import logging
import pandas as pd
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
sys.path.append("..")
import multiprocessing
from utils import DataHandler, check_equal_line_count, check_equal_files,  get_filename_from_path, get_files_in_dir, get_doc_paths_sorted
from stats import TextStatistics
from .preprocessor import Preprocessor


class SentenceTokenizer(DataHandler):
    def __init__(self, language):
        super().__init__(language, 'text_sentences', data_type='txt')
        if self.language == 'eng':
            self.model_name = 'en_core_web_sm'
        else:
            self.model_name = 'de_core_news_sm'
        self.nlp = spacy.load(self.model_name, disable=["lemmatizer"]) # Suppress warning
        self.nlp.add_pipe('sentencizer')

        # self.doc_paths = get_doc_paths_sorted(self.text_raw_dir)[:5]
        # self.doc_paths = list(reversed(self.doc_paths))
        # print('doc paths sorted', self.doc_paths)

    def tokenize_sentences_old(self, text):
        start = time.time()
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

        # Postprocess
        # Some ' are considered to be a single sentence
        new_sentences = []
        prev = None  
        for sent in all_sentences: ##################3
            if sent == "'":
                prev = sent
                continue
            if prev == "'":
                new_sentences.append("'" + sent)
            else:
                new_sentences.append(sent)
            prev = sent

        print(f'Time for sentence old tokenization: {time.time()-start}')
        return new_sentences
    
    def tokenize_sentences_best(self, text):
        start = time.time()
        all_sentences = []
        i = 0
        while i < len(text):
            # Process in chunks because of Spacy character limit
            current_text = text[i:i+500000]
    

            for doc in self.nlp.pipe([current_text], batch_size=1, disable=["tagger", "parser", "ner"], n_process=multiprocessing.cpu_count()-2):
                current_sentences = [(sent.text, sent.text_with_ws) for sent in doc.sents]

                # Ensures that complete sentences are extracted 
                # Check if there's only one sentence in the current chunk
                if len(current_sentences) == 1:
                    # Add the single sentence to the list of all sentences
                    all_sentences.extend([sents[0].strip() for sents in current_sentences])
                    i = float('inf')
                    break  # Exit the loop since the current chunk is fully processed
                else:
                    # Add all sentences except the last one to the list of all sentences
                    all_sentences.extend([sents[0].strip() for sents in current_sentences[:-1]])
                    # Update the index 'i' to skip the characters corresponding to the stripped whitespace
                    i += len(''.join([sents[1] for sents in current_sentences[:-1]]))
                             
        # Postprocess
        new_sentences = []
        prev = False
        ending = " '"
        for sent in all_sentences:
            if prev == True:
                sent = "'" + sent
                prev = False
            if sent.endswith(ending):
                sent = sent[:-len(ending)]
                prev = True
            new_sentences.append(sent)

        print(f'Time for sentence tokenization: {time.time() - start}')
        return new_sentences
        

    def create_data(self, doc_path=None):
        self.logger.info(f'Tokenizing sentences: {doc_path}')

        with open(doc_path, 'r') as reader:
            text = reader.read().strip()
        # Preprocess full text
        pp = Preprocessor(self.language, doc_path)
        text = pp.preprocess_text(text)
        
        # Tokenize
        print(f'{get_filename_from_path(doc_path)}-----------------------------------------')
        sentences = self.tokenize_sentences_best(text)

        self.save_data(data=sentences, file_name=get_filename_from_path(doc_path)) ###########################
        self.check_data(sentences)
        return sentences
    
    def check_data(self, all_sentences):
        sentence_lengths = [len(sentence.split()) for sentence in all_sentences]
        min_length = min(sentence_lengths)
        max_length = max(sentence_lengths)
        print("Minimum sentence length:", min_length)
        print("Maximum sentence length:", max_length)


    def create_all_data(self):
        startc = time.time()
        for i, doc_path in enumerate(self.doc_paths):
            _ = self.load_data(load=False, file_name=get_filename_from_path(doc_path), doc_path=doc_path)
        print(f'{time.time()-startc}s to tokenize all texts')
    
    # def tests_tokenization(self, doc_path):
    #     '''
    #     Check if tokenization changes if the same text is tokenized multiple times.
    #     '''
    #     start = time.time()
    #     all_sentences = self.create_data(doc_path=doc_path)
    #     text = ' '.join(all_sentences)
    #     new_all_sentences = self.create_data(text=text)
    #     if not all_sentences == new_all_sentences:
    #         print(len(all_sentences), len(new_all_sentences))
    #         print('Sentence tokenization is not always the same.')
    #     print(f'Time for calculating tokenization test: {time.time()-start}.')
    #     with open('test_sentence_tokenization.txt', "w") as file:
    #         for item1, item2 in zip(all_sentences, new_all_sentences):
    #             file.write(item1 + '\n')
    #             file.write(item2 + '\n')


class Tokenizer(DataHandler):
    def __init__(self, language):
        super().__init__(language, 'text_tokenized', data_type='txt')
        if self.language == 'eng':
            self.model_name = 'en_core_web_sm'
        else:
            self.model_name = 'de_core_news_sm'
        self.nlp = spacy.load(self.model_name, disable=["lemmatizer"]) # Suppress warning

        
    def tokenize_words_old(self, sentences):
        self.logger.info(f'Tokenizing words.')
        start = time.time()
        new_sentences = []
        for sent in sentences:
            sent = [token.text for token in self.nlp(sent)] 
            # Add line breaks to split sentences after : and ;
            # This is simpler than modifying the sentence tokenizer
            if len(sent) == 1:
                print('Sent only 1 long', ' '.join(sent))
            if len(sent) > 1:
                newsent = [x+'\n' if x in [':', ';'] else x+' ' for x in sent[:-1]]
                newsent.append(sent[-1])
            sent = ''.join(newsent)
            new_sentences.append(sent)
        print(f'Time for old word tokenization: {time.time()-start}')
        return new_sentences

        
    def tokenize_words(self, sentences):
        self.logger.info(f'Tokenizing words.')
        start = time.time()
        new_sentences = []
        for doc in self.nlp.pipe(sentences, batch_size=1, disable=["tagger", "parser", "ner"], n_process=multiprocessing.cpu_count()-2):
            sent = [token.text for token in doc] 
            # Add line breaks to split sentences after : and ;
            # This is simpler than modifying the sentence tokenizer
            if len(sent) == 1:
                print('Sent only 1 long', ' '.join(sent))
            if len(sent) > 1:
                newsent = [x+'\n' if x in [':', ';'] else x+' ' for x in sent[:-1]]
                newsent.append(sent[-1])
            sent = ''.join(newsent)
            new_sentences.append(sent)
        print(f'Time for word tokenization: {time.time()-start}')
        return new_sentences

    def create_data(self, doc_path=None):
        self.logger.info(f'Tokenizing sentences: {doc_path}')

        st = SentenceTokenizer(self.language)
        sentences = st.load_data(file_name=get_filename_from_path(doc_path), doc_path=doc_path)
        # sentences = self.tokenize_words_old(sentences)
        sentences = self.tokenize_words(sentences)

        self.save_data(data=sentences, file_name=get_filename_from_path(doc_path))
        self.check_data(sentences)
        return sentences
    
    def check_data(self, all_sentences):
        # assert check_equal_line_count(self.output_dir, self.chunkhandler.output_dir, self.data_type) ########################
        # assert check_equal_files(self.output_dir.replace('/text_tokenized', '/text_raw'), self.output_dir) ####################3
        pass

    def create_all_data(self):
        startx = time.time()
        for i, doc_path in enumerate(self.doc_paths):
            _ = self.load_data(load=False, file_name=get_filename_from_path(doc_path), doc_path=doc_path)
        print(f'{time.time()-startx}s to tokenize all texts')


class TextLoader(DataHandler):
    def __init__(self, language, tokens_per_chunk):
        super().__init__(language=language, output_dir='text_tokenized', data_type='txt', tokens_per_chunk=tokens_per_chunk)


    def load_data(self, doc_path, remove_punct=True, lower=True, as_chunk=True, **kwargs):
        print('doc path', doc_path)
        if as_chunk == False:
            print(get_filename_from_path(doc_path))
            path = os.path.join(self.output_dir, get_filename_from_path(doc_path) + '.txt')
            with open(path, 'r') as f:
                text = f.read()
            print(text)
            text = Postprocessor(remove_punct=remove_punct, lower=lower).postprocess_text(text)
            self.logger.info('Returning tokenized words as a single list of one string.')
            return text
        else:
            ch = ChunkHandler(self.language, self.tokens_per_chunk)
            chunks = ch.load_data(file_name=get_filename_from_path(doc_path), doc_path=doc_path, load=True, remove_punct=remove_punct, lower=lower, **kwargs)
            self.logger.info('Returning tokenized chunks as list of strings.')
            return chunks


class ChunkHandler(DataHandler):
    def __init__(self, language, tokens_per_chunk):
        super().__init__(language=language, output_dir='text_chunks', data_type='txt', tokens_per_chunk=tokens_per_chunk)
        self.tokens_per_chunk = tokens_per_chunk
        self.tolerance = 0.1 # Maximum allowed deviation from tokens per chunk
        self.t = Tokenizer(self.language)
        # self.shortest_texts = ['Altenberg_Peter_Wie-wunderbar_1914', 'Hebel_Johann-Peter_Kannitverstan_1808', 'Wildermuth_Ottilie_Streit-in-der-Liebe-und-Liebe-im-Streit_1910', 'Kleist_Heinrich_Das-Bettelweib-von-Locarno_1810', 'Kleist_Heinrich_Unwahrscheinlich-Wahrhaftigkeiten_1811', 'Wackenroder_Wilhelm_Morgenlaendisches-Maerchen_1799', 'Rilke_Rainer-Maria_Die-Turnstunde_1899', 'Sacher-Masoch_Leopold_Lola_1907', 'Rilke_Rainer-Maria_Die-Weise-von-Liebe-und-Tod_1904', 'Moerike_Eduard_Die-Hand-der-Jezerte_1853']
        # self.doc_paths = [os.path.join(self.text_raw_dir, x + '.txt') for x in self.shortest_texts]
        # self.doc_paths = get_doc_paths_sorted(self.text_raw_dir)[:5] ######################


    # def distribute_sents_to_chunks(self, sentences, limit):
        
    #     def distribute(sentences, limit, fits=None):
    #         chunks = []
    #         current_chunk = []
    #         tokens_count = 0
    #         chunk_count = 0
    #         for sentence in sentences:
    #             # Make first couple of chunks longer if function is called for the second time with fits
    #             if fits is not None and chunk_count < fits:
    #                 current_tpc = self.tokens_per_chunk + limit
    #             else:
    #                 current_tpc = self.tokens_per_chunk
    #             tokens = sentence.split()
    #             assert len(tokens) < current_tpc, sentence

    #             # Chunks can become longer than tpc
    #             if tokens_count < current_tpc:
    #                 current_chunk.append(sentence)
    #                 tokens_count += len(tokens)
    #             else:
    #                 chunks.append(current_chunk)
    #                 chunk_count += 1
    #                 current_chunk = [sentence]
    #                 tokens_count = len(tokens)
    #         if current_chunk:
    #             # Make new chunk, which can be a bit shorter
    #             if tokens_count > (self.tokens_per_chunk - limit):
    #                 chunks.append(current_chunk)
    #                 current_chunk = None
    #             # Append to last chunk
    #             elif tokens_count <= limit:
    #                 chunks[-1].extend(current_chunk)
    #                 current_chunk = None
                    
    #         return chunks, current_chunk, tokens_count
        
    #     def calculate_fits(number, limit):
    #         assert 1 <= number <= self.tokens_per_chunk
    #         n = number // limit
    #         return n
        
    #     limit = self.tokens_per_chunk * self.tolerance
    #     chunks, current_chunk, tokens_count = distribute(sentences, limit, fits=None)

    #     if current_chunk is not None:
    #         n = calculate_fits(tokens_count, limit)
    #         if len(chunks) >= n:
    #             # Redistribute sents to chunks
    #             # The first n chunks have an increased length
    #             # The remaining tokens are short enough to be added to the last chunk
    #             # Example: len(current_chunk) = 320 -> n=6 -> 6 first chunks have increased lenght, remaining 20 tokens are added at the end
    #             #### This must not necessarily work   
    #             chunks, current_chunk, tokens_count = distribute(sentences, limit, fits=n)
    #             assert current_chunk is None, f' {tokens_count}, {current_chunk}'
    #         else:
    #             # Make short chunk
    #             if tokens_count >= self.tokens_per_chunk//2:
    #                 chunks.append(current_chunk)
    #             else:
    #                 # Extend last chunk, make long chunk
    #                 chunks[-1].extend(current_chunk)
    #     return chunks

    def distribute_sents_to_chunks(self, sentences, limit):
        
        chunks = []
        current_chunk = []
        tokens_count = 0
        chunk_count = 0
        for sentence in sentences:
            tokens = sentence.split()
            # Chunks can become longer than tpc
            if tokens_count < self.tokens_per_chunk:
                current_chunk.append(sentence)
                tokens_count += len(tokens)
            else:
                chunks.append(current_chunk)
                chunk_count += 1
                current_chunk = [sentence]
                tokens_count = len(tokens)

        if current_chunk:
            # Make short chunk
            if tokens_count >= self.tokens_per_chunk//2:
                chunks.append(current_chunk)
            else:
                # Extend last chunk, make long chunk
                chunks[-1].extend(current_chunk)

        return chunks

    
    def create_data(self, doc_path=None):
        self.logger.info(f'Creating chunks.')
        sentences = self.t.load_data(file_name=get_filename_from_path(doc_path), doc_path=doc_path)
        chunks = self.distribute_sents_to_chunks(sentences, doc_path)
        self.save_data(data=chunks, file_name=get_filename_from_path(doc_path))
        self.logger.info(f'Finished splitting text into chunks.')

    def check_data(self):
        dc = self.DataChecker(self.language, chunks_dir=self.output_dir)
        dc.count_chunks_per_doc()
        # dc.plot_tokens_in_excluded_text()
        dc.count_tokens_per_chunk()

    def create_all_data(self):
        starty = time.time()
        for i, doc_path in enumerate(self.doc_paths):
            _ = self.load_data(load=False, file_name=get_filename_from_path(doc_path), doc_path=doc_path)
        self.check_data()
        print(f'{time.time()-starty}s to tokenize all texts')

    def load_data(self, load=True, file_name=None, remove_punct=False, lower=False, as_chunk=True, **kwargs):
        file_path = self.get_file_path(file_name=file_name, **kwargs)
        self.file_exists_or_create(file_path=file_path, **kwargs)

        chunks = None
        if load:
            if self.print_logs:
                self.logger.info(f'{self.__class__.__name__}: Loading {file_path} from file.')
            # List of strings, every string is a chunk
            chunks = self.load_data_type(file_path, **kwargs)
            chunks = [chunk.replace(self.separator, ' ') for chunk in chunks]
            chunks = Postprocessor(remove_punct=remove_punct, lower=lower).postprocess_chunks(chunks)

        return chunks


    class DataChecker(DataHandler):
        '''
        Class for checking chunking
        '''
        def __init__(self, language, chunks_dir):
            super().__init__(language, output_dir='text_statistics', data_type='png')
            self.chunks_dir = chunks_dir
            self.chunk_paths = get_files_in_dir(self.chunks_dir)

        def count_chunks_per_doc(self):
            # Count chunks
            nr_chunks_per_doc = {}

            # Nr chunks per doc
            for chunk_path in self.chunk_paths:
                with open(chunk_path, 'r') as f:
                    nr_chunks = sum(1 for _ in f)
                    nr_chunks_per_doc[get_filename_from_path(chunk_path)] = nr_chunks
            print("Number of Chunks per Document:")
            for k, v in nr_chunks_per_doc.items():
                print(k, v)

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
            plt.yticks(range(max_freq + 1))
            # plt.tight_layout()
            # plt.show()
            # Set the title string into a box at the top right corner of the plot
            plt.text(0.95, 0.95, title, transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
                    va='top', ha='right')

            self.save_data(data=plt, file_name='chunks-per-doc')
            plt.close()

        def plot_tokens_in_excluded_text(self):
            '''
            Plot length of the last part of the text that was not added to a chunk.
            '''
            path = os.path.join(self.output_dir, 'omitted_text.txt')
            try:
                df = pd.read_csv(path, sep=self.separator, names=['file_name', 'nr_tokens', 'next_chunk'])

                all_filenames = [get_filename_from_path(path) for path in os.listdir(self.text_raw_dir)]
                rows_to_add = []
                for fn in all_filenames:
                    if fn not in df['file_name'].values:
                        new_row = {'file_name': fn, 'nr_tokens': 0, 'next_chunk': None}
                        rows_to_add.append(new_row)
                new_rows_df = pd.DataFrame(rows_to_add)
                df = pd.concat([df, new_rows_df], ignore_index=True)
                
                df = df.sort_values(by='nr_tokens')

                plt.figure(figsize=(20, 6))
                plt.bar(df['file_name'], df['nr_tokens'])
                plt.xlabel('File name')
                plt.ylabel('Nr. tokens')
                plt.xticks(df.index, [])
                plt.title('Nr tokens in text not added to a chunk.')
                plt.xticks(rotation=45)
                self.save_data(data=plt, file_name='tokens-in-omitted-text')
                plt.close()
            except Exception as e:
                print(e)

        def count_tokens_per_chunk(self):
            ch = ChunkHandler(language=self.language, tokens_per_chunk=self.tokens_per_chunk)

            counts = {}
            for chunk_path in self.chunk_paths:
                chunks = ch.load_data(file_name=get_filename_from_path(chunk_path)) # load chunks as a list of chunks

                shortest = float('inf')  # Initialize with a large value
                longest = 0
                for chunk in chunks:
                    chunk_length = len(chunk.split())
                    shortest = min(shortest, chunk_length)
                    longest = max(longest, chunk_length)
                
                counts[get_filename_from_path(chunk_path)] = (shortest, longest)
            self.plot_tokens_per_chunk(counts)


        def plot_tokens_per_chunk(self, counts):
            sorted_counts = sorted(counts.items(), key=lambda x: x[1][0])  # Sort by shortest length

            texts = [text for text, _ in sorted_counts]
            shortest_lengths = [shortest for _, (shortest, _) in sorted_counts]
            longest_lengths = [longest for _, (_, longest) in sorted_counts]

            bar_width = 0.35
            indices = np.arange(len(texts))

            plt.figure(figsize=(10, 6))
            plt.bar(indices, shortest_lengths, bar_width, label='Shortest', color='r')
            plt.bar(indices + bar_width, longest_lengths, bar_width, label='Longest', color='b')

            plt.xlabel('Text')
            plt.ylabel('Tokens per chunk')
            plt.title('Tokens per chunk (Shortest vs. Longest)')
            # plt.xticks(indices, texts, rotation=45)
            plt.xticks([])  # Remove x-axis ticks
            plt.legend()

            plt.tight_layout()
            self.save_data(data=plt, file_name='tokens-per-chunk')




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
    


# %%
