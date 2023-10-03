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
import matplotlib.patches as mpatches


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

    
    def tokenize_sentences(self, text):
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
        self.logger.debug(f'Tokenizing sentences: {doc_path}')

        with open(doc_path, 'r') as reader:
            text = reader.read().strip()
        # Preprocess full text
        pp = Preprocessor(self.language, doc_path, self.tokens_per_chunk)
        text = pp.preprocess_text(text)
        
        # Tokenize
        sentences = self.tokenize_sentences(text)

        self.save_data(data=sentences, file_name=get_filename_from_path(doc_path))
        # self.check_data(sentences)
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

        
    def tokenize_words(self, sentences):
        self.logger.debug(f'Tokenizing words.')
        start = time.time()
        new_sentences = []
        for doc in self.nlp.pipe(sentences, batch_size=1, disable=["tagger", "parser", "ner"], n_process=multiprocessing.cpu_count()-2):
            sent = [token.text for token in doc] 
            # Add line breaks to split sentences after : and ;
            # This is simpler than modifying the sentence tokenizer
            # Tokenized dir and sentences dir have different number of lines
            if len(sent) == 1:
                print('Sent only 1 long', ' '.join(sent))
            if len(sent) > 1:
                newsent = [x+'\n' if x in [':', ';', '...'] else x+' ' for x in sent[:-1]]
                newsent.append(sent[-1])
            sent = ''.join(newsent)
            new_sentences.append(sent)
        print(f'Time for word tokenization: {time.time()-start}')
        return new_sentences

    def create_data(self, doc_path=None):
        self.logger.debug(f'Tokenizing sentences: {doc_path}')

        st = SentenceTokenizer(self.language)
        sentences = st.load_data(file_name=get_filename_from_path(doc_path), doc_path=doc_path)
        sentences = self.tokenize_words(sentences)

        self.save_data(data=sentences, file_name=get_filename_from_path(doc_path))
        return sentences
    

    def create_all_data(self):
        startx = time.time()
        for doc_path in self.doc_paths:
            _ = self.load_data(load=False, file_name=get_filename_from_path(doc_path), doc_path=doc_path)
        print(f'{time.time()-startx}s to tokenize all texts')



class ChunkHandler(DataHandler):
    def __init__(self, language, tokens_per_chunk):
        super().__init__(language=language, output_dir=f'text_chunks_tpc_{tokens_per_chunk}', data_type='txt', tokens_per_chunk=tokens_per_chunk)
        self.tokens_per_chunk = tokens_per_chunk
        self.tolerance = 0.1 # Maximum allowed deviation from tokens per chunk
        self.t = Tokenizer(self.language)

    def distribute_sents_to_chunks(self, sentences, limit):
        
        def distribute(sentences, limit, n_inc=0, iteration=0):
            # Not implemented: save intermediate chunks per iteration to make it more efficient
            chunks = []
            current_chunk = []
            cc_len = 0
            chunk_count = 0
            for sentence in sentences:
                # Increase the length of the first few chunks
                if n_inc != 0 and chunk_count < n_inc:
                    current_tpc = self.tokens_per_chunk + iteration
                else:
                    current_tpc = self.tokens_per_chunk
                tokens = sentence.split()
                assert len(tokens) < current_tpc, sentence

                # Chunks can become longer than tpc
                # Round up or down to nearer sentence boundary
                if (cc_len < current_tpc) or (cc_len + len(tokens) - current_tpc <= current_tpc - cc_len):
                    current_chunk.append(sentence)
                    cc_len += len(tokens)
                else:
                    chunks.append(current_chunk)
                    chunk_count += 1
                    current_chunk = [sentence]
                    cc_len = len(tokens)
            if current_chunk:
                # Make new chunk, which can be a bit shorter
                if cc_len > (self.tokens_per_chunk - limit):
                    chunks.append(current_chunk)
                    current_chunk = None
                    cc_len = None
                # Append to last chunk
                elif cc_len <= limit:
                    chunks[-1].extend(current_chunk)
                    current_chunk = None
                    cc_len = None    
            return chunks, current_chunk, cc_len
        
        
        limit = self.tokens_per_chunk * self.tolerance
        first_chunks, first_current_chunk, first_cc_len = distribute(sentences, limit, n_inc=0)

        if first_current_chunk is None:
            chunks = first_chunks
        else:
            iteration = 1
            chunks = first_chunks
            current_chunk = first_current_chunk
            while current_chunk is not None and iteration<=50: #  n_inc<=4
                print(f'iteration: {iteration}')
                n_inc = 1
                while current_chunk is not None and len(chunks) > n_inc: #  n_inc<=4
                    # Redistribute sents to chunks
                    # The first n chunks have an increased length
                    chunks, current_chunk, _ = distribute(sentences, limit, n_inc=n_inc, iteration=iteration)
                    if current_chunk is None:
                        print('Exiting loop with no chunks left.')
                        return chunks
                    else:
                        n_inc += 1
                iteration+=1
            else:
                print('splitting last chunk')
                # Make short chunk
                if first_cc_len >= self.tokens_per_chunk//2:
                    first_chunks.append(first_current_chunk)
                else:
                    # Extend last chunk, make long chunk
                    first_chunks[-1].extend(first_current_chunk)
                chunks = first_chunks
                
        return chunks

    
    def create_data(self, doc_path=None):
        bookname = get_filename_from_path(doc_path)
        self.logger.debug(f'\n----------------------------\nCreating chunks. {bookname}')
        sentences = self.t.load_data(file_name=bookname, doc_path=doc_path)
        chunks = self.distribute_sents_to_chunks(sentences, doc_path)

        shortest = float('inf')  # Initialize with a large value
        longest = 0
        for chunk in chunks:
            chunk_length = len(' '.join(chunk).split())
            shortest = min(shortest, chunk_length)
            longest = max(longest, chunk_length)
        
        print(bookname, shortest, longest, '\n-------------------\n')
        self.save_data(data=chunks, file_name=bookname)

    def create_all_data(self):
        starty = time.time()
        for i, doc_path in enumerate(self.doc_paths):
            _ = self.load_data(load=False, file_name=get_filename_from_path(doc_path), doc_path=doc_path)
        print(f'{time.time()-starty}s to split all texts into chunks.')

    def load_data(self, file_name=None, load=True, remove_punct=False, lower=False, as_chunk=True, as_sent=False, **kwargs):
        file_path = self.get_file_path(file_name=file_name, **kwargs)
        self.file_exists_or_create(file_path=file_path, **kwargs)

        chunks = None
        if load:
            self.logger.debug(f'Loading {file_path} from file.')
            # List of strings, every string is a chunk
            chunks = self.load_data_type(file_path, **kwargs)
            chunks = Postprocessor(remove_punct=remove_punct, lower=lower).postprocess_chunks(chunks)

            if not as_chunk:
                chunks = ' '.join(chunks)
                chunks = [chunks]

            if as_sent:
                chunks = [string.split(self.separator) for string in chunks]
            else:
                chunks = [string.replace(self.separator, ' ') for string in chunks]

            if as_chunk and as_sent:
                self.logger.info(f'Chunks as nested lists. Inner lists are chunk, strings inside are sentences.')
            elif not as_chunk and as_sent:
                self.logger.info(f'Chunks as nested lists. Single inner lists represents only chunk, strings inside are sentences.')
            elif as_chunk and not as_sent:
                self.logger.info('Chunks as list of strings.')
            else:
                self.logger.info('Chunks as one list of one string.')

        return chunks
    
    def check_data(self):
        dc = self.DataChecker(self.language, chunks_dir=self.output_dir)
        dc.check_completeness()
        token_counts_chunk = dc.compare_token_counts()
        nr_chunks_per_doc, total_nr_chunks = dc.count_chunks_per_doc()
        dc.check_chunks_for_shortests_texts(token_counts_chunk, nr_chunks_per_doc)
        dc.count_tokens_per_chunk(nr_chunks_per_doc)
        sentences_per_chunk, sentences_per_doc = dc.count_sentences_per_chunk()
    

    class DataChecker(DataHandler):
        '''
        Class for checking chunking
        '''
        def __init__(self, language, chunks_dir):
            super().__init__(language, output_dir='text_statistics', data_type='svg')
            self.chunks_dir = chunks_dir
            self.chunk_paths = get_files_in_dir(self.chunks_dir)
            self.ch = ChunkHandler(language=self.language, tokens_per_chunk=self.tokens_per_chunk)
            # Check if chunks have been created for all files
            assert check_equal_files(self.text_raw_dir, self.ch.output_dir)


        def check_completeness(self):
            assert len(os.listdir(self.ch.output_dir)) == self.nr_texts


        def compare_token_counts(self):
            '''
            Compare token counts in tokenized words files and chunk files to assert that no tokens got lost during chunking.
            '''
            def count_words_in_file(file_path):
                with open(file_path, 'r') as file:
                    text = file.read()
                    text = text.replace('ƒ', ' ')
                    text = text.split()
                    return len(text)
    
            t = Tokenizer(self.language)
            tok_files = os.listdir(t.output_dir)

            token_counts_chunk = {}
            for file_name in tok_files:
                tok_path = os.path.join(t.output_dir, file_name)
                chunk_path = os.path.join(self.ch.output_dir, file_name)

                if os.path.isfile(chunk_path):
                    tok_count = count_words_in_file(tok_path)
                    chunk_count = count_words_in_file(chunk_path)
                    assert tok_count == chunk_count, f"Word count is different for file {file_name}, {tok_count}, {chunk_count}"
                    token_counts_chunk[get_filename_from_path(chunk_path)] = chunk_count
            return token_counts_chunk
        

        def check_chunks_for_shortests_texts(self, token_counts_chunk, nr_chunks_per_doc):
            token_counts_chunk = dict(sorted(token_counts_chunk.items(), key=lambda item: item[1])[:10])
            chunks_and_tokens = {}
            for filename, value in token_counts_chunk.items():
                chunks_and_tokens[filename] = [nr_chunks_per_doc[filename], value,]
            
            print('---------------------------------')
            print('Nr. chunks in shortest texts:')
            for filename, values in chunks_and_tokens.items():
                print(f'{filename}: Chunks: {values[0]}, Tokens: {values[1]}.')


        def count_sentences_per_chunk(self):
            sentences_per_chunk = {}
            sentences_per_doc = {}

            for chunk_path in self.chunk_paths:
                total_count = 0
                with open(chunk_path, 'r') as f:
                    file_dict = {}
                    for index, line in enumerate(f):
                        count = line.count(self.separator) + 1 # if there are x separators, there x+1 sentences
                        file_dict[index] = count
                        total_count += count
                    sentences_per_chunk[get_filename_from_path(chunk_path)] = file_dict
                    sentences_per_doc[get_filename_from_path(chunk_path)] = total_count
            return sentences_per_chunk, sentences_per_doc


        def count_chunks_per_doc(self):
            # Count chunks
            nr_chunks_per_doc = {}

            # Nr chunks per doc
            for chunk_path in self.chunk_paths:
                with open(chunk_path, 'r') as f:
                    nr_chunks = sum(1 for _ in f)
                    nr_chunks_per_doc[get_filename_from_path(chunk_path)] = nr_chunks

            # Save to file
            df = pd.DataFrame(list(nr_chunks_per_doc.items()), columns=['file_name', 'nr_chunks'])
            self.save_data(data=df, file_name='chunks_per_doc.csv', )

            # Calculate total number of chunks
            total_nr_chunks = sum(nr_chunks_per_doc.values())

            # Chunk count distribution
            chunk_count_freq = Counter(nr_chunks_per_doc.values())

            min_chunks_doc = min(nr_chunks_per_doc, key=nr_chunks_per_doc.get)
            max_chunks_doc = max(nr_chunks_per_doc, key=nr_chunks_per_doc.get)

            title = (
                f"Total nr chunks: {total_nr_chunks}\n"
                f"Document with minimum chunks: {min_chunks_doc}, Chunks: {nr_chunks_per_doc[min_chunks_doc]}\n"
                f"Document with maximum chunks: {max_chunks_doc}, Chunks: {nr_chunks_per_doc[max_chunks_doc]}"
            )
            self.plot_chunks_per_doc(chunk_count_freq, title)
            return nr_chunks_per_doc, total_nr_chunks


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


        def count_tokens_per_chunk(self, nr_chunks_per_doc):
            ch = ChunkHandler(language=self.language, tokens_per_chunk=self.tokens_per_chunk)

            counts = {}
            lengths = [] 
            for chunk_path in self.chunk_paths:
                chunks = ch.load_data(file_name=get_filename_from_path(chunk_path))  # load chunks as a list of chunks

                chunk_lengths = [len(chunk.split()) for chunk in chunks]
                shortest = min(chunk_lengths)
                longest = max(chunk_lengths)
                average = np.mean(chunk_lengths)
                stddev = np.std(chunk_lengths)
                                
                counts[get_filename_from_path(chunk_path)] = (shortest, longest, average, stddev)

            self.plot_tokens_per_chunk(counts, nr_chunks_per_doc)


        def plot_tokens_per_chunk(self, counts, nr_chunks_per_doc):
                    sorted_counts = sorted(counts.items(), key=lambda x: x[1][0])  # Sort by shortest length

                    file_names = [fn for fn, _ in sorted_counts]
                    shortest_lengths = [shortest for _, (shortest, _, _, _) in sorted_counts]
                    longest_lengths = [longest for _, (_, longest, _, _) in sorted_counts]
                    average_lengths = [average for _, (_, _, average, _) in sorted_counts]
                    stddev_lengths = [stddev for _, (_, _, _, stddev) in sorted_counts]

                    assert len(file_names) == len(shortest_lengths) == len(longest_lengths) == len(average_lengths) == len(stddev_lengths) == self.nr_texts

                    bar_width = 0.3
                    indices = np.arange(len(file_names))


                    # Highlight texts with very few chunks
                    short_texts = [1, 2, 3] # consider texts with 1-3 chunks
                    bar_colors = ['b' for _ in file_names]
                    for i, fn in enumerate(file_names):
                        if nr_chunks_per_doc[fn] in short_texts:
                            bar_colors[i] = 'g'  # Highlight in green for values 1, 2, and 3


                    plt.figure(figsize=(12, 6))
                    plt.bar(indices, shortest_lengths, bar_width, label='Shortest', color='r')
                    plt.bar(indices + bar_width, longest_lengths, bar_width, label='Longest', color=bar_colors)
                    # Create a dummy bar for legend entry
                    plt.bar([-1], [0], width=0, label='Fewest Chunks', color='g')

                    plt.xlabel('Text')
                    plt.ylabel('Tokens per chunk')
                    plt.title('Tokens per chunk (Shortest and Longest)')

                    # legend = plt.legend()
                    # # Find the legend item for 'Fewest Chunks' and change its color to green
                    # for handle in legend.legendHandles:
                    #     if handle.get_label() == 'Longest':
                    #         handle.set_color('b')
                    legend_patches = [
                        mpatches.Patch(color='r', label='Shortest'),
                        mpatches.Patch(color='b', label='Longest'),
                        mpatches.Patch(color='g', label='Fewest Chunks')
                    ]

                    # Display the legend with custom legend patches
                    plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.35, 1), facecolor='lightgrey')


                    ytick_positions = np.arange(0, max(longest_lengths) + 50, 100)
                    plt.yticks(ytick_positions)

                    tick_positions = np.arange(0, len(file_names), 50)
                    tick_labels = [str(pos) for pos in tick_positions]  # Convert positions to strings
                    plt.xticks(tick_positions, tick_labels)  # Display tick positions as numbers on x-axis with rotation
                    
                    plt.tight_layout()
                    self.save_data(data=plt, file_name='tokens-per-chunk-shortest-longest')


                    # Second plot
                    plt.figure(figsize=(12, 6))
                    plt.bar(indices, stddev_lengths, bar_width, label='Std. Deviation', color='r')
                    plt.bar(indices + bar_width, average_lengths, bar_width, label='Average', color=bar_colors)
                    # Create a dummy bar for legend entry
                    plt.bar([-1], [0], width=0, label='Fewest Chunks', color='g')

                    plt.xlabel('Text')
                    plt.ylabel('Tokens per chunk')
                    plt.title('Tokens per chunk (Average and Std. Deviation)')

                    legend_patches = [
                        mpatches.Patch(color='r', label='Std. Deviation'),
                        mpatches.Patch(color='b', label='Average'),
                        mpatches.Patch(color='g', label='Fewest Chunks')
                    ]

                    # Display the legend with custom legend patches
                    plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.35, 1), facecolor='lightgrey')


                    ytick_positions = np.arange(0, max(stddev_lengths + average_lengths) + 20, 20)
                    plt.yticks(ytick_positions)

                    xtick_positions = np.arange(0, len(file_names), 50)
                    xtick_labels = [str(pos) for pos in xtick_positions]  # Convert positions to strings
                    plt.xticks(xtick_positions, xtick_labels)  # Display tick positions as numbers on x-axis with rotation
                    
                    plt.tight_layout()
                    self.save_data(data=plt, file_name='tokens-per-chunk-average-stddev')


        # def plot_tokens_per_chunk(self, counts, nr_chunks_per_doc):
        #     sorted_counts = sorted(counts.items(), key=lambda x: x[1][0])  # Sort by shortest length

        #     texts = [text for text, _ in sorted_counts]
        #     shortest_lengths = [shortest for _, (shortest, _, _, _) in sorted_counts]
        #     longest_lengths = [longest for _, (_, longest, _, _) in sorted_counts]
        #     average_lengths = [average for _, (_, _, average, _) in sorted_counts]
        #     stddev_lengths = [stddev for _, (_, _, _, stddev) in sorted_counts]

        #     assert len(texts) == len(shortest_lengths) == len(longest_lengths) == len(average_lengths) == len(stddev_lengths) == self.nr_texts

        #     bar_width = 0.3
        #     indices = np.arange(len(texts))

        #     plt.figure(figsize=(12, 6))
        #     plt.bar(indices, shortest_lengths, bar_width, label='Shortest', color='r')
        #     plt.bar(indices + bar_width, longest_lengths, bar_width, label='Longest', color='b')

        #     plt.xlabel('Text')
        #     plt.ylabel('Tokens per chunk')
        #     plt.title('Tokens per chunk (Shortest and Longest)')
        #     plt.legend()

        #     ytick_positions = np.arange(0, max(longest_lengths) + 50, 100)
        #     plt.yticks(ytick_positions)

        #     tick_positions = np.arange(0, len(texts), 50)
        #     tick_labels = [str(pos) for pos in tick_positions]  # Convert positions to strings
        #     plt.xticks(tick_positions, tick_labels)  # Display tick positions as numbers on x-axis with rotation
            
        #     plt.tight_layout()
        #     self.save_data(data=plt, file_name='tokens-per-chunk-shortest-longest')


        #     # Second plot
        #     plt.figure(figsize=(12, 6))
        #     plt.bar(indices, stddev_lengths, bar_width, label='Std. Deviation', color='r')
        #     plt.bar(indices + bar_width, average_lengths, bar_width, label='Average', color='b')

        #     plt.xlabel('Text')
        #     plt.ylabel('Tokens per chunk')
        #     plt.title('Tokens per chunk (Average and Std. Deviation)')
        #     plt.legend()

        #     ytick_positions = np.arange(0, max(stddev_lengths + average_lengths) + 20, 20)
        #     plt.yticks(ytick_positions)

        #     xtick_positions = np.arange(0, len(texts), 50)
        #     xtick_labels = [str(pos) for pos in xtick_positions]  # Convert positions to strings
        #     plt.xticks(xtick_positions, xtick_labels)  # Display tick positions as numbers on x-axis with rotation
            
        #     plt.tight_layout()
        #     self.save_data(data=plt, file_name='tokens-per-chunk-average-stddev')


class Postprocessor():
    def __init__(self, remove_punct=False, lower=False):
        self.remove_punct = remove_punct
        self.lower = lower
        self.logger = logging.getLogger(__name__)

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
        self.logger.info(f'Returning postprocessed text as list of chunks.')
        return new_chunks
    


# %%
