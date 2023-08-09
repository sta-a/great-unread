# %%
import os
import logging
import spacy
import time
import string
import sys
sys.path.append("..")
logging.basicConfig(level=logging.DEBUG)
import sys
from utils import DataHandler
from utils import DataHandler, check_equal_line_count, check_equal_files,  get_filename_from_path
from check import DataChecker

class SentenceTokenizer():
    def __init__(self, language):
        self.language = language
        if self.language == 'eng':
            self.model_name = 'en_core_web_sm'
        else:
            self.model_name = 'de_core_news_sm'
        self.nlp = spacy.load(self.model_name)

    def create_data(self, doc_path=None, text=None):
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

        print(f'Time for tokenization: {time.time()-start}')
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
        with open('tests_tokenization.txt', "w") as file:
            for item1, item2 in zip(all_sentences, new_all_sentences):
                file.write(item1 + '\n')
                file.write(item2 + '\n')


class TextChunker(DataHandler):
    def __init__(self, language, tokens_per_chunk):
        super().__init__(language=language, output_dir='text_chunks', data_type='txt', tokens_per_chunk=tokens_per_chunk)
        self.tokens_per_chunk = tokens_per_chunk
        self.threshold = 0.2
        self.tpc_increase = 5
        self.sentence_tokenizer = SentenceTokenizer(self.language)

    def _distribute_sents_to_chunks(self, tpc):
        chunks = []
        current_chunk = []
        token_count = 0
        for sentence in self.sentences:
            sentence_tokens = sentence.split()
            
            if token_count  < tpc:
                current_chunk.extend(sentence_tokens)
                token_count += len(sentence_tokens)
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = sentence_tokens
                token_count = len(sentence_tokens)
        return chunks, current_chunk
    
    def create_data(self, doc_path):
        self.sentences = self.sentence_tokenizer.create_data(doc_path)

        chunks, current_chunk = self._distribute_sents_to_chunks(self.tokens_per_chunk)
        print('current_chunk', len(current_chunk), 'chunks', len(chunks))
        with open('chunktest', 'w') as f:
            f.write('\n----------------------\n'.join(chunks) + '\n')
            f.write('current chunks' + ' '.join(current_chunk))

        # Deal with remaining tokens at the end of the texts that are shorter thant tokens_per_chunk
        tpc = self.tokens_per_chunk
        while len(current_chunk) > self.threshold*self.tokens_per_chunk:
            tpc += self.tpc_increase
            chunks, current_chunk = self._distribute_sents_to_chunks(tpc)
            
        # If they are below threshold long, add them to previous chunk
        assert len(current_chunk) <= self.threshold*self.tokens_per_chunk, 'Nr of tokens at the end of the text that belong to no chunk are above threshold.'
        assert all(len(string.split()) >= self.tokens_per_chunk for string in chunks), "Some strings don't meet the token count requirement."
        print('current_chunk', len(current_chunk), 'chunks', len(chunks))
        chunks[-1] += ' '.join(current_chunk)

        with open('chunktest', 'a') as f:
            f.write('\n\n' + '\n----------------------\n'.join(chunks) + '\n')
            f.write('current chunks' + ' '.join(current_chunk))

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
        # Split full text into chunks

        start = time.time()
        new_chunks = []
        for chunk in chunks:
            # chunk = [token.text for token in self.nlp(chunk)] ##########
            # chunk = ' '.join(chunk)
            new_chunks.append(chunk)
        print(f'Time for spacy tokenization: {time.time()-start}')
        return new_chunks
    

    def create_data(self, doc_path, remove_punct=False, lower=False, as_chunk=True):
        chunks = self.textchunker.load_data(file_name=get_filename_from_path(doc_path), doc_path=doc_path) # load chunks as a list of chunks
        # with open(doc_path, 'r') as reader: # Helper for testing ################################
        #     text = reader.read().strip()
        # pp = Preprocessor(self.language, doc_path)
        # chunks = pp.preprocess_text(text)

        logging.info(f'Tokenizing {doc_path}')
        chunks = self.tokenize_words(chunks)
        self.save_data(file_name=get_filename_from_path(doc_path), data=chunks)

        chunks = Postprocessor(remove_punct=remove_punct, lower=lower).postprocess_chunks(chunks)

        if as_chunk == False:
            chunks = ' '.join(chunks)
        #     self.logger.info('Returning tokenized words as one string.')
        # else:
        #     self.logger.info('Returning tokenized chunks as list of strings.')
        return chunks
    
    def create_all_data(self):
        start = time.time()
        for i, doc_path in enumerate(self.doc_paths):
            _ = self.load_data(load=False, file_name=get_filename_from_path(doc_path), doc_path=doc_path)
        print(f'{time.time()-start}s to tokenize all texts')
    
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
  