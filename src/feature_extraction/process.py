import os
import re
import logging
logging.basicConfig(level=logging.DEBUG)
from tqdm import tqdm
import pickle
import spacy
import sys
sys.path.append("..")
from utils import load_list_of_lines, save_list_of_lines
import logging
logging.basicConfig(level=logging.DEBUG)
import time
import string
from unidecode import unidecode
import regex as re


def load_spacy_model(language):
    if language == 'eng':
        model_name = 'en_core_web_sm'
    elif language == 'ger':
        model_name = 'de_core_news_sm'
    else:
        raise Exception(f'Not a valid language {language}')

    try:
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        print(f'The model {model_name} for Spacy is missing.')
    

class Tokenizer():
    def __init__(self, language):
        self.language = language
        self.nlp = load_spacy_model(self.language)

    def get_tokenized_sentences(self, doc_path):
        start = time.time()

        def tokenize_sentences(doc_path):
            logging.info('Tokenizing sentences.')
            with open(doc_path, 'r') as reader:
                text = reader.read().strip()
            
            text = Preprocessor(doc_path).preprocess_text(text)

            i = 0
            all_sentences = []
            while True:
                current_text = text[i:i+500000]
                # Keep text_with_ws to count number of characters in sents
                current_sentences = [(sent.text, sent.text_with_ws) for sent in self.nlp(current_text).sents]
                if len(current_sentences) == 1:
                    all_sentences.extend([sents[0].strip() for sents in current_sentences])
                    break
                else:
                    all_sentences.extend([sents[0].strip() for sents in current_sentences[:-1]])
                    i += len(''.join([sents[1] for sents in current_sentences[:-1]]))
            all_sentences = [sent for sent in all_sentences if len(sent) > 3]
            return all_sentences

        tokenized_sentences_path = doc_path.replace('/raw_docs', '/tokenized_sentences')
        if os.path.exists(tokenized_sentences_path):
            sentences = load_list_of_lines(tokenized_sentences_path, 'str')
        else:
            sentences = tokenize_sentences(doc_path)
            # Write to file so that each sentence is on a new line
            save_list_of_lines(sentences, tokenized_sentences_path, 'str')
        print('Time to tokenize sentences:  ', time.time()-start)
        return sentences
    

    def get_tokenized_words(self, doc_path):

        def tokenize_words(doc_path):
            '''
            Tokenize sentences so that each sentence is a list of words.
            '''
            def tokenize_words_helper(text):
                # Remove word if it consists only of punctuation mark
                # "'" is ignored if it is part of a word (Mary's -> Mary + 's)
                punctuation = set(string.punctuation + '’' + '‘' + '—' + '“' + '”' + '–')
                text = [word for word in text if not set(word) <= punctuation]
                text = ' '.join(text)
                # text = re.sub('[^a-zA-ZäöüÄÖÜ\']+', ' ', text).strip() #####################
                # text = text.split() ##############3
                # text = ' '.join(text) #####################
                text = text.lower()
                Preprocessor(doc_path).check_characters(text) #####################################
                return text
            
            all_words = []
            sentences = self.get_tokenized_sentences(doc_path)[10:30] #########################
            logging.info('Tokenizing words.')
            start = time.time()
            for sentence in sentences: 
                # Use spacy tokenizer
                words = [token.text for token in self.nlp(sentence)]
                words = tokenize_words_helper(words)
                all_words.append(words)
            print(f'Time for tokenizing 1 doc: {time.time()-start}')
            assert len(sentences) == len(all_words)
            return all_words
        
        tokenized_words_path = doc_path.replace('/raw_docs', '/tokenized_words')     
        if os.path.exists(tokenized_words_path):
            words = load_list_of_lines(tokenized_words_path, 'str')
        else:
            words = tokenize_words(doc_path)
            save_list_of_lines(words, tokenized_words_path, 'str')
        return words
    

class Preprocessor():
    # Preprocess text before sentence tokenization          
    def __init__(self, doc_path):
        self.doc_path = doc_path

    def check_annotations(self, text):
        annotation_words_eng = [
            'Annotation',
            'Footnote',
            'Footer',
            'Note',
            'Comment',
            'Addendum',
            'Remark',
            'Supplement',
            'Appendix',
            'Clarification',
            'Corrigendum',
            ]
        annotation_words_german = [
            'Anmerkung',
            'Kommentar',
            'Annotation',
            'Fußnote',
            'Fussnote'
            'Ergänzung',
            'Nebenbemerkung',
            'Bemerkung',
            'Erklärung',
            'Vertiefung',
            'Anhang',
            'Referenz',
            'Erweiterung',
            ]
        word_list = annotation_words_eng + annotation_words_german

        lowercase_text = text.lower()
        lowercase_words = [word.lower() for word in word_list]

        with open('annotation_words.txt', 'a') as f:
            for word in lowercase_words:
                regex_string = r'\b' + word + r'\b'
                idx = re.search(regex_string, text)
                if bool(idx):
                    idx = idx.start()
                    step = 50
                    if idx<step:
                        idx = 0
                    if idx > (len(text) - step):
                        idx = (len(text) - step)
                    f.write(self.doc_path + '\t' + word + '\t' + text[idx-step:idx+step].replace('\n', '') + '\n')


    def check_characters(self, text):
            # inner "'" is escaped
            text = re.sub('[\'A-Za-z0-9ÄÖÜäöü,?!-;()\[\] ]+', '', text)
            # Remove "=" between numbers
            chars = re.sub(r'\d[ ]?=[ ]?\d', '', text)
            if chars:
                with open('char_check.txt', 'a') as f:
                    f.write(self.doc_path + '\t' + chars + '\n')

        # def check_uppercase(text):
        #     upper_words = [word for word in text if (word.isupper() and len(word)>1)]
        #     if upper_words:
        #         with open('upper_check.txt', 'a') as f:
        #             f.write(self.doc_path + ',' + ' '.join(upper_words) + '\n')

    def replace_utf_tags(self, text):
        '''
        Replace UTF tags with the format [...]<[...]>[...]
        '''
        utf_replace = re.compile(r"""
            (^|\s)  # Start at whitespace (not at word boundary because \b matches only alphanumeric chars but not <)
            \S*     # Match non-whitespace up to the last UTF tag (greedy)
            (<.*?>)   # Match UTF tag (non-greedy, only up to next >)
            \S*?     # Match trailing chars up to next whitespace or dot
            ((?=\s)|(?=\.))
        """, flags = re.VERBOSE | re.MULTILINE)

        text = utf_replace.sub('', text)
        return text
        
    def replace_multiple(self, text, rep_dict):
        # https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string
        rep = dict((re.escape(k), v) for k, v in rep_dict.items()) 
        pattern = re.compile("|".join(rep.keys()))
        text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
        return text
    
    def get_rep_dict(self):
        rep_dict = {
            '\r\n': ' ',
            '\n': ' ',
            '\t': ' ',
            'ß': 'ss', 
            '’s': "'s",             # 'o’clock'-> "o'clock"
            'dr.': 'dr',            # Trollope_Anthony_Barchester-Towers_1857.txt
            'Dr.': 'Dr', 
            'Mr.': 'Mr', 
            'mr.': 'mr', 
            'Mrs.': 'Mrs', 
            'mrs.': 'mrs',
            'feb.': 'feb',          # Owenson_Sydney_The-Wild-Irish-Girl_1806
            'st.': 'st',            # Doyle_Arthur-Conan_The-Man-with-the-Twisted-Lip_1891
            'St.': 'St',      
            # '&c.': 'xxxxxetc',    # Sterne_Laurence_Tristram-Shandy_1759
            '--': ' ',              # Shelley_Mary_Perkin-Warbeck_1830.txt
            "'-": '',               # Carleton_William_Fardorougha-the-Miser_1839, Carleton_William_The-Emigrants-of-Ahadarra_1847
            '»': "'",
            '«': "'",
            # ',—': ' ',              # Corelli_Marie_The-Sorrows-of-Satan_1895.txt
            # '"—': ' ',              # Grand_Sarah_The-Heavenly-Twins_1893
            # '—"': ' ',              # Grand_Sarah_The-Heavenly-Twins_1893
            # '\'—"': ' ',              # Grand_Sarah_The-Heavenly-Twins_1893
            # '—"\'': ' ',              # Grand_Sarah_The-Heavenly-Twins_1893
            "scoundrel?'": 'scoundrel ', # Collins_Wilkie_Armadale_1864
            'I was;3': 'I was; 3',   # Dickens_Charles_The-Pickwick-Papers_1836
            'Kukuanaland.1': 'Kukuanaland.', # Haggard_H-Rider_King-Solomons-Mines_1885.txt
            'P.V.P.M.P.C.,1': 'P.V.P.M.P.C.', # Dickens_Charles_The-Pickwick-Papers_1836
            'G.C.M.P.C.,2': 'G.C.M.P.C.', 
            ' /': '', # Lohenstein_Daniel_Arminius_1689
            '<': "'", # Keller_Gottfried_Die-Leute-von-Seldwyla_1856.txt
            '>': "'", 
            '=>': '', # Sterne_Laurence_Tristram-Shandy_1759
            ' align="left">': '', # Hays_Mary_Memoirs-of-Emma-Courtney_1796
        }
        return rep_dict

    def preprocess_text(self, text):
        docs_with_line = ['Anonymous_Anonymous_Vertue-Rewarded_1693', 
                        'Anonymous_Anonymous_The-Adventures-of-Anthony-Varnish_1786', 
                        'Anonymous_Anonymous_The-Triumph-Prudence-Over-Passion_1781',
                        'Chaigneau_William_The-History-of-Jack-Connor_1752']
        if any(file_name in self.doc_path for file_name in docs_with_line):
            text = text.replace('|', '')

        if 'Ebers_George_Eine-aegyptische-Koenigstocher_1864' in self.doc_path:
            # Some notes are marked with '..
            # With DOTALL, newlines inside pattern are also replaced!
            text = re.sub(r'[ ]?(\(Anm\. \d+\)).*?\.\.', '.', text, flags=re.DOTALL)
            # Remove everything until the end of the line. This is probably too much.
            # Annotations are not formatted consistently and cannot be properly removed with regex.
            text = re.sub(r'[ ]?(\(Anm\. \d+\)).*?\n', '.\n', text)

        if 'Grand_Sarah_The-Heavenly-Twins_1893' in self.doc_path:
            text = text.replace('[she wrote]', 'she wrote')
            # Remove footnotes and descriptions of illustrations.
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)

        if 'OGrady_Standish_The-Coming-of-Cuculain_1894' in self.doc_path:
            text = re.sub(r'Footnote: .*?\n', '', text)

        if 'Barrie_J-M_Peter-and-Wendy_1911' in self.doc_path:
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)

        if 'Haggard_H-Rider_Allan-Quartermain_1887' in self.doc_path:
            # Remove endnotes at the end of the text
            idx = text.rindex('Endnote 1\n')
            text = text[:idx]
            # Remove endnotes inside the text
            text = re.sub(r'Endnote \d*?(,| )', '', text)

        if 'Fielding_Henry_Amelia_1752' in self.doc_path:
            text = text.replace('{Containing the exordium, &c. ', '')
            text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)

        if 'Scott_Walter_The-Abbot' in self.doc_path:
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
            # Footnotes don't have a clear format.
            # Remove everything up to the end of the line.
            # This is probably too much.
            text = re.sub(r'Footnote.*?\n', '\n', text)

        self.check_annotations(text)

        edgeworth_list = ['Edgeworth_Maria_The-Contrast_1804.txt', 
                          'Edgeworth_Maria_Murad-the-Unlucky_1804',
                          'Edgeworth_Maria_Rosanna_1804',
                          'Edgeworth_Maria_The-Will_1804',
                          'Edgeworth_Maria_Lame-Jervas_1804',
                          'Edgeworth_Maria_The-Lottery_1804',
                          'Edgeworth_Maria_The-Manufacturers_1804']
        if any(file_name in self.doc_path for file_name in edgeworth_list):
            text = re.sub(r' \{Footnote.*?\}', '', text, flags=re.DOTALL)
        
        edgeworth_list = ['Edgeworth_Maria_Ormond_1817', 
                          'Edgeworth_Maria_Helen_1834',
                          'Edgeworth_Maria_Patronage_1814']
        if any(file_name in self.doc_path for file_name in edgeworth_list):
            text = re.sub(r'\[Footnote.*?\]', '', text, flags=re.DOTALL)

        if 'Edgeworth_Maria_Orlandino_1848' in self.doc_path:
            text = re.sub(r'\[\d\]', '', text)
        
        edgeworth_list = ['Edgeworth_Maria_The-Grateful-Negro_1804',
                          'Edgeworth_Maria_To-Morrow_1804',
                          'Edgeworth_Maria_The-Limerick-Gloves_1804']
        if any(file_name in self.doc_path for file_name in edgeworth_list):
            text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
        #if in self.doc_path:

        edgeworth_list = ['Edgeworth_Maria_The-Irish-Incognito_1802',
                         'Edgeworth_Maria_Castle-Rackrent_1800',
                         'Edgeworth_Maria_The-Modern-Griselda_1804']
        if any(file_name in self.doc_path for file_name in edgeworth_list):
            text = re.sub(r' \[.*?\]', '', text, flags=re.DOTALL)

        # Replace all "'d" at the end of a word (train'd -> trained)
        text = re.sub(r"(?:\w)\'d\b", 'ed',text)
        # Remove commas inside numbers
        text = re.sub(r'(?<=\d+)[,\'\.](?=\d+)', '', text)
        # Replace UTF tags
        text = self.replace_utf_tags(text)
        # Replace initials
        # text = re.sub('(.\.){2,}', 'xxxxxinitials', text)

        # Replace spelling variations, remove accents but keep umlauts
        rep_dict = self.get_rep_dict()
        umlaut_dict = {
            'ä': 'xxxxxae', 
            'ö': 'xxxxxoe', 
            'ü': 'xxxxxue', 
            'Ä': 'xxxxxbigae', 
            'Ö': 'xxxxxbigoe', 
            'Ü': 'xxxxxbigue'}
        rep_dict.update(umlaut_dict)
        umlaut_dict_swap = {v: k for k, v in umlaut_dict.items()}
        text = self.replace_multiple(text, rep_dict)
        text = unidecode(text)
        text = self.replace_multiple(text, umlaut_dict_swap)
        text = text.split()
        text = ' '.join(text)
        self.check_characters(text)
        return text
        #if in self.doc_path:



# class Doc2VecProcessor():
#     def __init__(self, language, processed_chunk_sentence_count=500, stride=500):
#         self.language = lang
#         self.sentence_tokenizer = SentenceTokenizer(self.lang)
#         self.processed_chunk_sentence_count = processed_chunk_sentence_count
#         self.stride = stride

#     def process(self, doc_paths):
#         logging.info('Processing texts...')
#         if self.processed_chunk_sentence_count is not None:
#             os.makedirs('/'.join(doc_paths[0].split('/')[:-1]).replace('/raw_docs', f'/processed_docs_sc_{self.processed_chunk_sentence_count}_st_{self.stride}'), exist_ok=True)
#         else:
#             os.makedirs('/'.join(doc_paths[0].split('/')[:-1]).replace('/raw_docs', f'/processed_doc2vec_full'), exist_ok=True)

#         for doc_path in tqdm(doc_paths):
#             with open(doc_path, 'r') as doc_reader:
#                 doc = doc_reader.read()

#             if self.processed_chunk_sentence_count is not None:
#                 if os.path.exists(doc_path[:-4].replace('/raw_docs', f'/tokenized_sentences') + '.pickle'):
#                     sentences = pickle.load(open(doc_path[:-4].replace('/raw_docs', f'/tokenized_sentences') + '.pickle', 'rb'))
#                 else:
#                     sentences = self.sentence_tokenizer.tokenize(doc)
#                     pickle.dump(sentences, open(doc_path[:-4].replace('/raw_docs', f'/tokenized_sentences') + '.pickle', 'wb'))
#                 sentences = [tokenize_sentences_helper(sentence) for sentence in sentences]

#                 for i in range(0, len(doc), self.stride):
#                     current_chunk = sentences[i:i+self.processed_chunk_sentence_count]
#                     if (len(current_chunk) < self.processed_chunk_sentence_count) and i != 0:
#                         break
#                     processed_doc_path = doc_path[:-4].replace('/raw_docs', f'/processed_doc2vec_sc_{self.processed_chunk_sentence_count}_st_{self.stride}') +  f'_pt_{i}.txt'
#                     with open(processed_doc_path, 'w') as doc_writer:
#                         doc_writer.write(' '.join(current_chunk))
#             else:
#                 doc = tokenize_sentences_helper(doc)
#                 processed_doc_path = doc_path.replace('/raw_docs', '/processed_doc2vec_full')
#                 with open(processed_doc_path, 'w') as doc_writer:
#                     doc_writer.write(doc)
#         logging.info('Processed texts.')


# class BertProcessor():
#     def __init__(self, lang, pad):
#         self.lang = lang
#         self.pad = pad
#         if self.lang == 'eng':
#             self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         elif self.lang == 'ger':
#             self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
#         else:
#             raise Exception(f'Not a valid language {self.lang}')
#         self.sentence_tokenizer = SentenceTokenizer(self.lang)

#     def process(self, doc_paths):
#         logging.info('Processing texts...')
#         os.makedirs('/'.join(doc_paths[0].split('/')[:-1]).replace('/raw_docs', f'/processed_bert_512_tokens'), exist_ok=True)
#         os.makedirs('/'.join(doc_paths[0].split('/')[:-1]).replace('/raw_docs', f'/processed_bert_sentence_tokens'), exist_ok=True)
#         os.makedirs('/'.join(doc_paths[0].split('/')[:-1]).replace('/raw_docs', f'/tokenized_sentences'), exist_ok=True)

#         for doc_path in tqdm(doc_paths):
#             with open(doc_path, 'r') as doc_reader:
#                 doc = doc_reader.read()
#             if os.path.exists(doc_path[:-4].replace('/raw_docs', f'/tokenized_sentences') + '.pickle'):
#                 sentences = pickle.load(open(doc_path[:-4].replace('/raw_docs', f'/tokenized_sentences') + '.pickle', 'rb'))
#             else:
#                 sentences = self.sentence_tokenizer.tokenize(doc)
#                 pickle.dump(sentences, open(doc_path[:-4].replace('/raw_docs', f'/tokenized_sentences') + '.pickle', 'wb'))

#             current_paragraph = ''
#             current_token_count = 0
#             tokenized_paragraphs = []
#             tokenized_sentences = []
#             for current_sentence in sentences:
#                 tokenized_sentence = self.bert_tokenizer(current_sentence, return_tensors='pt')
#                 if tokenized_sentence['input_ids'].shape[1] > 512:
#                     continue
#                 current_tokenized_length = tokenized_sentence['input_ids'].shape[1] - 2
#                 if self.pad:
#                     tokenized_sentence = self.bert_tokenizer(current_sentence, return_tensors='pt', padding=True, truncation=True)
#                 tokenized_sentences.append(tokenized_sentence)
#                 if current_token_count + current_tokenized_length <= 510:
#                     current_token_count += current_tokenized_length
#                     if current_paragraph == '':
#                         current_paragraph = current_sentence
#                     else:
#                         current_paragraph += ' ' + current_sentence
#                 else:
#                     if self.pad:
#                         tokenized_paragraph = self.bert_tokenizer(current_paragraph, return_tensors='pt', padding=True, truncation=True)
#                     else:
#                         tokenized_paragraph = self.bert_tokenizer(current_paragraph, return_tensors='pt', padding=False, truncation=False)
#                     tokenized_paragraphs.append(tokenized_paragraph)
#                     current_paragraph = current_sentence
#                     current_token_count = current_tokenized_length
#             for tokenized_paragraph in tokenized_paragraphs:
#                 if tokenized_paragraph['input_ids'].shape[1] > 512:
#                     print('Long paragraph detected:', doc_path.split('/')[-1])
#                     break
            
#             for tokenized_sentence in tokenized_sentences:
#                 if tokenized_sentence['input_ids'].shape[1] > 512:
#                     print('Long sentence detected:', doc_path.split('/')[-1])
#                     break
            
#             if self.pad:
#                 pickle_path = doc_path[:-4].replace('/raw_docs', f'/processed_bert_512_tokens_padded') + '.pickle'
#             else:
#                 pickle_path = doc_path[:-4].replace('/raw_docs', f'/processed_bert_512_tokens_not_padded') + '.pickle'
#             pickle.dump(tokenized_paragraphs, open(pickle_path, 'wb'))
            
#             if self.pad:
#                 pickle_path = doc_path[:-4].replace('/raw_docs', f'/processed_bert_sentence_tokens_padded') + '.pickle'
#             else:
#                 pickle_path = doc_path[:-4].replace('/raw_docs', f'/processed_bert_sentence_tokens_not_padded') + '.pickle'
#             pickle.dump(tokenized_sentences, open(pickle_path, 'wb'))
            
#         logging.info('Processed texts.')
