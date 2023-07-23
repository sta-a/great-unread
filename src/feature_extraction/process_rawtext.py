# %%
import os
import re
import logging
import spacy
import time
import string
from unidecode import unidecode
import regex as re
import sys
sys.path.append("..")
from utils import load_list_of_lines, save_list_of_lines, get_bookname
logging.basicConfig(level=logging.DEBUG)
import sys
from utils import DataHandler
from pathlib import Path
import pickle
from collections import Counter
import os
import heapq
sys.path.insert(1, '/home/annina/scripts/pydelta')
import delta
sys.path.append("..")
from utils import DataHandler


class Tokenizer():
    def __init__(self, language=None, doc_paths=None, tokens_per_chunk=500):
        self.language = language
        self.doc_paths = doc_paths
        self.tokens_per_chunk = tokens_per_chunk
        self.nlp = self.load_spacy_model()
        self.logger = logging.getLogger(__name__)
        self.sentence_chars = ['.', ':', ';', '?', '!', '...']

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
    

    def split_into_chunks(self, text):
        '''
        Split text into chunks with length around tokens_per_chunk.
        A chunk has tokens_per_chunk tokens and is then extended until the next sentence terminating character is found.
        '''
        all_chunks = []
        text_split = text.split()
        i_start = 0
        i_end = 0
        while i_end < len(text_split) - self.tokens_per_chunk - 1:
            i_end = i_start + self.tokens_per_chunk
            while i_end < len(text_split):
                element = text_split[i_end]
                if any([x in element for x in self.sentence_chars]):
                    chunk = text_split[i_start:i_end+1]
                    all_chunks.append(chunk)
                    i_start = i_end + 1
                    break
                else:
                    i_end += 1
            
        all_chunks = [' '.join(chunk) for chunk in all_chunks]
        return all_chunks


    def tokenize_words(self, doc_path):
        with open(doc_path, 'r') as reader:
            text = reader.read().strip()

        # Preprocess full text before tokenizing
        # Single chunks cannot be preprocessed because content in brackets is split if there are any sentence terminating characters inside the brackets.
        pp = Preprocessor(self.language, doc_path, self.sentence_chars)
        text = pp.preprocess_text(text)
        
        # Split full text into chunks
        all_chunks = self.split_into_chunks(text)

        logging.info('Tokenizing words.')
        start = time.time()
        new_chunks = []
        for chunk in all_chunks:
            # chunk = [token.text for token in self.nlp(chunk)] ##########
            # chunk = ' '.join(chunk)
            new_chunks.append(chunk)
        print(f'Time for spacy tokenization: {time.time()-start}')
        return new_chunks
    

    def get_tokenized_words(self, doc_path, remove_punct=False, lower=False, as_chunk=True):

        tokenized_words_path = doc_path.replace('/raw_docs', '/tokenized_words')     
        if os.path.exists(tokenized_words_path):
            all_chunks = load_list_of_lines(tokenized_words_path, 'str')
        else:
            all_chunks = self.tokenize_words(doc_path)
            logging.info(f'Tokenizing {doc_path}')
            save_list_of_lines(all_chunks, tokenized_words_path, 'str')

        all_chunks = Postprocessor(remove_punct=remove_punct, lower=lower).postprocess_chunks(all_chunks)

        if as_chunk == False:
            all_chunks = ' '.join(all_chunks)
            # self.logger.info('Returning tokenized words as one string.')
        # else:
        #     self.logger.info('Returning tokenized words as list of chunks.')
        return all_chunks
    
    def tokenize_all_texts(self):
        start = time.time()
        for i, doc_path in enumerate(self.doc_paths):
            _ = self.get_tokenized_words(doc_path, remove_punct=False, lower=False)
        print(f'{time.time()-start}s to tokenize all texts')
    

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

    def postprocess_chunks(self, all_chunks):
        new_chunks = []
        for chunk in all_chunks:
            new_chunk = self.postprocess_text(chunk)
            new_chunks.append(new_chunk)
        return new_chunks
    

class Preprocessor():
    # Preprocess text before sentence tokenization          
    def __init__(self, language, doc_path, sentence_chars):
        self.language = language
        self.doc_path = doc_path
        self.sentence_chars = sentence_chars
        self.greek_tag = ''
        self.latin_tag = ''
        self.bookname = get_bookname(self.doc_path)

    def check_annotations(self, text):
        annotation_words_eng = [
            'Addendum',
            'Annotation',
            'Appendix',
            'cf.',   # confer
            'Corrigendum',
            'e.g.',  # exempli gratia
            'endnote',###########################
            'Footnote',
            'Footer',
            'i.e.',  # id est
            # 'Note',
            '[see',
            '(see',
            # 'Supplement',
            # 'viz.',   # videlicet,
            'annotator',############
            'editor',################
            # 'greek',
            # 'french',
            # 'latin',
            # 'italian',
            # 'spanish',
            # 'german'
            ] # comment, remark
        annotation_words_ger = [
            'Anm.', ##################
            'Anmerkung',
            'Annotation',
            'Ergänzung',
            'Fussnote'
            # 'Fußnote',
            # 'Kommentar',
            'Nebenbemerkung',
            'Referenz',
            's. a.',
            's.a.', # siehe auch
            '(siehe',
            '[siehe',
            'siehe auch',
            'vergleiche auch', 
            'vgl.',
            'Herausgeber',###############
            'Herausg.',
            'Hrsg.',
            # 'griechisch',
            # 'französisch'
            # 'latein',
            # 'italienisch',
            # 'spanisch',
            # 'englisch'
            ]  # vergleiche
            # v.', 'vergleiche', siehe, s., Erklärung, Bemerkung, Anhang

        if self.language == 'eng':
            word_list = annotation_words_eng
        else:
            word_list = annotation_words_ger

        word_list = [word for word in word_list] #r'\b' + 
        word_list = [word.replace('.', r'\.') for word in word_list]
        word_list = [word.replace('[', r'\[') for word in word_list]
        word_list = [word.replace('(', r'\(') for word in word_list]

        text = text.lower()
        lowercase_words = [word.lower() for word in word_list]

        with open(f'annotation_words_{self.language}.txt', 'a') as f:
            for word in lowercase_words:
                idx = re.search(word, text)
                if bool(idx):
                    idx = idx.start()
                    step = 30
                    if idx<step:
                        idx = 0
                    if idx > (len(text) - step):
                        idx = (len(text) - step)
                    f.write(self.bookname + '\t' + word + '\t' + text[idx-3*step:idx+(step*4)].replace('\n', ' ').replace('"', '') + '\n')


    def check_characters(self, text):

        def get_char_idxs(char, text):
            return [i for i, letter in enumerate(text) if letter == char]

        allowed_chars = r'[\'A-Za-z0-9ÄÖÜäöü,?!-;_— ]+' # "'" is escaped
        chars = re.sub(allowed_chars, '', text)
        # Remove "=" between numbers
        chars = re.sub(r'\d[ ]?=[ ]?\d', '', chars)
        if chars:
            step = 25
            with open(f'special_chars_{self.language}.txt', 'a') as f:
                # Write part of text where char occurrs first to file
                # Might
                for char in set(chars):
                    idxs = get_char_idxs(char, text)
                    for idx in idxs:
                        if idx<step:
                            idx = 0
                        if idx > (len(text) - step):
                            idx = (len(text) - step)
                        f.write(self.bookname + '\t' + char + '\t' + text[idx-3*step:idx+4*step].replace('\n', ' ') + '\n')

    def replace_utf_tags(self, text):
        '''
        Replace UTF tags with the format [...]<[...]>[...]
        '''
        utf_replace = re.compile(r"""
            (^|\s)  # Start at beginning of line or at whitespace (not at word boundary because \b matches only alphanumeric chars but not <)
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
            u'\xa0': u' ',
            '\r\n': ' ',
            '\n': ' ',
            '\t': ' ',
            'ß': 'ss', 
            'å': 'ä',              # Lohenstein_Daniel_Arminius_1689
            'ů': 'ü',              # Lohenstein_Daniel_Arminius_1689
            'æ': 'ä',              # Gessner_Salomon_Daphnis_1854
            'œ': 'ö',              # Gessner_Salomon_Daphnis_1854
            '’': "'",               # 'o’clock'-> "o'clock"
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
            '‘': "'",
            '”': "'",
            '“': "'",
            '`': "'",
            ',—': ' ',              # Corelli_Marie_The-Sorrows-of-Satan_1895.txt
            "scoundrel?'": 'scoundrel ', # Collins_Wilkie_Armadale_1864
            'I was;3': 'I was; 3',   # Dickens_Charles_The-Pickwick-Papers_1836
            'P.V.P.M.P.C.,1': 'P.V.P.M.P.C.', # Dickens_Charles_The-Pickwick-Papers_1836
            'G.C.M.P.C.,2': 'G.C.M.P.C.', 
            '†': '',
            '<': "'", # Keller_Gottfried_Die-Leute-von-Seldwyla_1856.txt
            '>': "'", 
            '›': "'",
            '‹': "'",
            '=>': '', # Sterne_Laurence_Tristram-Shandy_1759
            ' align="left">': '', # Hays_Mary_Memoirs-of-Emma-Courtney_1796
            '[sic]': '',
            '(sic)': '',
            "{sic}": '', #Edgeworth_Maria_The-Manufacturers_1804
            '[*]': '',
            '[+]': '',
            '[#]': '', # Blackmore_R-D_Clara-Vaughan_1864
            '[**]': '', # Fielding_Henry_Tom-Jones_1749
            '[A]': '',
            '[1]': '',
        }
        return rep_dict

    def preprocess_individual_files(self, text):
        files_list = ['Anonymous_Anonymous_Vertue-Rewarded_1693', 
                        'Anonymous_Anonymous_The-Adventures-of-Anthony-Varnish_1786', 
                        'Anonymous_Anonymous_The-Triumph-Prudence-Over-Passion_1781',
                        'Chaigneau_William_The-History-of-Jack-Connor_1752']
        if any(file_name in self.doc_path for file_name in files_list):
            text = text.replace('|', '')
            text = text.replace('¦', '')

        if 'Blackmore_R-D_Clara-Vaughan_1864' in self.doc_path:
            long_string = """"[Greek: Kynòs ómmat' echôn, kradíen d' eláphoio.]"--C.V. 1864."""
            text = text.replace(long_string, '')

        if 'Haggard_H-Rider_King-Solomons-Mines_1885' in self.doc_path:            
            text = text.replace(' – Editor.', '')
            text = text.replace('Kukuanaland.1', 'Kukuanaland.')

        if '' in self.doc_path:
            text = text.replace('{little dogs, puppies, whelps,}', '')
            text = text.replace('[Greek text]', self.greek_tag)

        if 'Morrison_Arthur_A-Child-of-the-Jago_1896' in self.doc_path:
            text = text.replace('(which = watch stealer)', '')

        if 'La-Roche_Sophie_Fraeulein-von-Sternheim_1771' in self.doc_path:
            text = text.replace('[', '')

        if 'Lohenstein_Daniel_Arminius_1689' in self.doc_path:
            text = text.replace('[', '')

        if 'Fielding_Henry_Joseph-Andrews_1742' in self.doc_path:
            text = text.replace(' Footnote 5: ', '')

        if 'Scott_Walter_The-Heart-of-Midlothian_1818' in self.doc_path:
            text = text.replace('â€˜the', 'the')

        if 'Alexis_Willibald_Schloss-Avalon_1826' in self.doc_path:
            text = text.replace(':|:', '')

        files_list = ['Kingsley_Charles_Westward-Ho_1855']
        if any(file_name in self.doc_path for file_name in files_list):
            text = text.replace('[Greek text]', self.greek_tag)

        files_list = ['Reade_Charles_It-Is-Never-Too-Late-to-Mend_1856']
        if any(file_name in self.doc_path for file_name in files_list):
            text = text.replace('[Greek letters]', self.greek_tag)

        files_list = ['Blackmore_R-D_Lorna-Doone_1869']
        if any(file_name in self.doc_path for file_name in files_list):
            text = text.replace('[Greek word]', self.greek_tag)

        # if 'Hardy_Thomas_The-Woodlanders_1886' in self.doc_path:
        #     text = re.sub(r'{Greek word:\nirony}', '', text, flags=re.DOTALL)

        files_list = ['Dickens_Charles_Little-Dorrit_1855',
                      'Scott_Walter_The-Antiquary_1816',
                      'Hardy_Thomas_Far-from-the-Madding-Crowd_1874',
                      'Maurier_George-du_Trilby_1894',
                      'Carleton_William_The-Geography-of-an-Irish-Oath_1830',
                      'Richardson_Samuel_Clarissa_1748',
                      'Porter_Jane_The-Scottish-Chiefs_1810',
                      'Sims_George_Rogues-and-Vagabonds_1892',
                      'Mundt_Theodor_Moderne-Lebenswirren-Briefe-und-Zeitabenteuer-eines-Salzschreibers_1834',
                      'Auerbach_Berthold_Die-Frau-Professorin_1846',
                      ]
        if any(file_name in self.doc_path for file_name in files_list):
            text = text.replace('=', '')

        if 'Hughes_Thomas_Tom-Browns-Schooldays_1857' in self.doc_path:
            text = text.replace('[Greek text]', self.greek_tag)
            text = text.replace('[greek text deleted]', self.greek_tag)

        if 'Buchan_John_John-Burnet-of-Barns_1897' in self.doc_path:
            text = text.replace('[Greek: polypenthés]\nto [Greek: polypous]', self.greek_tag)
            text = text.replace('[Greek: polypenthes]\nto [Greek: polypous]', self.greek_tag)
            text = text.replace('[Greek: polypenthés]', self.greek_tag)
            text = text.replace('[Greek: polypenthes]', self.greek_tag)
            text = text.replace('[Greek: polypous]', self.greek_tag)

        if 'Eliot_George_Adam-Bede_1859' in self.doc_path:
            text = text.replace('[two greek words omitted]', self.greek_tag)

        if 'Scott_Walter_Redgauntlet_1824' in self.doc_path:
            long_string = '''
            [The original of this catch is to be found in Cowley's witty comedy of THE GUARDIAN, the first edition. It does not exist in the second and revised edition, called THE CUTTER OF COLEMAN STREET.
            CAPTAIN BLADE.  Ha, ha, boys, another catch. AND ALL OUR MEN ARE VERY VERY MERRY, AND ALL OUR MEN WERE DRINKING. CUTTER.    ONE MAN OF MINE. DOGREL.    TWO MEN OF MINE. BLADE.     THREE MEN OF MINE. CUTTER.    AND ONE MAN OF MINE. OMNES.     AS WE WENT BY THE WAY WE WERE DRUNK, DRUNK, DAMNABLY DRUNK, AND ALL OUR MEN WERE VERY VERY MERRY, &c.
            Such are the words, which are somewhat altered and amplified in the text. The play was acted in presence of Charles II, then Prince of Wales, in 1641. The catch in the text has been happily set to music.]
            '''
            text = text.replace(long_string, '')

        if 'Eliot_George_Janets-Repentance_1857' in self.doc_path:
            long_string = """
            [Greek: deinon to tiktein estin.]
            """
            text = text.replace(long_string, self.greek_tag)

        if 'Eliot_George_Sad-Fortunes-of-Rev-Amos-Barton_1857' in self.doc_path:
            long_string = """ [Greek: all edu gar toi ktema tes uikes labien tolma dikaioi d' authis ekphanoumetha.]"""
            text = text.replace(long_string, self.greek_tag)

        if 'Disraeli_Benjamin_Vivian-Grey_1826' in self.doc_path:
            rep_dict = {
                '[Greek: to kalon]': self.greek_tag,
                '[Greek: gnothi seauton]': self.greek_tag,
                '[Greek: hetairai]': self.greek_tag,
                '[Greek: euraeka]': self.greek_tag,
            }
            text = self.replace_multiple(text, rep_dict)

        if 'Unger_Friederike-Helene_Julchen-Gruenthal_1784' in self.doc_path:
            text = text.replace('@', 'li')

        if 'Hyan_Hans_1000-Mark-Belohnung_1913' in self.doc_path:
            pattern = r'\b\w+\s*=\s*\w+\b'
            text = re.sub(pattern, '', text)


        # if '' in self.doc_path:
        #     text = text.replace(r'Updater\'s note: the word "time" missing?]', '')
        #     rep_dict = {
        #         '[': '',
        #         ']': ''
        #     }
        #     text = self.replace_multiple(text, rep_dict)

        if 'Radcliffe_Ann_The-Romance-of-the-Forest_1791' in self.doc_path:
            text = text.replace('t\orpidity', 'torpidity')
            rep_dict = {
                '[': '',
                ']': ''
            }
            text = self.replace_multiple(text, rep_dict)

        if 'La-Roche_Sophie_Rosaliens-Briefe_1780' in self.doc_path:
            text = text.replace('[«] [Anschlußfehler in der Vorlage]' ,'')
            text = text.replace("[Anschlußfehler in der Vorlage, M.L.]" ,'')
            text = text.replace('@', '')
            text = text.replace('[»]', '')

        if 'Hoffmann_ETA_Berganza_1814' in self.doc_path:
            text = text.replace(". Anmerk. des Verlegers [C. F. Kunz]." ,'')
        
        if 'Eliot_George_Romola_1862' in self.doc_path:
            text = text.replace('(See note at the end.)','')

        if 'Nesbit_Edith_The-Wouldbegoods_1901' in self.doc_path:
            # text = text.replace('', '') #text = text.replace('', '')
            text = re.sub(r'Note .\. \(See Note .\.\)', '', text, flags=re.DOTALL)
            text = re.sub(r'\(See Note .\.\)', '', text, flags=re.DOTALL)

        if 'Mundt_Theodor_Madonna_1835' in self.doc_path:
            long_string = '(1801-1882), "Wlasta" (1829), großes böhmischnationales Heldengedicht. – Anm.d.Hrsg.'
            text = text.replace(long_string, '')

        if 'Stifter_Adalbert_Die-Narrenburg_1843' in self.doc_path:
            text = text.replace('*) [*)Alpenstock]' ,'')

        files_list = ['Fontane_Theodor_Mathilde-Moehring_1906',
                        'Reuter_Christian-Friedrich_Schelmuffsky_1696',
                        'Holcroft_Thomas_The-Adventures-of-Hugh-Trevor_1794',
                        'Sterne_Laurence_Tristram-Shandy_1759',
                        'Smollett_Tobias_Humphry-Clinker_1771',
                        'Richardson_Samuel_Clarissa_1748',
                        'More_Hannah_Coelebs_1814',
                        'Scott_Walter_St-Ronans-Well_1824',
                        'Burney_Frances_Cecilia_1782',
                        'Hughes_Thomas_Tom-Browns-Schooldays_1857',
                        'Burney_Frances_Camilla_1796',
                        'Frenssen_Gustav_Joern-Uhl_1901',
                        'Hoffmann_ETA_Der-Sandmann_1816',
                        'Wezel_Johann-Karl_Belphegor_1776',
                        'Wyss_Johann-David_Der-Schweizerische-Robinson_1812',
                        'Freytag_Gustav_Die-verlorene-Handschrift_1855',
                        'Grosse_Karl_Der-Genius_1791',
                        'Buechner_Georg_Lenz_1839',
                        'Motte-Fouque_Friedrich_Eine-Geschichte-vom-Galgenmaennlein_1810',
                        'Immermann_Karl_Die-Epigonen_1836',
                        'Hoffmann_ETA_Das-Steinerne-Herz_1816',
                        'Alexis_Willibald_Walladmor_1824',
                        'Hoffmann_ETA_Das-Sanctus_1816',
                        'Arnim_Achim_Hollins-Liebeleben_1802',
                        'Hoffmann_ETA_Das-Sanctus_1816',
                        'Arnim_Achim_Hollins-Liebeleben_1802',
                        'Grillparzer_Franz_Der-arme-Spielmann_1847',
                        'Hoffmann_ETA_Kater-Murr_1820',
                        'Ehrmann_Marianne_Amelie_1788',
                        'Hoffmann_ETA_Das-oede-Haus_1816',
                        'Hoffmann_ETA_Die-Serapions-Brueder_1819'
                        ]
        # In some of these texts, the brackets could be replaced automatically.
        # Finding errors is easier by removing them.
        if any(file_name in self.doc_path for file_name in files_list):
            rep_dict = {
                '[': '',
                ']': ''
            }
            text = self.replace_multiple(text, rep_dict)
        
        if 'Edgeworth_Maria_Lame-Jervas_1804' in self.doc_path:
            # One footnote seems to be by the author, while the others are by the editor
            # Remove only those by the editor
            text = text.replace('{Footnote: Extracts from William Smith', 'Extracts from William Smith')
            text = text.replace(r"such terms as the sultan did not understand.'}", r"such terms as the sultan did not understand.'")
            text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)

        if 'Scott_Walter_The-Abbot_1820' in self.doc_path:
            # Footnotes are left in the text, unclear if by author
            # long_string = r"""
            #     [Note on old Scottish spelling: leading y = modern 'th'; leading v =\nmodern 'u']"""
            # text = text.replace(long_string, '')
            text = text.replace('Footnote:', '')

        if 'Marlitt_Eugenie_Im-Hause-des-Kommerzienrates_1877' in self.doc_path:
            text = text.replace('28[.]', '')
            # Find numbers enclosed by brackets.
            text = re.sub(r'\[\d*?\]', '', text)

        if 'Wells_H-G_The-First-Men-in-the-Moon_1901' in self.doc_path:
            text = text.replace('* Footnote', '')

        if 'May_Karl_Das-Waldroeschen_1883' in self.doc_path:
            text = text.replace('– Anmerkung des Verfassers.]', '')
            text = text.replace('[', '')
            
        if 'Stevenson-Grift_Robert-Louis-Fanny-van-de_The-Dynamiter_1885' in self.doc_path:
            text = text.replace('\with','with')

        if 'Brooke_Frances_Lady-Julia-Mandeville_1763' in self.doc_path:
            text = text.replace('=Your', '')

        files_list = [
            'Lee_Vernon_Prince-Alberic-and-the-Snake-Lady_1895',
            'Lamb_Charles_Hamlet_1807',
            'Tieck_Ludwig_Abdallah_1792',
            'Boehlau_Helene_Das-Recht-der-Mutter_1896',
            'Tieck_Ludwig_Der-junge-Tischlermeister_1836',
            'Alexis_Willibald_Walladmor_1824',
            'Auerbach_Berthold_Diethlem-von-Buchenberg_1852',
            'Kirchbach_Wolfgang_Das-Leben-auf-der-Walze_1892',
            'Frenssen_Gustav_Joern-Uhl_1901'
        ]
        if any(file_name in self.doc_path for file_name in files_list):
            text = text.replace('^', '')

        if 'Kirchbach_Wolfgang_Das-Leben-auf-der-Walze_1892' in self.doc_path:
            text = text.replace('^', '')
            text = text.replace('Katzenkopp = Schlosser', '')

        if 'Schloegl_Friedrich_Wiener-Blut_1873' in self.doc_path:
            text = text.replace('^', '')
            text = text.replace('=', '-')

        files_list = [
            'Stifter_Adalbert_Der-Hochwald_1841',
            'Stifter_Adalbert_Die-Narrenburg_1843',
        ]
        if any(file_name in self.doc_path for file_name in files_list):
            text = text.replace('@', '')

        if 'Baldwin_Louisa_Sir-Nigel-Otterburnes-Case_1895' in self.doc_path:
            text = text.replace('~', 'v')

        if 'Baldwin_Louisa_Many-Waters-Cannot-Quench-Love_1895' in self.doc_path:
            text = text.replace('th~', '')
            text = text.replace('~', '')
            text = text.replace(']', '')

        files_list = ['Baldwin_Louisa_The-Empty-Picture-Frame_1895',
                      'Galbraith_Lettice_The-Case-of-Lady-Lukestan_1893']
        # Remove tilde
        if any(file_name in self.doc_path for file_name in files_list):
            text = text.replace('~', '')

        if 'Forrester_Andrew_The-Female-Detective_1864' in self.doc_path:
            text = text.replace('~', '')
            text = text.replace(' ~', '')
            text = text.replace('~ ', '')
            text = text.replace('{', '')

        if 'Wells_H-G_A-Modern-Utopia_1905' in self.doc_path:
            text = text.replace('(See also Chapter I., § 6, and Chapter X., §§ 1 and 2.)', '')
            text = re.sub(r'\[Footnote: See.*?\]', '', text, flags=re.DOTALL)
            text = text.replace('Footnote: ', '')

        files_list = ['Eliot_George_Daniel-Deronda_1876', 
                         'Kingsley_Charles_Alton-Locke_1850',
                         'Wells_H-G_The-Time-Machine_1895'
                         ]
        # Replace only the word footnote
        if any(file_name in self.doc_path for file_name in files_list):
            # There are footnotes by the author. Only replace word "Footnote".
            text = text.replace('Footnote: ', '')

        if 'Holcroft_Thomas_Anna-St-Ives_1792' in self.doc_path:
            text = text.replace('[Footnote 1: Omitted.]', '')
            text = text.replace('Footnote 1: ', '')
            text = text.replace('Footnote: ', '')
            text = text.replace('[1]', '')

        if 'Porter_Jane_Thaddeus-of-Warsaw_1803' in self.doc_path:
            text = text.replace('Footnote: ', '')
            text = re.sub(r'\[Illustration:.*?\]', '', text)

        if 'Radcliffe_Ann_Udolpho_1794' in self.doc_path:
            text = re.sub(r'\(\*Note:.*?\)', '', text, flags=re.DOTALL)

        if 'Lohenstein_Daniel_Arminius_1689' in self.doc_path:
            rep_dict = {
                ' \\': 'ö', # replace space followed by backslash
                ' /': '\n',
            }
            text = self.replace_multiple(text, rep_dict)

        if 'Marryat_Frederick_The-Kings-Own_1830' in self.doc_path:
            text = text.replace('(see note 1)', '')

        if 'Scott_Walter_Old-Mortality_1816' in self.doc_path:
            # Insert missing bracket 
            text = text.replace("[COMMANDER-IN-CHIEF OF KING CHARLES II.'s FORCES IN SCOTLAND.]", "COMMANDER-IN-CHIEF OF KING CHARLES II.'s FORCES IN SCOTLAND.")
            text = text.replace('spells like a chambermaid.', 'spells like a chambermaid.]')
            text = re.sub(r'\[Note:.*?\]', '', text, flags=re.DOTALL)
            
        if 'OGrady_Standish_Early-Bardic-Literature_1879' in self.doc_path:
            text = text.replace("[Transcriber's Note: Greek in the original]", '')
            text = text.replace('[Note: "Dream of Angus," Révue Celtique, Vol. III., page 349.', '')
            text = text.replace('(see p. 257; vol. i.)', '')
            text = text.replace('Note: Vol. I., page 155.', '')
            text = text.replace('Note: Publications of Ossianic Society, Vol. I. of Oscar, on pages 34 and 35, Vol. I.', '') 
            text = text.replace('[Note: "Dream of Angus," Révue Celtique, Vol. III., page 349.', '')
            # Match everything after "Note:" up to the next dot or newline character or end of line
            # This doesn't match everything because some notes have dots in them
            pattern = r'Note: (.*?)(?:\.|\n|$)'
            text = re.sub(pattern, '', text, re.DOTALL)            
            text = text.replace('|', '')

        if 'Ebers_George_Eine-aegyptische-Koenigstocher_1864' in self.doc_path:
            text = text.replace(u'\xa0', u' ') # Replace first so that other replacements don't have to take space into consideration
            text = text.replace('Gleich[ge]wicht', 'Gleichgewicht')
            text = text.replace('S. Anmerk. 147.', 's')
            rep_dict = {
                'Anm. 121) S. Band II. S. 26 und Anmerk. 25 [24]': '',
                'Anmerkung 62 und 63 [64 und 65]': '',
                'Siehe auch Th. I. Anmerkung 53. ': '',
                'I. Theil Anmerkung 53. II. Theil Anmerkung 73.': '',
                'Siehe I. Theil Anmerkung 83.': '',
                'Siehe auch Th. I. ': '',
                'Siehe Anmerkung 22 [21] des II. Theils.': '',
            }
            text = re.sub(r'\(Anm\..140..Ham.*?Anmerk.*?121\.', '', text, flags=re.DOTALL)
            text = self.replace_multiple(text, rep_dict)
            text = re.sub(r'Siehe I+\. Theil Anmerkung \d* \[\d*\]', '', text, flags=re.DOTALL)
            text = re.sub(r'Siehe I+\. Theil*?\.', '', text, flags=re.DOTALL)
            text = re.sub(r'Siehe Anmerkung \d+ (des|im) I+\. Theils?.', '', text, flags=re.DOTALL)
            text = re.sub(r'Siehe Anmerkung \d*?\.', '', text, flags=re.DOTALL)
            text = re.sub(r'Anmerkung \d*?\.', '', text, flags=re.DOTALL)
            text = re.sub(r'\([Aa]nm\. \d* \[\d*\]\)', '', text)
            text = re.sub(r'(\(Anm\. \d+\)).*?\.\.', '.', text, flags=re.DOTALL)
            text = re.sub(r'(\(Anm\. \d+\)).*?\n', r'.\n', text)
            text = re.sub(r'\([Aa]nm\. \d*\)', '', text)
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
            # text = text.replace('Anmerkung', '') ###############################

            # Annotations are not formatted consistently and cannot be properly removed with regex.
            # Remove everything up to the end of the line. This is probably too much.

        if 'Grand_Sarah_The-Heavenly-Twins_1893' in self.doc_path:
            text = text.replace('[she wrote]', 'she wrote')
            # Remove footnotes and descriptions of illustrations.
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
            rep_dict = {
                '"—': ' ',
                '—"': ' ',
                '\'—"': ' ',
                '—"\'': ' ',
                '=': ''
                }
            text = self.replace_multiple(text, rep_dict)

        if 'OGrady_Standish_The-Coming-of-Cuculain_1894' in self.doc_path: ############
            text = text.replace('Tuatha=nations, De=gods, Danan=of Dana', '')
            text = re.sub(r'Footnote: .*?\n', '', text)

        if 'Dickens_Charles_Oliver-Twist_1837' in self.doc_path:
            text = text.replace('Footnote:', '')

        if 'Moerike_Eduard_Das-Stuttgarter-Hutzelmaennchen_1853' in self.doc_path:
            text = text.replace('koloczaer kodex altdeutscher ged., hrsg. von mailath usw., s. 232. ', '')

        if 'Tieck_Ludwig_Aufruhr-in-den-Cevennen_1826' in self.doc_path:
            text = text.replace('camisa = chemise, Bluse.', '')
            text = text.replace('Leim = Lehm, Erde', '')
            text = text.replace('Chancelant = uneinig, wankelmütig.', '')
            text = re.sub(r'\(.*?\)', '', text, flags=re.DOTALL)


        if 'Reade_Charles_The-Cloister-and-the-Hearth_1861' in self.doc_path: #############
            # Notes in text were removed manually
            long_pattern = r'(1) Beat down Satan under our feet .* never heard of a cat of\nGod."'
            text = re.sub(long_pattern, '', text, flags=re.DOTALL)
            long_pattern = r'Kyrie Eleison.*tradas nos\.'
            text = re.sub(long_pattern, self.latin_tag, text, flags=re.DOTALL)
            long_pattern = r'\{ou di eautwn.*tys pistews\}'
            text = re.sub(long_pattern, self.greek_tag, text, flags=re.DOTALL)
            text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
            text = re.sub(r'\(\d*?\)', '', text)
            text = text.replace('(', '')
            text = text.replace(')', '')

        if 'Defoe_Daniel_Journal-of-the-Plague-Year_1722' in self.doc_path:
            text = text.replace('[Footnote in the original.]', '')
            text = text.replace('[Footnotes in the original.]', '')
            text = text.replace('{*}', '')
            text = text.replace('*', '')
            text = text.replace('{}', '')

        if 'Fielding_Henry_Jonathan-Wilde_1742' in self.doc_path:
            text = text.replace('[he]', 'he')
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)

        if 'Haggard_H-Rider_Allan-Quartermain_1887' in self.doc_path:
            # Remove endnotes at the end of the text
            long_pattern = r"""Endnote 1\n Among the Zulus a man assumes the ring.*Endnote 21"""
            text = re.sub(long_pattern, '', text, flags=re.DOTALL)
            # Remove endnotes inside the text
            text = re.sub(r'Endnote \d*?(,| )', '', text)
            text = re.sub(r'welded on to the steel Endnote 5', r'welded on to the steel', text)
            text = text.replace('{Endnote 15}', '')
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)

        if 'Fielding_Henry_Amelia_1752' in self.doc_path:
            text = text.replace('{Containing the exordium, &c. ', '')
            text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)

        if 'Scott_Walter_Waverley_1814' in self.doc_path:
            text = re.sub(r'Footnote: See [Nn]ote \d+.?\n?', '', text)
            text = re.sub(r'\[Footnote.*?\]', '', text, flags=re.DOTALL|re.IGNORECASE)
            # Footnotes are not formatted consistently and cannot be properly removed using regex.
            # Remove everything up to the end of the line. This is probably too much.
            text = re.sub(r'Footnote.*?\n', '\n', text)

        if 'Scott_Walter_Guy-Mannering_1815' in self.doc_path:
            long_string = "[*We must again have recourse to the contribution to Blackwood's Magazine, April 1817 :—"
            text = text.replace(long_string, '')
            text = text.replace('Fare ye wee]', 'Fare ye wee')
            text = re.sub('\[.*?\]', '', text, flags=re.DOTALL)
            text = re.sub(r'\*.*?(?=\s)', '', text)

        if  'Scott_Walter_The-Monastery_1820.txt' in self.doc_path:
            text = re.sub(r'\{Footnote.*?\}', '', text, flags=re.DOTALL)
            # Footnotes are not formatted consistently and cannot be properly removed using regex.
            # Remove everything up to the end of the line. This is probably too much.
            text = re.sub(r'Footnote.*?\n', '\n', text)
    

        if 'Shelley_Mary_Mathilda_1820.txt' in self.doc_path:
            rep_dict = {
                '[sic]': '',
                '[': '',
                ']': ''
            }
            text = self.replace_multiple(text, rep_dict)

        if 'Swift_Jonathan_Gullivers-Travels_1726' in self.doc_path:
            text = text.replace('[As given in the original edition.]\n', '')

        if 'Barrie_J-M_Peter-and-Wendy_1911' in self.doc_path:
            text = text.replace('tinker = tin worker', '')
            text = text.replace('Diana = goddess', '')

        if 'Tressell_Robert_The-Ragged-Trousered-Philanthropists_1914' in self.doc_path:
            text = text.replace('}', '')


        if 'MacDonald_George_Robert-Falconer_1868' in self.doc_path:
            text = text.replace('Ã†', 'e')
            text = text.replace('Ã¦', 'e')
            text = text.replace('A|', 'e')
            text = re.sub(r'\d+:', '', text)

        if 'Auerbach_Berthold_Die-Frau-Professorin_1846' in self.doc_path:
            text = text.replace('Fußnoten1 ', '')

        if 'Owenson_Sydney_The-Wild-Irish-Girl_1806' in self.doc_path:
            text = text.replace('Ã¦', 'ae')
            text = text.replace('A|', 'ae')
            text = text.replace('=', '')

        if 'Baldwin_Louisa_The-Shadow-on-the-Blind_1895' in self.doc_path:
            text = text.replace('o~', 'or')

        files_list = ['Edgeworth_Maria_The-Contrast_1804.txt', 
                          'Edgeworth_Maria_Murad-the-Unlucky_1804',
                          'Edgeworth_Maria_Rosanna_1804',
                          'Edgeworth_Maria_The-Will_1804',
                          'Edgeworth_Maria_The-Lottery_1804',
                          'Edgeworth_Maria_The-Manufacturers_1804']
        if any(file_name in self.doc_path for file_name in files_list):
            text = re.sub(r'\{Footnote.*?\}', '', text, flags=re.DOTALL)

        curly_list = ['Edgeworth_Maria_The-Grateful-Negro_1804',
                          'Edgeworth_Maria_To-Morrow_1804',
                          'Edgeworth_Maria_The-Limerick-Gloves_1804',
                          'Hardy_Thomas_A-Pair-of-Blue-Eyes_1873',
                          'Wells_H-G_The-Island-of-Dr-Moreau_1896',
                          'Butler_Samuel_Erewhon_1872',
                          'Edgeworth_Maria_Patronage_1814',
                          'Hardy_Thomas_The-Woodlanders_1886'
                          ]
        # Everything between curly brackets
        if any(file_name in self.doc_path for file_name in curly_list):
            text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)

        if 'Reynolds_George_The-Mysteries-of-London_1844' in self.doc_path:
            text = text.replace('}', '')
            text = text.replace('|', '')

        if 'Haggard_H-Rider_She_1886' in self.doc_path:
            pattern = r'<U+0391><U+039C><U+0395><U+039D><U+0391><U+03A1><U+03A4><.*<U+0395><U+03A5>S<U+0391><U+039C><U+0397><U+039D>'
            text = re.sub(pattern, '', text, flags=re.DOTALL)
            pattern = r'<U+1F08>µe<U+03BD><U+1F71><U+03C1>ta<U+03C2>, t<U+03BF><U+1FE6> ß.*<U+03C8>e<U+03C5>s<U+1F71>µ<U+03B7><U+03BD>'
            text = re.sub(pattern, '', text, flags=re.DOTALL)
            pattern = r' <U+039F><U+03A5><U+039A><U+0391><U+039D><U+0394><U+03A5><U+039D><U+0391><U+0399>.*<U+03B9><U+03BA><U+03C1><U+1F71>te<U+03B9> t<U+1FF7> pa<U+03B9>d<U+1F77>\.'
            text = re.sub(pattern, '', text, flags=re.DOTALL)
            pattern = r'<U+03A4>O<U+039D>T<U+0395>O<U+039D><U+0391><U+039D><U+03A4><U+0399>S<U+03A4><U+0391><U+039D><U+03A4>O<U+039D><U+0395><U+03A0>.*<U+03B9><U+03BA><U+03C1><U+1F71>t<U+03B7><U+03C2> t<U+1FF7> pa<U+03B9>d<U+1F77>\.'
            text = re.sub(pattern, '', text, flags=re.DOTALL)
            pattern = r'Quib~ fact~ iuravit <U+017F>e patre¯ tuu¯ quoq~ im¯ortale¯ o<U+017F>te¯<U+017F>ura.*Domini MCCCCLXXXXV°'
            text = re.sub(pattern, '', text, flags=re.DOTALL)
            pattern = r'Amenartas e.*i MCCCCLXXXXV°\.'
            text = re.sub(pattern, '', text, flags=re.DOTALL)
            text = text.replace('{', '')
            text = text.replace('}', '')

        if 'Scott_Walter_The-Fortunes-of-Nigel_1822' in self.doc_path:
            text = text.replace('[ZOOS ESTI KAI EPI THONI DERKOV]', self.greek_tag)
            text = re.sub(r'\[Footnote.*?\]', '', text, flags=re.DOTALL)
            text = text.replace('=', '')

        if 'Scott_Walter_The-Betrothed_1825' in self.doc_path:
            text = text.replace('|east', 'least')
            text = re.sub(r'Footnote :', '', text, flags=re.DOTALL)

        files_list = ['Edgeworth_Maria_Ormond_1817', 
                          'Edgeworth_Maria_Helen_1834',
                          'Edgeworth_Maria_Patronage_1814',
                          ]
        if any(file_name in self.doc_path for file_name in files_list):
            text = re.sub(r'\[Footnote.*?\]', '', text, flags=re.DOTALL)

        if 'MacDonald_George_Sir-Gibbie_1879' in self.doc_path:
            print('MacDonald_George_Sir-Gibbie_1879 found!!_---------------------')
            long_string = r'{compilers note:  spelled in Greek:  Theta, Epsilon, Omicron, Upsilon; Lambda, Omicron with stress, Gamma, Omicron, Sigma}'
            text = text.replace(long_string, '')
            long_string = r'{compilers note:  spelled in Greek:  Tau, Upsilon with stress, Pi, Tau, Omega}'
            text = text.replace(long_string, '')
            text = re.sub(r'\[\d*?\]', '', text)
            text = re.sub(r'{', '', text)
            text = re.sub(r'}', '', text)
            text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)

        files_list = ['Edgeworth_Maria_Orlandino_1848', 
                          'Brunton_Mary_Discipline_1814',
                          'Reynolds_George_The-Mysteries-of-London_1844.txt',
                          'Dickens_Charles_Sketches-by-Boz_1833',
                          'Newman_John-Henry_Loss-and-Gain_1848',
                          'Fielding_Henry_Shamela_1741',
                          'Ferrier_Susan_Marriage_1818',
                          'Crockett_S-R_Cleg-Kelly_1896'
                          ]
        if any(file_name in self.doc_path for file_name in files_list):
            # Find numbers enclosed by brackets.
            text = re.sub(r'\[\d*?\]', '', text)

        if 'Collins_Wilkie_The-Woman-in-White_1859' in self.doc_path:
            text = text.replace('(see Sermon XXIX. in the Collection by the late Rev. Samuel Michelson, M.A.)', '')

        if 'Edgeworth_Maria_Castle-Rackrent_1800' in self.doc_path:
            text = text.replace('[See GLOSSARY 11]', '') # Brackets within brackets
            text = text.replace('[See GLOSSARY 28]', '')
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)

        if 'Scott_Walter_Kenilworth_1821' in self.doc_path:
            text = text.replace('[See Note 6]', '') # Brackets within brackets
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)

        if 'Sheehan_Patrick-Augustine_My-New-Curate_1900' in self.doc_path:
            #long_pattern = r' PRO MENSE AUGUSTO.\n (Die I^ma Mensis.)\n 1. Excerpta ex Statutis Dioecesanis et Nationalibus.\n 2. De Inspiratione Canonicorum Librorum.\n 3. Tractatus de Contractibus (Crolly).'
            long_pattern = r'PRO MENSE.*?\(Crolly\)'
            text = re.sub(long_pattern, '', text, flags=re.DOTALL)

        if 'Edgeworth_Maria_The-Irish-Incognito_1802' in self.doc_path:
            text = text.replace('right^', 'right')
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)

        if '' in self.doc_path:
            long_string = r"""
                [Lat.; zusammengezogen aus (ita) me Dius Fidius (iuvet) = So wahr mir (der treue) Gott helfe! Bei Gott! Wahrhaftig!]  (Ein römischer Schwur, der wohl schwer zu übersetzen, aber nicht unerklärlich ist.)
                """
            text = text.replace(long_string, '')

        if 'Paul_Jean_Flegeljahre_1804' in self.doc_path:
            text = text.replace('(^-^-)', '')
            text = text.replace('(^^-^)', '')
            text = text.replace('( ^^)', '')
            text = text.replace('(--^^)', '')

        if 'Freytag_Gustav_Die-Ahnen_1872' in self.doc_path:
            text = text.replace('~', 'v')

        files_list = [
            'Edgeworth_Maria_Castle-Rackrent_1800',
            'Edgeworth_Maria_The-Modern-Griselda_1804',
            'Bulwer-Lytton_Edward_Eugene-Aram_1832',
            'Barrie_J-M_Peter-and-Wendy_1911',
            'Scott_Walter_Chronicles-of-the-Canongate_1827',
            'Scott_Walter_Rob-Roy_1817',
            'Scott_Walter_Montrose_1819',
            'Scott_Walter_The-Talisman_1825',
            'Scott_Walter_The-Antiquary_1816',
            'Scott_Walter_The-Black-Dwarf_1816',
            'Scott_Walter_The-Surgeons-Daughter_1827',
            'Scott_Walter_The-Fair-Maid-of-Perth_1828',
            'Scott_Walter_The-Bride-of-Lammermoor_1819',
            'Scott_Walter_The-Heart-of-Midlothian_1818',
            'Scott_Walter_The-Peveril-of-the-Peak_1822.txt',
            'Scott_Walter_Chronicles-of-the-Canongate_1827',
            'Roche_Regina-Maria_The-Children-of-the-Abbey_1796',
            'Kipling_Rudyard_Kim_1901',
            'Equiano_Olaudah_Life-of-Equiano_1789',
            'Radcliffe_Ann_The-Italian_1797',
            'Haggard_H-Rider_King-Solomons-Mines_1885',
            'Eliot_George_Daniel-Deronda_1876',
            'MacDonald_George_Alec-Forbes-of-Howglen_1865',
            'Reade_Charles_It-Is-Never-Too-Late-to-Mend_1856',
            'Maurier_George-du_Trilby_1894',
            'Gore_Catherine_Theresa-Marchmont_1830',
            'Carleton_William_The-Black-Prophet_1847',
            'Nesbit_Edith_The-Story-of-the-Amulet_1906',
            'Bierbaum_Otto_Stilpe_1897',
            'Amory_Thomas_The-Life-of-John-Buncle_1756',
            'Brentano_Clemens_Godwi_1801',
            'Gutzkow_Karl_Briefe-einers-Narren-und-einer-Naerrin_1832',
            'Freytag_Gustav_Die-Ahnen_1872',
            'Weerth_Georg_Fragment-eines-Romans_1845',
            'Alexis_Willibald_Isegrimm_1854',
            'Holz-Schlaf_Arno-Johannes_Papa-Hamlet_1889'
            ]
        if any(file_name in self.doc_path for file_name in files_list):
            # Everything in brackets
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
        return text


    def preprocess_text(self, text):
        # Replace sentence terminating chars so that they are not affected by preprocessing
        text = self.preprocess_individual_files(text)
        description_dict = {
            '.': 'DOT',
            ':': 'COLON',
            ';': 'SEMICOLON',
            '?': 'QUESTIONMARK',
            '!': 'EXCLAMATIONMARK',
            '...': 'ELLIPSIS',
        }
        terminating_dict = {char: 'xxxxx' + description_dict[char] for char in self.sentence_chars}
        text = self.replace_multiple(text, terminating_dict)

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
        terminating_dict_swap = {v: k for k, v in terminating_dict.items()}
        text = self.replace_multiple(text, terminating_dict_swap)
        self.check_annotations(text)
        # List of texts whose remaining special characters are not a problem
        chars_ok_list = [
            'Collins_Wilkie_Armadale_1864',
            'Carleton_William_Fardorougha-the-Miser_1839',
            'Radcliffe_Ann_The-Italian_1797',
            'Conrad_Joseph_Heart-of-Darkness_1899',
            'Sterne_Laurence_Tristram-Shandy_1759',
            'Kipling_Rudyard_The-Bisara-of-Pooree_1888',
            'Doyle_Arthur-Conan_The-Adventure-of-the-Missing-Three-Quarter_1905',
            'Carleton_William_The-Battle-of-the-Factions_1830',
            'Scott_Walter_Waverley_1814',
            'Dickens_Charles_The-Pickwick-Papers_1836',
            'Sheridan_Frances_Miss-Sidney-Bidulph_1761',
            'Scott_Walter_Woodstock_1826',
            'Dickens_Charles_Sketches-by-Boz_1833',
            'Richardson_Samuel_Pamela_1740',
            'Kipling_Rudyard_The-Conversion-of-Aurelian-McGoggin_1888',
            'Fielding_Sarah_The-Countess-of-Dellwyn_1759',
            'Reade_Charles_The-Cloister-and-the-Hearth_1861',
            'Kipling_Rudyard_Yoked-with-an-Unbeliever_1888',
            'Wells_H-G_The-Invisible-Man_1897',
            'Yonge_Charlotte_The-Heir-of-Redclyffe_1853',
            'Scott_Walter_The-Pirate_1822',
            'Bulwer-Lytton_Edward_Paul-Clifford_1830',
            'Trollope_Anthony_The-Way-We-Live-Now_1875',
            'Hays_Mary_Memoirs-of-Emma-Courtney_1796',
            'Morier_James_Hajji-Baba_1824',
            'Smith_Charlotte_The-Old-Manor-House_1793',
            'Thackerey_William-Makepeace_Pendennis_1848',
            'Mackenzie_Henry_The-Man-of-Feeling_1771',
            #'Dickens_Charles_Oliver-Twist_1837',
            'Collins_Wilkie_The-Moonstone_1868',
            'Edgeworth_Maria_Helen_1834',
            'Trollope_Anthony_Can-You-Forgive-Her_1864',
            'Forster_E-M_The-Longest-Journey_1907',
            'Kipling_Rudyard_His-Chance-in-Life_1888',
            'Doyle_Arthur-Conan_The-Lost-World_1912',
            'Conrad_Joseph_The-Secret-Agent_1907',
            'Walpole_Horace_Otranto_1764',
            'Shelley_Mary_Valperga_1823',
            'Barker_Jane_A-Patch-Work-Screen-for-the-Ladies_1723',
            'Godwin_William_Caleb-Williams_1794',
            'Edgeworth_Maria_Tales-of-a-Fashionable-Life_1809',
            'Gissing_George_New-Grub-Street_1891',
            'Gaskell_Elizabeth_Ruth_1853',
            'Reade_Charles_It-Is-Never-Too-Late-to-Mend_1856',
            'Kipling_Rudyard_The-Jungle-Book_1894',
            'Galbraith_Lettice_In-the-Seance-Room_1893',
            'Kipling_Rudyard_Watches-of-the-Night_1888',
            'Brunton_Mary_Self-Control_1811',
            'LeFanu_Joseph-Sheridan_The-House-by-the-Church-Yard_1863',
            'Wells_H-G_Love-and-Mr-Lewisham_1900',
            'Scott_Walter_The-Betrothed_1825',
            'Edgeworth_Maria_Belinda_1801',
            'Cholmondeley_Mary_Red-Pottage_1899',
            'Gaskell_Elizabeth_Cranford_1851',
            'Wells_H-G_Kipps_1906',
            'LeFanu_Joseph-Sheridan_Mr-Justice-Harbottle_1872',
            'Trollope_Anthony_The-Eustace-Diamonds_1871',
            'Edgeworth_Maria_Patronage_1814',
            'Fielding_Henry_Tom-Jones_1749',
            'Godwin_William_Fleetwood_1805',
            'Galbraith_Lettice_A-Ghosts-Revenge_1893',
            'Wells_H-G_Ann-Veronica_1909',
            'Defoe_Daniel_Journal-of-the-Plague-Year_1722',
            'Collins_Wilkie_No-Name_1862',
            'Moore_George_A-Mummers-Wife_1885',
            'Sheehan_Patrick-Augustine_My-New-Curate_1900',
            'Amory_Thomas_The-Life-of-John-Buncle_1756',
            'Forrester_Andrew_The-Female-Detective_1864',
            'Richardson_Samuel_Clarissa_1748',
            'Scott_Walter_The-Abbot_1820',
            'Meyrink_Gustav_Des-deutschen-Spiessers-Wunderhorn_1913',
            'Lasker-Schueler_Else_Mein-Herz_1912',
            'Storm_Theodor_Pole-Poppenspaeler_1874',
            'Wieland_Christoph-Martin_Aristipp-und-einige-seiner-Zeitgenossen_1800',
            'Musaeus_Johann-Karl-August_Grandison-der-Zweyte_1760',
            'Paul_Jean_Titan_1800',
            'Kurz_Hermann_Der-Sonnenwirt_1856',
            'Hoffmann_ETA_Ignaz-Denner_1816',
            'Hawel_Rudolf_Im-Reiche-der-Homunkuliden_1910',
            'Bierbaum_Otto_Stilpe_1897',
            'Gotthelf_Jeremias_Leiden-und-Freuden-eines-Schulmeisters_1838',
        ]
        if self.bookname not in chars_ok_list:
            self.check_characters(text)
        return text
    

class NgramCounter(DataHandler):
    def __init__(self, language, doc_paths, chunks):
        super().__init__(language, 'ngram_counts', 'pkl')
        self.modes = ['unigram_counts', 'bigram_counts', 'trigram_counts']
        self.doc_paths = doc_paths
        self.chunks = chunks # Chunks for all files

    def create_filename(self, **kwargs):
        return f'{self.modes[0]}.{self.data_type}'

    def load_all_data(self):
        all_data = {}
        for mode in self.modes:
            if not os.path.exists(os.path.join(self.output_dir, f'{mode}.pkl')):
                data  = self.load_data(mode=mode)
                all_data[mode] = data
        return all_data
    
    def create_data(self, **kwargs):
        self.logger.info('Counting Ngrams')
        total_unigram_counts, total_bigram_counts, total_trigram_counts = Counter(), Counter(), Counter()
        book_unigram_mapping, book_bigram_mapping, book_trigram_mapping = {}, {}, {}


        for doc_chunks in self.chunks: # Chunks for one document
            book_unigram_counts = {}
            book_bigram_counts = {}
            book_trigram_counts = {}
            for chunk in doc_chunks: # Individual chunks belonging to one document
                file_name = chunk.file_name

                for unigram, counts in chunk.unigram_counts.items():
                    book_unigram_counts[unigram] = book_unigram_counts.get(unigram, 0) + counts
                for bigram, counts in chunk.bigram_counts.items():
                    book_bigram_counts[bigram] = book_bigram_counts.get(bigram, 0) + counts
                for trigram, counts in chunk.trigram_counts.items():
                    book_trigram_counts[trigram] = book_unigram_counts.get(trigram, 0) + counts

            book_unigram_mapping[file_name] = book_unigram_counts
            book_bigram_mapping[file_name] = book_bigram_counts
            book_trigram_mapping[file_name] = book_trigram_counts
            total_unigram_counts.update(book_unigram_counts)
            total_bigram_counts.update(book_bigram_counts)
            total_trigram_counts.update(book_trigram_counts)

        save_dict = {
            'total_unigram_counts': total_unigram_counts,
            'book_unigram_mapping': book_unigram_mapping,
        }
        # Save data for each unigram separately to avoid memory problems
        self.save_data(data=save_dict, file_name=f'unigram_counts.{self.data_type}')
        del total_unigram_counts
        del book_unigram_mapping
        self.logger.info('Counted unigrams')


        # Keep only 2000 most frequent bi- and trigrams because of data sparsity
        total_bigram_counts = dict(heapq.nlargest(2000, total_bigram_counts.items(), key=lambda x: x[1]))
        total_trigram_counts = dict(heapq.nlargest(2000, total_trigram_counts.items(), key=lambda x: x[1]))


        book_bigram_mapping_filtered = {}
        for book, book_dict in book_bigram_mapping.items():
            book_dict_ = {}
            for ngram in set(total_bigram_counts.keys()):
                if ngram in book_dict:
                    book_dict_[ngram] = book_dict[ngram]
            book_bigram_mapping_filtered[book] = book_dict_
        del book_bigram_mapping
        save_dict = {
            'total_bigram_counts': total_bigram_counts,
            'book_bigram_mapping': book_bigram_mapping_filtered}
        self.save_data(data=save_dict, file_name=f'bigram_counts.{self.data_type}')
        del total_bigram_counts
        self.logger.info('Counted bigrams')


        book_trigram_mapping_filtered = {}
        for book, book_dict in book_trigram_mapping.items():
            book_dict_ = {}
            for ngram in set(total_trigram_counts.keys()):
                if ngram in book_dict:
                    book_dict_[ngram] = book_dict[ngram]
            book_trigram_mapping_filtered[book] = book_dict_
        del book_trigram_mapping
        save_dict = {
            'book_trigram_mapping': book_trigram_mapping_filtered,
            'total_trigram_counts': total_trigram_counts}
        self.save_data(data=save_dict, file_name=f'trigram_counts.{self.data_type}')
        self.logger.info('Counted trgrams')
