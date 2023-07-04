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
from utils import load_list_of_lines, save_list_of_lines
logging.basicConfig(level=logging.DEBUG)

# %%


class Tokenizer():
    def __init__(self, language=None, doc_paths=None, tokens_per_chunk=500):
        self.language = language
        self.doc_paths = doc_paths
        self.tokens_per_chunk = tokens_per_chunk # Nr tokens in chunk
        self.nlp = self.load_spacy_model()

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
        terminating_chars = ['.', ':', ';', '?', '!', ')', ']', '...']
        all_chunks = []
        text_split = text.split()
        i_start = 0
        i_end = 0
        while i_end < len(text_split) - self.tokens_per_chunk - 1:
            i_end = i_start + self.tokens_per_chunk
            while i_end < len(text_split):
                element = text_split[i_end]
                if any([x in element for x in terminating_chars]):
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
        all_chunks = self.split_into_chunks(text)

        logging.info('Tokenizing words.')
        start = time.time()
        new_chunks = []
        pp = Preprocessor(self.language, doc_path)
        for chunk in all_chunks:
            chunk = pp.preprocess_text(chunk)
            chunk = [token.text for token in self.nlp(chunk)] ##########
            new_chunks.append(chunk)
        print(f'Time for spacy tokenization: {time.time()-start}')
        #assert len(all_chunks) == len(all_chunks) #################
        return new_chunks
    

    def get_tokenized_words(self, doc_path, remove_punct=False, lower=False, as_chunk=True):
        tokenized_words_path = doc_path.replace('/raw_docs', '/tokenized_words')     
        if os.path.exists(tokenized_words_path):
            all_chunks = load_list_of_lines(tokenized_words_path, 'str')
            # logging.info(f'Loaded tokenized words for {doc_path}')
        else:
            all_chunks = self.tokenize_words(doc_path)
            logging.info(f'Tokenizing {doc_path}')
            save_list_of_lines(all_chunks, tokenized_words_path, 'str')

        all_chunks = Postprocessor(remove_punct=remove_punct, lower=lower).postprocess_chunks(all_chunks)

        if as_chunk == False:
            all_words = []
            for chunk in all_chunks:
                chunk = chunk.split()
                all_words.extend(chunk)
            all_chunks = all_words
        # logging.info('Returning tokenized words as list of chunks.')
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
            # "'" is ignored if it is part of a word (Mary's -> Mary + 's)
            punctuation = set(string.punctuation + '’' + '‘' + '—' + '“' + '”' + '–')
            text = text.split()
            # Remove list items that consit only of puncutation as well as punctuation that belongs to other tokens.
            text = [''.join(char for char in item
                        if char not in punctuation)
                        for item in text if item != '']
            text = ' '.join(text)

        # text = re.sub('[^a-zA-Z0-9äöüÄÖÜ\']+', ' ', text).strip() #####################
        # text = text.split() ##############3
        # text = ' '.join(text) #####################
        # Preprocessor().check_characters(text) #####################################
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
    def __init__(self, language=None, doc_path=None):
        self.language = language
        self.doc_path = doc_path

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
            'Note',
            '[see',
            '(see',
            # 'Supplement',
            # 'viz.',   # videlicet,
            'annotator',############
            'editor',################
            'greek',
            'french',
            'latin',
            'italian',
            'spanish',
            'german'
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
            'griechisch',
            'französisch'
            'latein',
            'italienisch',
            'spanisch',
            'englisch'
            ]  # vergleiche
            # v.', 'vergleiche', siehe, s., Erklärung, Bemerkung, Anhang

        if self.language == 'eng':
            word_list = annotation_words_eng
        else:
            word_list = annotation_words_ger

        word_list = [r'\b' + word for word in word_list]
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
                    f.write(self.doc_path + '\t' + word + '\t' + text[idx-step:idx+(step*4)].replace('\n', ' ') + '\n')


    def check_characters(self, text):

        def get_char_idxs(char, text):
            return [i for i, letter in enumerate(text) if letter == char]

        # inner "'" is escaped
        chars = re.sub(r'[\'A-Za-z0-9ÄÖÜäöü,?!-;_— ]+', '', text)
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
                        f.write(self.doc_path + '\t' + char + '\t' + text[idx-step:idx+4*step].replace('\n', ' ') + '\n')

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
            '"—': ' ',              # Grand_Sarah_The-Heavenly-Twins_1893
            '—"': ' ',              # Grand_Sarah_The-Heavenly-Twins_1893
            '\'—"': ' ',              # Grand_Sarah_The-Heavenly-Twins_1893
            '—"\'': ' ',              # Grand_Sarah_The-Heavenly-Twins_1893
            "scoundrel?'": 'scoundrel ', # Collins_Wilkie_Armadale_1864
            'I was;3': 'I was; 3',   # Dickens_Charles_The-Pickwick-Papers_1836
            'Kukuanaland.1': 'Kukuanaland.', # Haggard_H-Rider_King-Solomons-Mines_1885.txt
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
        docs_with_line = ['Anonymous_Anonymous_Vertue-Rewarded_1693', 
                        'Anonymous_Anonymous_The-Adventures-of-Anthony-Varnish_1786', 
                        'Anonymous_Anonymous_The-Triumph-Prudence-Over-Passion_1781',
                        'Chaigneau_William_The-History-of-Jack-Connor_1752']
        if any(file_name in self.doc_path for file_name in docs_with_line):
            text = text.replace('|', '')
            text = text.replace('¦', '')

        if 'Blackmore_R-D_Clara-Vaughan_1864' in self.doc_path:
            long_string = """"[Greek: Kynòs ómmat' echôn, kradíen d' eláphoio.]"--C.V. 1864."""
            text = text.replace(long_string, '')

        greek_list = ['Reade_Charles_Hard-Cash_1863',
                      'Kingsley_Charles_Westward-Ho_1855']
        if any(file_name in self.doc_path for file_name in greek_list):
            text = text.replace('[Greek text]', 'xxxxxGREEK')

        greek_list = ['Reade_Charles_It-Is-Never-Too-Late-to-Mend_1856']
        if any(file_name in self.doc_path for file_name in greek_list):
            text = text.replace('[Greek letters]', 'xxxxxGREEK')

        greek_list = ['Blackmore_R-D_Lorna-Doone_1869']
        if any(file_name in self.doc_path for file_name in greek_list):
            text = text.replace('[Greek word]', 'xxxxxGREEK')

        if 'Hardy_Thomas_The-Woodlanders_1886' in self.doc_path:
            text = text.replace('{Greek word: irony}', 'xxxxxGREEK')

        if 'Hughes_Thomas_Tom-Browns-Schooldays_1857' in self.doc_path:
            text = text.replace('[Greek text]', 'xxxxxGREEK')
            text = text.replace('[greek text deleted]', 'xxxxxGREEK')

        if 'Buchan_John_John-Burnet-of-Barns_1897' in self.doc_path:
            text = text.replace('[Greek: polypenthes]', 'xxxxxGREEK')
            text = text.replace('[Greek: polypous]', 'xxxxxGREEK')

        if 'Eliot_George_Adam-Bede_1859' in self.doc_path:
            text = text.replace('[two greek words omitted]', 'xxxxxGREEK')

        if 'Eliot_George_Janets-Repentance_1857' in self.doc_path:
            long_string = """
            [Greek: deinon to tiktein estin.]
            """
            text = text.replace(long_string, 'xxxxxGREEK')

        if 'Eliot_George_Sad-Fortunes-of-Rev-Amos-Barton_1857' in self.doc_path:
            long_string = """ [Greek: all edu gar toi ktema tes uikes labien
                            tolma dikaioi d' authis ekphanoumetha.]"""
            text = text.replace(long_string, 'xxxxxGREEK')

        if '' in self.doc_path:
            rep_dict = {
                '[Greek: to kalon]': 'xxxxxGREEK',
                '[Greek: gnothi seauton]': 'xxxxxGREEK',
                '[Greek: hetairai]': 'xxxxxGREEK',
                '[Greek: euraeka]': 'xxxxxGREEK',
            }
            text = self.replace_multiple(text, rep_dict)

        brackets_list = ['Fontane_Theodor_Mathilde-Moehring_1906',
                        'Reuter_Christian-Friedrich_Schelmuffsky_1696',
                        'Holcroft_Thomas_The-Adventures-of-Hugh-Trevor_1794',
                        'Sterne_Laurence_Tristram-Shandy_1759',
                        'Smollett_Tobias_Humphry-Clinker_1771',
                        'Richardson_Samuel_Clarissa_1748',
                        'More_Hannah_Coelebs_1814']
        # In some of these texts, the brackets could be replaced automatically.
        # Finding errors is easier by removing them.
        if any(file_name in self.doc_path for file_name in brackets_list):
            rep_dict = {
                '[': '',
                ']': ''
            }
            text = self.replace_multiple(text, rep_dict)

        footnote_list = ['Eliot_George_Daniel-Deronda_1876', 
                         'Kingsley_Charles_Alton-Locke_1850',
                         'Wells_H-G_The-Time-Machine_1895'
                         'Scott_Walter_The-Abbot_1820',
                         'Wells_H-G_A-Modern-Utopia_1905',
                         'Edgeworth_Maria_Lame-Jervas_1804'
                         ]
        if any(file_name in self.doc_path for file_name in footnote_list):
            # There are footnotes by the author. Only replace word "Footnote".
            text = text.replace('Footnote: ', '')

        if 'Porter_Jane_Thaddeus-of-Warsaw_1803' in self.doc_path:
            text = text.replace('Footnote: ', '')
            text = re.sub(r'\[Illustration:.*?\]', '', text)

        if 'Radcliffe_Ann_Udolpho_1794' in self.doc_path:
            text = re.sub(r'\(*Note:.*?\)', '', text)

        if 'Lohenstein_Daniel_Arminius_1689' in self.doc_path:
            rep_dict = {
                ' \\': 'ö', # replace space followed by backslash
                ' /': '\n',
            }
            text = self.replace_multiple(text, rep_dict)

        if 'Scott_Walter_Old-Mortality_1816' in self.doc_path:
            text = re.sub(r'\[Note:.*?\]', '', text, flags=re.DOTALL)

        if 'OGrady_Standish_Early-Bardic-Literature_1879' in self.doc_path:
            text = text.replace("[Transcriber's Note: Greek in the original]", '')
            text = text.replace('[Note: "Dream of Angus," Révue Celtique, Vol. III., page 349.', '')


        if 'Ebers_George_Eine-aegyptische-Koenigstocher_1864' in self.doc_path:
            text = text.replace(u'\xa0', u' ') # Replace first so that other replacements don't have to take space into consideration
            rep_dict = {
                'Anm. 121) S. Band II. S. 26 und Anmerk. 25 [24]': '',
                'Anmerkung 62 und 63 [64 und 65]': '',
                'Siehe auch Th. I. Anmerkung 53. ': '',
                'Siehe auch Th. I. ': '',
                'Siehe Anmerkung 22 [21] des II. Theils.': ''
            }
            text = self.replace_multiple(text, rep_dict)
            text = re.sub(r'Siehe I+\. Theil Anmerkung \d* \[\d*\]', '', text, flags=re.DOTALL)
            text = re.sub(r'Siehe I+\. Theil*?\.', '', text, flags=re.DOTALL)
            text = re.sub(r'Siehe Anmerkung \d+ (des|im) I+\. Theils?.', '', text, flags=re.DOTALL)
            text = re.sub(r'Siehe Anmerkung \d*?\.', '', text, flags=re.DOTALL)
            text = re.sub(r'Anmerkung \d*?\.', '', text, flags=re.DOTALL)
            text = re.sub(r'\([Aa]nm\. \d* \[\d*\]\)', '', text)
            text = re.sub(r'\([Aa]nm\. \d*\)', '', text)
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
            # text = text.replace('Anmerkung', '') ###############################

            # Annotations are not formatted consistently and cannot be properly removed with regex.
            # Remove everything up to the end of the line. This is probably too much.
            text = re.sub(r'[ ]?(\(Anm\. \d+\)).*?\.\.', '.', text, flags=re.DOTALL)
            text = re.sub(r'[ ]?(\(Anm\. \d+\)).*?\n', '.\n', text)

        if 'Grand_Sarah_The-Heavenly-Twins_1893' in self.doc_path:
            text = text.replace('[she wrote]', 'she wrote')
            # Remove footnotes and descriptions of illustrations.
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)

        if 'OGrady_Standish_The-Coming-of-Cuculain_1894' in self.doc_path: ############
            text = re.sub(r'Footnote: .*?\n', '', text)

        if 'Haggard_H-Rider_Allan-Quartermain_1887' in self.doc_path:
            # Remove endnotes at the end of the text
            long_pattern = r"""Endnote 1\n Among the Zulus a man assumes the ring.*Endnote 21"""
            text = re.sub(long_pattern, '', text, flags=re.DOTALL)
            # Remove endnotes inside the text
            text = re.sub(r'Endnote \d*?(,| )', '', text)

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
            text = text.replace('[As given in the original edition.]', '')
    
        if 'Auerbach_Berthold_Die-Frau-Professorin_1846' in self.doc_path:
            text = text.replace('Fußnoten1 ', '')

        if 'Owenson_Sydney_The-Wild-Irish-Girl_1806' in self.doc_path:
            text = text.replace('Ã¦', 'ae')
            text = text.replace('A|', 'ae')

        edgeworth_list = ['Edgeworth_Maria_The-Contrast_1804.txt', 
                          'Edgeworth_Maria_Murad-the-Unlucky_1804',
                          'Edgeworth_Maria_Rosanna_1804',
                          'Edgeworth_Maria_The-Will_1804',
                          'Edgeworth_Maria_The-Lottery_1804',
                          'Edgeworth_Maria_The-Manufacturers_1804']
        if any(file_name in self.doc_path for file_name in edgeworth_list):
            text = re.sub(r'\{Footnote.*?\}', '', text, flags=re.DOTALL)

        
        curly_list = ['Edgeworth_Maria_The-Grateful-Negro_1804',
                          'Edgeworth_Maria_To-Morrow_1804',
                          'Edgeworth_Maria_The-Limerick-Gloves_1804',
                          'Hardy_Thomas_A-Pair-of-Blue-Eyes_1873',
                          'Wells_H-G_The-Island-of-Dr-Moreau_1896',
                          'Butler_Samuel_Erewhon_1872',
                          'Edgeworth_Maria_Patronage_1814'
                          ]
        if any(file_name in self.doc_path for file_name in curly_list):
            text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)


        edgeworth_list = ['Edgeworth_Maria_Ormond_1817', 
                          'Edgeworth_Maria_Helen_1834',
                          'Edgeworth_Maria_Patronage_1814',
                          'Scott_Walter_The-Betrothed_1825',
                          'Scott_Walter_The-Fortunes-of-Nigel_1822']
        if any(file_name in self.doc_path for file_name in edgeworth_list):
            text = re.sub(r'\[Footnote.*?\]', '', text, flags=re.DOTALL)

        edgeworth_list = ['Edgeworth_Maria_Orlandino_1848', 
                          'Brunton_Mary_Discipline_1814',
                          'Reynolds_George_The-Mysteries-of-London_1844.txt',
                          'Marlitt_Eugenie_Im-Hause-des-Kommerzienrates_1877',
                          'Dickens_Charles_Sketches-by-Boz_1833',
                          'Holcroft_Thomas_Anna-St-Ives_1792',
                          'Newman_John-Henry_Loss-and-Gain_1848',
                          'Fielding_Henry_Shamela_1741',
                          'MacDonald_George_Sir-Gibbie_1879',
                          'Ferrier_Susan_Marriage_1818',
                          'Collins_Wilkie_The-Woman-in-White_1859',
                          'Amory_Thomas_The-Life-of-John-Buncle_1756',
                          'Crockett_S-R_Cleg-Kelly_1896'
                          ]
        if any(file_name in self.doc_path for file_name in edgeworth_list):
            # Find numbers enclosed by brackets.
            text = re.sub(r'\[\d*?\]', '', text)

        if 'Reade_Charles_The-Cloister-and-the-Hearth_1861' in self.doc_path: #############
            # Notes in text were removed manually
            long_pattern = r'(1) Beat down Satan under our feet .* never heard of a cat of God."'
            text = re.sub(long_pattern, '', text)
            text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
            text = re.sub(r'\(.*?\)', '', text)

        if 'Defoe_Daniel_Journal-of-the-Plague-Year_1722' in self.doc_path:
            text = text.replace('[Footnote in the original.]', '')

        if 'Fielding_Henry_Jonathan-Wilde_1742' in self.doc_path:
            text = text.replace('[he]', 'he')
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)

        brackets_list = ['Edgeworth_Maria_The-Irish-Incognito_1802',
                         'Edgeworth_Maria_Castle-Rackrent_1800',
                         'Edgeworth_Maria_The-Modern-Griselda_1804',
                         'Bulwer-Lytton_Edward_Eugene-Aram_1832',
                         'Barrie_J-M_Peter-and-Wendy_1911',
                         'Scott_Walter_Chronicles-of-the-Canongate_1827',
                         'Scott_Walter_Rob-Roy_1817',
                         'Scott_Walter_Montrose_1819',
                         'Scott_Walter_Kenilworth_1821',
                         'Scott_Walter_The-Talisman_1825',
                         'Scott_Walter_The-Antiquary_1816',
                         'Scott_Walter_The-Black-Dwarf_1816',
                         'Scott_Walter_The-Surgeons-Daughter_1827',
                         'Scott_Walter_The-Fair-Maid-of-Perth_1828',
                         'Scott_Walter_The-Bride-of-Lammermoor_1819',
                         'Scott_Walter_The-Heart-of-Midlothian_1818',
                         'Scott_Walter_The-Peveril-of-the-Peak_1822.txt',
                         'Scott_Walter_Chronicles-of-the-Canongate_1827',
                         'Bierbaum_Otto_Stilpe_1897',
                         'Roche_Regina-Maria_The-Children-of-the-Abbey_1796',
                         'Kipling_Rudyard_Kim_1901'
                         'Equiano_Olaudah_Life-of-Equiano_1789'
                         ]
        if any(file_name in self.doc_path for file_name in brackets_list):
            # Match everything in parentheses
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
        return text


    def preprocess_text(self, text):
        text = self.preprocess_individual_files(text)
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
        text = unidecode(text)########################################
        text = self.replace_multiple(text, umlaut_dict_swap)
        self.check_annotations(text)
        text = text.split()
        text = ' '.join(text)
        self.check_characters(text)
        return text