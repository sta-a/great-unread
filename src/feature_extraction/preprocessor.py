  
import os
import pandas as pd
import regex as re
from unidecode import unidecode
import sys
sys.path.append("..")
from utils import get_filename_from_path, DataHandler
import hashlib
from difflib import ndiff
from pathlib import Path
import time


class Preprocessor(DataHandler):
    # Preprocess text before sentence tokenization        
    def __init__(self, language, doc_path, tokens_per_chunk, output_dir='preprocess', data_type='csv'):
        super().__init__(language=language, tokens_per_chunk=tokens_per_chunk, data_type=data_type)  
        self.language = language
        self.output_dir = os.path.join(self.data_dir, output_dir) # No language subdirs
        self.doc_path = doc_path
        self.number_tag = '0NUMERICTAG'
        # self.initials_tag = '<INITIALS>'
        self.bookname = get_filename_from_path(self.doc_path)
        self.regex_rep_counter = 0
        self.simple_rep_df = self.get_simple_rep_df()

    def get_simple_rep_df(self):
        df = pd.read_csv(os.path.join(self.output_dir, 'replacement_values.csv'), sep=self.separator, header=0, index_col=None, engine='python').fillna("''")

        if df['file_name'].str.contains(self.bookname).any():
            df = df[df['file_name'].str.contains(self.bookname)]
            return df
        else:
            return None

    def preprocess_text(self, text):
        self.logger.debug(f'Preprocessing.')
        text = self.preprocess_individual_files(text)
        if self.simple_rep_df is not None:
            for _, row in self.simple_rep_df.iterrows():
                text = self.simple_replace(text, row['to_replace'], row['replace_with'])

        # Replace all "'d" at the end of a word (train'd -> trained)
        # Doesn't work because there are other words, like had and would that use this abbreviation
        # text = self.regex_replace(False, True, r"(?:\w)\'d\b", 'ed', text)

        # Replace unicode tags
        text = self.replace_unicode_tags(text)

        # Replace initials
        # Doesn't work because dot at the end could be sentence boundary
        # pattern = r'\b(?:[A-Z]\.)+'
        # text = self.regex_replace(False, True, pattern, self.initials_tag, text)

        # Correct spelling variations, preprocess for easier handling of text
        # Remove accents but keep umlauts
        rep_dict = self.get_rep_dict()
        umlaut_dict = {
            'ä': 'xxxxxae', 
            'ö': 'xxxxxoe', 
            'ü': 'xxxxxue', 
            'Ä': 'xxxxxcapae', 
            'Ö': 'xxxxxcapoe', 
            'Ü': 'xxxxxcapue'}
        rep_dict.update(umlaut_dict)
        umlaut_dict_swap = {v: k for k, v in umlaut_dict.items()}
        text = self.replace_multiple(text, rep_dict)
        text = unidecode(text)
        text = self.replace_multiple(text, umlaut_dict_swap)

        # Check for mistakes, inconsistencies, annotations
        self.check_annotations(text)
        # List of files with special chars which can be removed automatically
        chars_ok_list = self.get_chars_ok_list() # List of texts whose remaining special characters are not a problem
        # if self.bookname not in chars_ok_list: #############
        #     self.check_characters(text)
        self.check_characters(text)

        # Remove all non-allowed chars
        allowed_chars = r'[\'A-Za-z0-9ÄÖÜäöü,?!-;_— ]+'
        text = self.regex_replace(False, False, f'[^{allowed_chars}]', ' ', text)

        # Remove commas inside numbers
        text = self.regex_replace(False, False, r'(?<=\d+)[,\'\.](?=\d+)', '', text)
        # Replace numbers
        text = self.regex_replace(False, False, r'\d+', self.number_tag, text)

        # Clean up whitespace
        text = text.split()
        text = ' '.join(text)
        self.logger.debug(f'Finished preprocessing.')
        return text

    def check_annotations(self, text):
        annotation_words_eng = [
            #'Addendum',
            #'Annotation',
            #'Appendix',
            # 'cf.',   # confer
            'Corrigendum',
            #'e.g.',  # exempli gratia
            'endnote',
            'Footnote',
            # 'Footer',
            #'i.e.',  # id est
            # 'Note',
            '[see',
            #'(see',
            # 'Supplement',
            # 'viz.',   # videlicet,
            'annotator',
            #'editor',
            # 'greek',
            # 'french',
            # 'latin',
            # 'italian',
            # 'spanish',
            # 'german'
            ] # comment, remark
        annotation_words_ger = [
            'Anm.', 
            # 'Anmerkung',
            'Annotation',
            # 'Ergänzung',
            'Fussnote'
            # 'Fußnote',
            # 'Kommentar',
            'Nebenbemerkung',
            # 'Referenz',
            's. a.',
            's.a.', # siehe auch
            '(siehe',
            '[siehe',
            'siehe auch',
            'vergleiche auch', 
            'vgl.',
            # 'Herausgeber',
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
                pattern = re.escape(word)
                idx = re.search(pattern, text)
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
        chars = self.regex_replace(False, False, allowed_chars, '', text)
        # Remove "=" between numbers
        chars = self.regex_replace(False, True, r'\d[ ]?=[ ]?\d', '', chars)
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

       
    def replace_unicode_tags(self, text):
        text = self.regex_replace(False, True, r'<U\+[0-9A-F]+>', '', text, flags = re.VERBOSE | re.MULTILINE)
        return text

    def replace_multiple(self, text, rep_dict):
        # Replace multiple values. Don't check if replacements have occurred. Not all chars to replace are contained in all texts.
        # https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string
        rep = dict((re.escape(k), v) for k, v in rep_dict.items()) 
        pattern = re.compile("|".join(rep.keys()))
        text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
        return text


    def get_rep_dict(self):
        rep_dict = {
            u'\xa0': u' ',
            '/\n': '\n',            # Lohenstein_Daniel_Arminius_1689
            '\r\n': ' ',
            '\n': ' ',
            '\t': ' ',
            '|': '',
            '¦': '',
            'ß': 'ss', 
            'å': 'ä',              # Lohenstein_Daniel_Arminius_1689
            'ů': 'ü',              # Lohenstein_Daniel_Arminius_1689
            'æ': 'ä',              # Gessner_Salomon_Daphnis_1854
            'œ': 'ö',              # Gessner_Salomon_Daphnis_1854
            '’': "'",               # 'o’clock'-> "o'clock"
            # '&c.': 'xxxxxetc',    # Sterne_Laurence_Tristram-Shandy_1759
            '--': ' ',              # Shelley_Mary_Perkin-Warbeck_1830.txt
            "'-": '',               # Carleton_William_Fardorougha-the-Miser_1839, Carleton_William_The-Emigrants-of-Ahadarra_1847
            '"\'': '"',          # Amory_Thomas_The-Life-of-John-Buncle_1756, double quote followed by single quote
            '\'"': '"',
            '»': "'",
            '«': "'",
            '‘': "'",
            '”': "'",
            '“': "'",
            '"': "'",
            '›': "'",
            '‹': "'",
            '`': "'",
            ',—': ' ',              # Corelli_Marie_The-Sorrows-of-Satan_1895.txt
            '†': '',
            '<': "'", # Keller_Gottfried_Die-Leute-von-Seldwyla_1856.txt
            '>': "'", 
            '=>': '', # Sterne_Laurence_Tristram-Shandy_1759
            'align="left">': '', # Hays_Mary_Memoirs-of-Emma-Courtney_1796
            '[sic]': '',
            '(sic)': '',
            "{sic}": '', #Edgeworth_Maria_The-Manufacturers_1804
            '[*]': '',
            '[+]': '',
            '[#]': '', # Blackmore_R-D_Clara-Vaughan_1864
            '[**]': '', # Fielding_Henry_Tom-Jones_1749
            '[A]': '',
            '[1]': '', 
            'Ms.': 'Ms',
            'ms.': 'ms', 
            'Mr.': 'Mr',
            'mr.': 'mr',
            'Mrs.': 'Mrs',
            'mrs.': 'mrs',
            'M.D.': 'MD',
            'm.d.': 'md',
            'Dr.': 'Dr', # Trollope_Anthony_Barchester-Towers_1857.txt
            'dr.': 'dr',
            'Prof.': 'Prof',
            'prof.': 'prof',
            'St.': 'St', # Doyle_Arthur-Conan_The-Man-with-the-Twisted-Lip_1891
            'st.': 'st',
            'Ave.': 'Ave',
            'ave.': 'ave',
            'Rd.': 'Rd',
            'rd.': 'rd',
            'Jr.': 'Jr',
            'jr.': 'jr',
            'Sr.': 'Sr',
            'sr.': 'sr',
            'Ltd.': 'Ltd',
            'ltd.': 'ltd',
            'Bros.': 'Bros',
            'bros.': 'bros',
            'A.M.': 'AM',
            'P.M.': 'PM',
            'Co.': 'Co',
            'co.': 'co',
            'Jan.': 'Jan',
            'Feb.': 'Feb',
            'Mar.': 'Mar',
            'Apr.': 'Apr',
            'May': 'May',
            'Jun.': 'Jun',
            'Jul.': 'Jul',
            'Aug.': 'Aug',
            'Sep.': 'Sep',
            'Oct.': 'Oct',
            'Nov.': 'Nov',
            'Dec.': 'Dec',
            'jan.': 'jan',
            'feb.': 'feb',
            'mar.': 'mar',
            'apr.': 'apr',
            'may': 'may',
            'jun.': 'jun',
            'jul.': 'jul',
            'aug.': 'aug',
            'sep.': 'sep',
            'oct.': 'oct',
            'nov.': 'nov',
            'dec.': 'dec',
        }
        return rep_dict


    def hash_text(self, text):
        hash_object = hashlib.md5(text.encode())
        return hash_object.hexdigest()

    def simple_replace(self, text, to_replace, replace_with):
        with open('test', 'w') as f:
            f.write(text)
        new_text = text.replace(to_replace, replace_with)
        assert self.hash_text(text) != self.hash_text(new_text), f"{self.doc_path},{to_replace},{replace_with}"
        return new_text
    

    def get_context(self, text, start, end, stride=20):
        start = max(0, start - stride)
        end = min(len(text), end + stride)
        context = text[start:end]
        return context

    def rep_callable(self, match, replace_with, text):
        context = self.get_context(text, match.start(), match.end())
        removed_text = text[match.start():match.end()+1] # the part of the text that matches the pattern
        reptup = (match.start(), match.end(), removed_text, context)
        return replace_with, reptup

    def regex_rep_checks(self, rep_overview, new_text, pattern, num_replacements):
        rep_overview_empty = all(not tpl for tpl in rep_overview) # Check if rep_overview only contains empty tuples
        if not rep_overview_empty:
            rep_overview = [(num_replacements, x[0], x[1], x[2], x[3], self.get_context(new_text, x[0], x[1]), pattern, self.regex_rep_counter, self.bookname) for x in rep_overview]
            df = pd.DataFrame(rep_overview, columns=['num_replacements', 'start', 'end', 'removed_text', 'rep_context', 'new_context', 'pattern', 'rep_counter', 'doc_path'])
            df.insert(0, 'length', df['end'] - df['start'])
            df = df.drop(columns=['start', 'end'])
            df = df[(df['length'] > 80)] # | (df['num_replacements'] > 20)]
            df = df.sort_values(by=['rep_counter', 'length', 'num_replacements'], ascending=[True, False, False])
            if not df.empty:
                self.add_subdir('regex_checks')
                file_path = self.get_file_path(file_name=self.bookname, subdir=True)
                file_exists = os.path.exists(file_path)
                header = not file_exists
                df.to_csv(file_path, sep=self.separator, mode='a', index=False, header=header)
                self.regex_rep_counter += 1


    def regex_replace(self, necessary, check, pattern, replace_with, text, flags=0):
        '''
        necessary: replacement has to be made
        check: check replacement and write check to file
        '''
        rep_overview = []

        def replace_lambda(match):
            rep_with, reptup = self.rep_callable(match, replace_with, text)
            rep_overview.append(reptup)  
            return rep_with
        
        if check:
            new_text, num_replacements = re.subn(pattern=pattern, repl=replace_lambda, string=text, flags=flags)
        else:
            new_text = re.sub(pattern=pattern, repl=replace_with, string=text, flags=flags)
            num_replacements = None

        if necessary:
            assert num_replacements != 0, f"{self.doc_path},{pattern},{replace_with}"

        self.regex_rep_checks(rep_overview, new_text, pattern, num_replacements)
        return new_text


    def preprocess_individual_files(self, text):
        if 'Hyan_Hans_1000-Mark-Belohnung_1913' in self.doc_path:
            pattern = r'\b\w+\s*=\s*\w+\b'
            text = self.regex_replace(True, True, pattern, '', text)
        if 'Nesbit_Edith_The-Wouldbegoods_1901' in self.doc_path:
            text = self.regex_replace(True, True, r'Note .\. \(See Note .\.\)', '', text, flags=re.DOTALL)
            text = self.regex_replace(True, True, r'\(See Note .\.\)', '', text, flags=re.DOTALL)

        if 'Rosegger_Peter_Die-Schriften-des-Waldschulmeister_1875' in self.doc_path:
            pattern = r'Hier scheint ein Irrtum.*?\(Der Herausgeber\)'
            text = self.regex_replace(True, True, pattern, '', text, flags=re.DOTALL)
        
        if 'Edgeworth_Maria_Lame-Jervas_1804' in self.doc_path:
            # One footnote seems to be by the author, while the others are by the editor
            # Remove only those by the editor
            text = self.simple_replace(text, '{Footnote: Extracts from William Smith', 'Extracts from William Smith')
            text = self.simple_replace(text, r"such terms as the sultan did not understand.'}", r"such terms as the sultan did not understand.'")

        if 'Marlitt_Eugenie_Im-Hause-des-Kommerzienrates_1877' in self.doc_path:
            text = self.simple_replace(text, '28[.]', '')
            # Find numbers enclosed by brackets.
            text = self.regex_replace(True, True, r'\[\d*?\]', '', text)
            
        if 'Wells_H-G_A-Modern-Utopia_1905' in self.doc_path:
            text = self.simple_replace(text, '(See also Chapter I., § 6, and Chapter X., §§ 1 and 2.)', '')
            text = self.regex_replace(True, True, r'\[Footnote: See.*?\]', '', text, flags=re.DOTALL)
            text = self.simple_replace(text, 'Footnote: ', '')
            # text = self.regex_replace(True, True, r'\[.*?\]', '', text, flags=re.DOTALL) # not removed, seem to be by author

        if 'Porter_Jane_Thaddeus-of-Warsaw_1803' in self.doc_path:
            text = self.simple_replace(text, 'Footnote: ', '')
            text = self.regex_replace(True, True, r'\[Illustration:.*?\]', '', text)

        if 'Radcliffe_Ann_Udolpho_1794' in self.doc_path:
            text = self.regex_replace(True, True, r'\(\*Note:.*?\)', '', text, flags=re.DOTALL)

        # Not removed, notes by author
        # if 'Scott_Walter_Old-Mortality_1816' in self.doc_path:
        #     # Insert missing bracket 
        #     text = self.simple_replace(text, "[COMMANDER-IN-CHIEF OF KING CHARLES II.'s FORCES IN SCOTLAND.]", "COMMANDER-IN-CHIEF OF KING CHARLES II.'s FORCES IN SCOTLAND.")
        #     text = self.simple_replace(text, 'spells like a chambermaid.', 'spells like a chambermaid.]')
        #     text = self.regex_replace(True, True, r'\[Note:.*?\]', '', text, flags=re.DOTALL)
        #     text = self.simple_replace(text, '[', '')
        #     text = self.simple_replace(text, ']', '')
            
        if 'OGrady_Standish_Early-Bardic-Literature_1879' in self.doc_path:
            text = self.simple_replace(text, "[Transcriber's Note: Greek in the original]", '')
            text = self.simple_replace(text, '[Note: "Dream of Angus," Révue Celtique, Vol. III., page 349.', '')
            text = self.simple_replace(text, '(see p. 257; vol. i.)', '')
            text = self.simple_replace(text, 'Note: Vol. I., page 155.', '')
            text = self.simple_replace(text, 'Note: Publications of Ossianic Society, Vol. I. of Oscar, on pages 34 and 35, Vol. I.', '') 
            # Match everything after "Note:" up to the next dot or newline character or end of line
            # This doesn't match everything because some notes have dots in them
            pattern = r'Note: (.*?)(?:\.|\n|$)'
            text = self.regex_replace(True, True, pattern, '', text, flags=re.DOTALL)            

        if 'Grand_Sarah_The-Heavenly-Twins_1893' in self.doc_path:
            text = self.simple_replace(text, '[she wrote]', 'she wrote')
            text = self.regex_replace(True, True, r'\[.*?\]', '', text, flags=re.DOTALL)


        if 'OGrady_Standish_The-Coming-of-Cuculain_1894' in self.doc_path:
            text = self.simple_replace(text, 'Tuatha=nations, De=gods, Danan=of Dana', '')
            text = self.regex_replace(True, True, r'Footnote: .*?\n', '', text)

        if 'Tieck_Ludwig_Aufruhr-in-den-Cevennen_1826' in self.doc_path:
            text = self.simple_replace(text, 'camisa = chemise, Bluse.', '')
            text = self.simple_replace(text, 'Leim = Lehm, Erde', '')
            text = self.simple_replace(text, 'Chancelant = uneinig, wankelmütig.', '')
            # text = self.regex_replace(True, True, r'\(.*?\)', '', text, flags=re.DOTALL), not removed, some seem to be by author


        if 'Reade_Charles_The-Cloister-and-the-Hearth_1861' in self.doc_path:
            text = self.regex_replace(True, True, r'\{.*?\}', '', text, flags=re.DOTALL)
            text = self.regex_replace(True, True, r'\(\d*?\)', '', text)
            text = self.simple_replace(text, '(', '')
            text = self.simple_replace(text, ')', '')

        if 'Fielding_Henry_Jonathan-Wilde_1742' in self.doc_path:
            text = self.simple_replace(text, '[he]', 'he')
            # text = self.regex_replace(True, True, r'\[.*?\]', '', text, flags=re.DOTALL) # Footnotes, not removed, probably by author
            
        if 'Haggard_H-Rider_Allan-Quartermain_1887' in self.doc_path:
            # Endnotes at the end of the text
            # long_pattern = r"""Endnote 1\n Among the Zulus a man assumes the ring.*Endnote 21"""
            # text = self.regex_replace(True, True, long_pattern, '', text, flags=re.DOTALL) # Not removed, seem to be by author
            # Remove endnotes inside the text
            text = self.regex_replace(True, True, r'Endnote \d*?(,| )', '', text)
            text = self.regex_replace(True, True, r'welded on to the steel Endnote 5', r'welded on to the steel', text)
            text = self.simple_replace(text, '{Endnote 15}', '')
            text = self.regex_replace(True, True, r'\[.*?\]', '', text, flags=re.DOTALL)
            text = self.simple_replace(text, 'Endnote 6', '')

        if 'Fielding_Henry_Amelia_1752' in self.doc_path:
            text = self.simple_replace(text, 'Containing the exordium, &c. ', '')
            long_pattern = r'{This chapter occurs.*?— ED.'
            text = self.regex_replace(True, True, long_pattern, '', text)
            text = self.simple_replace(text, 'physic.}', 'physic.')
            text = self.regex_replace(True, True, r'\{.*?\}', '', text, flags=re.DOTALL)

        if 'Scott_Walter_Waverley_1814' in self.doc_path:
            text = self.regex_replace(True, True, r'Footnote: See [Nn]ote \d+.?\n?', '', text)
            # Some notes are by author
            # text = self.regex_replace(True, True, r'\[Footnote.*?\]', '', text, flags=re.DOTALL|re.IGNORECASE)
            # Footnotes are not formatted consistently and cannot be properly removed using regex.
            # Remove everything up to the end of the line. This is probably too much.
            # text = self.regex_replace(True, True, r'Footnote.*?\n', '\n', text)

        if 'Scott_Walter_Guy-Mannering_1815' in self.doc_path:
            long_string = "[*We must again have recourse to the contribution to Blackwood's Magazine, April 1817 :—"
            text = self.simple_replace(text, long_string, '')
            text = self.simple_replace(text, 'Fare ye wee]', 'Fare ye wee')
            text = self.regex_replace(True, True, '\[.*?\]', '', text, flags=re.DOTALL)
            text = self.regex_replace(True, True, r'\*.*?(?=\s)', '', text)

        # Not removed, footnotes seem to be by author
        # if  'Scott_Walter_The-Monastery_1820.txt' in self.doc_path:
        #     text = self.regex_replace(True, True, r'\{Footnote.*?\}', '', text, flags=re.DOTALL)
        #     # Footnotes are not formatted consistently and cannot be properly removed using regex.
        #     # Remove everything up to the end of the line. This is probably too much.
        #     text = self.regex_replace(True, True, r'Footnote.*?\n', '\n', text)

        if 'MacDonald_George_David-Elginbrod_1863.txt' in self.doc_path:
            pattern = r'1\sch.*?of the World.'
            text = self.regex_replace(True, True, pattern, '', text, flags=re.DOTALL)


        if 'MacDonald_George_Robert-Falconer_1868' in self.doc_path:
            # pattern = r"""1:\[.*?Christ."'\]""" # Match notes at the end that are probably by author
            # text = self.regex_replace(True, True, pattern, '', text, flags=re.DOTALL)
            text = self.simple_replace(text, 'tÃªte-Ã -tÃªte', 'tête-à-tête')
            text = self.simple_replace(text, 'Ã†', 'e')
            text = self.simple_replace(text, 'Ã¦', 'e')
            text = self.regex_replace(True, True, r'\d+:', '', text)
            text = self.simple_replace(text, '[', '')
            text = self.simple_replace(text, ']', '')

        files_list = ['Edgeworth_Maria_The-Contrast_1804.txt', 
                          'Edgeworth_Maria_The-Will_1804',
                          'Edgeworth_Maria_The-Lottery_1804',
                          'Edgeworth_Maria_The-Manufacturers_1804']
        if any(file_name in self.doc_path for file_name in files_list):
            text = self.regex_replace(True, True, r'\{Footnote.*?\}', '', text, flags=re.DOTALL)

        curly_list = ['Edgeworth_Maria_The-Limerick-Gloves_1804',
                          'Hardy_Thomas_A-Pair-of-Blue-Eyes_1873',
                          'Wells_H-G_The-Island-of-Dr-Moreau_1896',
                          'Butler_Samuel_Erewhon_1872', 
                          ]
        # More texts with notes, not removed 
        #'Edgeworth_Maria_Patronage_1814', 'Edgeworth_Maria_The-Grateful-Negro_1804', 'Edgeworth_Maria_To-Morrow_1804',
        # Everything between curly brackets
        if any(file_name in self.doc_path for file_name in curly_list):
            text = self.regex_replace(True, True, r'\{.*?\}', '', text, flags=re.DOTALL)

        if 'Reynolds_George_The-Mysteries-of-London_1844' in self.doc_path:
            text = self.simple_replace(text, '}', '')
            text = self.simple_replace(text, '|', '')
            text = self.regex_replace(True, True, r'\[\d*?\]', '', text)

        if 'Haggard_H-Rider_She_1886' in self.doc_path:
            # pattern = r'<U+0391><U+039C><U+0395><U+039D><U+0391><U+03A1><U+03A4><.*<U+0395><U+03A5>S<U+0391><U+039C><U+0397><U+039D>'
            # text = self.regex_replace(True, True, pattern, '', text, flags=re.DOTALL)
            # pattern = r'<U+1F08>µe<U+03BD><U+1F71><U+03C1>ta<U+03C2>, t<U+03BF><U+1FE6> ß.*<U+03C8>e<U+03C5>s<U+1F71>µ<U+03B7><U+03BD>'
            # text = self.regex_replace(True, True, pattern, '', text, flags=re.DOTALL)
            # pattern = r' <U+039F><U+03A5><U+039A><U+0391><U+039D><U+0394><U+03A5><U+039D><U+0391><U+0399>.*<U+03B9><U+03BA><U+03C1><U+1F71>te<U+03B9> t<U+1FF7> pa<U+03B9>d<U+1F77>\.'
            # text = self.regex_replace(True, True, pattern, '', text, flags=re.DOTALL)
            # pattern = r'<U+03A4>O<U+039D>T<U+0395>O<U+039D><U+0391><U+039D><U+03A4><U+0399>S<U+03A4><U+0391><U+039D><U+03A4>O<U+039D><U+0395><U+03A0>.*<U+03B9><U+03BA><U+03C1><U+1F71>t<U+03B7><U+03C2> t<U+1FF7> pa<U+03B9>d<U+1F77>\.'
            # text = self.regex_replace(True, True, pattern, '', text, flags=re.DOTALL)
            # pattern = r'Quib~ fact~ iuravit <U+017F>e patre¯ tuu¯ quoq~ im¯ortale¯ o<U+017F>te¯<U+017F>ura.*Domini MCCCCLXXXXV°'
            # text = self.regex_replace(True, True, pattern, '', text, flags=re.DOTALL)
            pattern = r'Amenartas e.*i MCCCCLXXXXV°\.'
            text = self.regex_replace(True, True, pattern, '', text, flags=re.DOTALL)
            text = self.simple_replace(text, '{', '')
            text = self.simple_replace(text, '}', '')
            text = self.simple_replace(text, '[plate 1]', '')
            text = self.simple_replace(text, '[plate 2]', '')
            text = self.simple_replace(text, '[sketch omitted]', '')
            text = self.simple_replace(text, '[out]', '')

        if 'Scott_Walter_The-Fortunes-of-Nigel_1822' in self.doc_path:
            # text = self.regex_replace(True, True, r'\[Footnote.*?\]', '', text, flags=re.DOTALL) # Not removed, by author
            text = self.simple_replace(text, '=', '')
            text = self.simple_replace(text, '[', '')
            text = self.simple_replace(text, ']', '')

        if 'Pestalozzi_Johann_Lienhard-und-Gertrud_1779' in self.doc_path:
            pattern = r'Dieser Abschnitt.*?des Herausgebers'
            text = self.regex_replace(True, True, pattern, '', text, flags=re.DOTALL)

        if 'Scott_Walter_The-Betrothed_1825' in self.doc_path:
            text = self.simple_replace(text, '|east', 'least')
            # text = self.regex_replace(True, True, r'Footnote:', '', text, flags=re.DOTALL) # Not removed, seem to be by author

        # files_list = ['Edgeworth_Maria_Ormond_1817', 
        #                 'Edgeworth_Maria_Patronage_1814',
        #                   ]
        # if any(file_name in self.doc_path for file_name in files_list):
            text = self.regex_replace(True, True, r'\[Footnote.*?\]', '', text, flags=re.DOTALL)

        if 'MacDonald_George_Sir-Gibbie_1879' in self.doc_path:
            long_string = r'{compilers note:  spelled in Greek:  Theta, Epsilon, Omicron, Upsilon; Lambda, Omicron with stress, Gamma, Omicron, Sigma}'
            text = self.simple_replace(text, long_string, '')
            long_string = r'{compilers note:  spelled in Greek:  Tau, Upsilon with stress, Pi, Tau, Omega}'
            text = self.simple_replace(text, long_string, '')
            text = self.regex_replace(True, True, r'\[\d*?\]', '', text)
            text = self.regex_replace(True, True, r'{', '', text)
            text = self.regex_replace(True, True, r'}', '', text)

        files_list = ['Brunton_Mary_Discipline_1814',
                          'Dickens_Charles_Sketches-by-Boz_1833',
                          'Newman_John-Henry_Loss-and-Gain_1848',
                          'Fielding_Henry_Shamela_1741',
                          'Ferrier_Susan_Marriage_1818',
                          'Crockett_S-R_Cleg-Kelly_1896'
                          ]
        if any(file_name in self.doc_path for file_name in files_list):
            # Find numbers enclosed by brackets.
            text = self.regex_replace(True, True, r'\[\d*?\]', '', text)

        if 'Edgeworth_Maria_Orlandino_1848' in self.doc_path:
            text = self.regex_replace(True, True, r'\[\d*?\]', '', text)
            text = self.simple_replace(text, '[', '')
            text = self.simple_replace(text, ']', '')

        if 'Edgeworth_Maria_Castle-Rackrent_1800' in self.doc_path:
            text = self.simple_replace(text, ' [See GLOSSARY 11].]', '.')
            text = self.simple_replace(text, '[See GLOSSARY 28]]', '')
            text = self.regex_replace(True, True, r"\[See GLOSSARY \d+\]", '', text, flags=re.DOTALL)

        # Not removed, seem to be by author
        # if 'Scott_Walter_Kenilworth_1821' in self.doc_path:
        #     text = self.simple_replace(text, '[See Note 6]', '') # Brackets within brackets
        #     text = self.regex_replace(True, True, r'\[.*?\]', '', text, flags=re.DOTALL)

        if 'Sheehan_Patrick-Augustine_My-New-Curate_1900' in self.doc_path:
            #long_pattern = r' PRO MENSE AUGUSTO.\n (Die I^ma Mensis.)\n 1. Excerpta ex Statutis Dioecesanis et Nationalibus.\n 2. De Inspiratione Canonicorum Librorum.\n 3. Tractatus de Contractibus (Crolly).'
            long_pattern = r'PRO MENSE.*?\(Crolly\)'
            text = self.regex_replace(True, True, long_pattern, '', text, flags=re.DOTALL)

        if 'Edgeworth_Maria_The-Irish-Incognito_1802' in self.doc_path:
            text = self.simple_replace(text, 'right^', 'right')
            text = self.regex_replace(True, True, r'\[.*?\]', '', text, flags=re.DOTALL)

        if 'Kickham_Charles_Knockagow_1879' in self.doc_path:
            pattern = r'\* "Lan na.*?-- the favourite.'
            text = self.regex_replace(True, True, pattern, '', text, flags=re.DOTALL)


        files_list = [
            'Edgeworth_Maria_The-Modern-Griselda_1804',
            'Barrie_J-M_Peter-and-Wendy_1911',
            'Scott_Walter_The-Bride-of-Lammermoor_1819',
            'Scott_Walter_The-Peveril-of-the-Peak_1822.txt',
            'Roche_Regina-Maria_The-Children-of-the-Abbey_1796',
            'Kipling_Rudyard_Kim_1901',
            'Equiano_Olaudah_Life-of-Equiano_1789',
            'Radcliffe_Ann_The-Italian_1797',
            'Haggard_H-Rider_King-Solomons-Mines_1885',
            'MacDonald_George_Alec-Forbes-of-Howglen_1865',
            'Reade_Charles_It-Is-Never-Too-Late-to-Mend_1856',
            'Maurier_George-du_Trilby_1894',
            'Gore_Catherine_Theresa-Marchmont_1830',
            'Carleton_William_The-Black-Prophet_1847',
            'Nesbit_Edith_The-Story-of-the-Amulet_1906',
            'Bierbaum_Otto_Stilpe_1897',
            'Brentano_Clemens_Godwi_1801',
            'Freytag_Gustav_Die-Ahnen_1872',
            'Weerth_Georg_Fragment-eines-Romans_1845',
            'Alexis_Willibald_Isegrimm_1854',
            'Holz-Schlaf_Arno-Johannes_Papa-Hamlet_1889',
            'Bronte_Charlotte_Shirley_1849',
            'Butler_Samuel_Erewhon_1872',
            'Carlyle_Thomas_Sartor-Resartus_1834',
            'Conrad_Joseph_Typhoon_1903',
            'Hardy_Thomas_Jude-the-Obscure_1894',
            'Hogg_James_Private-Memoirs-and-Confessions-of-a-Justified-Sinner_1824',
            'Kipling_Rudyard_At-the-End-of-the-Passage_1890',
            'Moore_George_The-Untilled-Field_1903',
            'Porter_Jane_The-Scottish-Chiefs_1810',
            'Radcliffe_Ann_A-Sicilian-Romance_1790',
            'Swift_Jonathan_A-Tale-of-Tub_1704',
            'Thackerey_William-Makepeace_Vanity-Fair_1847',
            ]
        # 'Scott_Walter_The-Fair-Maid-of-Perth_1828', 'Scott_Walter_The-Black-Dwarf_1816', 'Scott_Walter_The-Antiquary_1816', 'Scott_Walter_Montrose_1819', Scott_Walter_Chronicles-of-the-Canongate_1827', 'Scott_Walter_The-Talisman_1825',(by author)
        # Mixed by author and not: 'Scott_Walter_The-Surgeons-Daughter_1827'
        if any(file_name in self.doc_path for file_name in files_list):
            # Everything in brackets
            text = self.regex_replace(True, True, r'\[.*?\]', '', text, flags=re.DOTALL)

        return text


    def get_chars_ok_list(self):
        chars_ok_list = [
                    'Lee_Vernon_Prince-Alberic-and-the-Snake-Lady_1895'
                    'Childers_Erskine_The-Riddle-of-the-Sand_1903', # not separable
                    'Doyle_Arthur-Conan_The-Hound-of-the-Baskervilles_1901',
                    'Forster_E-M_Howards-End_1910',
                    'Hardy_Thomas_Interlopers-at-the-Knap_1888',
                    'Reade_Charles_Hard-Cash_1863',
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
        return chars_ok_list



