# %%
# Preprocess from Deutsches Textarchiv
# https://www.deutschestextarchiv.de/book/show/ebers_koenigstochter01_1864
import os
import re
from spellchecker import SpellChecker

data_dir = '../../data/manually_corrected_texts'
input_dir = os.path.join(data_dir, 'koenigstochter')
output_dir = os.path.join(data_dir, 'ger')
file_name = 'Ebers_George_Eine-aegyptische-Koenigstocher_1864.txt'

def replace_multiple(text, rep_dict):
    # https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string
    rep = dict((re.escape(k), v) for k, v in rep_dict.items()) 
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    return text

def find_misspelled_words(text):
    # 'I' is often spelled as 'J'
    pattern = r'\b\w*J\w*\b'
    # Find all words with a J
    wordlist = set(re.findall(pattern, text))
    # Write the words to a file
    output_file_path = 'words_with_j.txt'
    with open(output_file_path, 'w') as output_file:
        for word in wordlist:
            output_file.write(word + '\n')


    spell = SpellChecker(language="de")  # Deutsch als Sprache einstellen
    # Finde Schreibfehler und schlage mögliche Korrekturen vor
    misspelled = spell.unknown(wordlist)
    print(misspelled)
    correctly_spelled = [x for x in wordlist if x not in misspelled]
    assert len(misspelled) + len(correctly_spelled) == len(wordlist)
    with open('j-correct.txt', 'w') as f:
        for word in misspelled:
            candidates = spell.candidates(word)
            if candidates is not None:
                candidates = ','.join(candidates)
                final_string = f"'{word}: '{candidates}'\n"
                print(final_string)
                f.write(final_string)


def preprocess_text(text):
    text = text.replace('\x0C', '') # Form Feed character
    text = text.replace('ſ', 's')

    # Remove everything before content
    pattern = r'.*?(?=Erstes Kapitel\.)'
    text = re.sub(pattern, '', text, flags=re.DOTALL, count=1)


    # Remove headers before paragraphs
    pattern = r'Ebers, Eine ägyptische Königstochter.*?$'
    text = re.sub(pattern, '', text, flags=re.MULTILINE)

    # Remove brackets
    pattern = r'\[[\d/]+\]'
    text = re.sub(pattern, '', text)
    pattern = r'\[/\d+\]'
    text = re.sub(pattern, '', text)

    # Remove notes
    pattern = r'Anmerkungen\..*$' #Everything from "Anmerkungen." up to the end of the text.
    text = re.sub(pattern, '', text, flags=re.DOTALL)

    placeholder = '='*100
    text = text.replace('\n\n\n\n', placeholder)
    text = text.replace('\n\n\n', placeholder)
    text = text.replace('\n\n', placeholder)
    text = text.replace('\n', ' ')
    text = text.replace(placeholder, '\n')

    # Remove hyphens
    pattern = r'(?<=[a-zA-Z])-\s(?=[a-zA-Z])'
    text = re.sub(pattern, '', text)

    # Remove annotations that have a '*' at the beginning of the line.
    pattern = r'^\*.*?$'
    text = re.sub(pattern, '', text, flags=re.MULTILINE)

    text = text.replace(' ***)', '')
    text = text.replace('***)', '')
    text = text.replace(' **)', '')
    text = text.replace('**)', '')
    text = text.replace(' *)', '')
    text = text.replace('*)', '')

    text = text.replace('Jntaphernes', 'Intaphernes')
    text = text.replace('Jxabates', 'Ixabates')

    pattern = r'\bJhr\b'
    replacement = 'Ihr'
    text = re.sub(pattern, replacement, text)

    pattern = r'\bJhre\b'
    replacement = 'Ihre'
    text = re.sub(pattern, replacement, text)

    pattern = r'\bJch\b'
    replacement = 'Ich'
    text = re.sub(pattern, replacement, text)

    pattern = r'\bJn\b'
    replacement = 'In'
    text = re.sub(pattern, replacement, text)

    pattern = r'\bJrr\b'
    replacement = 'Irr'
    text = re.sub(pattern, replacement, text)

    # Remove numbers followed by ')'
    pattern = r'\d+\)'
    text = re.sub(pattern, '', text)


    return text

def process_files_in_directory(directory_path):
    preprocessed_texts = []

    # Iterate through files in the directory
    filenames = ['ebers_koenigstochter01_1864.txt', 'ebers_koenigstochter02_1864.txt', 'ebers_koenigstochter03_1864.txt']
    for filename in filenames:
        print(filename)
        file_path = os.path.join(directory_path, filename)

        # Check if the path is a file (not a directory)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
                preprocessed_content = preprocess_text(content)
                preprocessed_texts.append(preprocessed_content)

    # Join preprocessed texts into one
    combined_text = '\n'.join(preprocessed_texts)
    return combined_text

# Directory containing the text files

# Process files in the directory
text = process_files_in_directory(input_dir)


j_list = ["Jadmon", "Jberien", "Jbis", "Jbisschnäbeln", "Jbykos", "Jbykus", "Jdentität", "Jhm", "Jhnen", "Jkaria", "Jllumination", "Jm", "Jmbiß", "Jmmer", "Jmmerhin", "Jnbegriff", "Jndem", "Jndern", "Jndessen", "Jndien", "Jndusmündung", "Jnfanterie", "Jnfanteristen", "Jngrimm", "Jnhalt", "Jnhalte", "Jnhalts", "Jnmitten", "Jnnere", "Jnneres", "Jnnern", "Jnnigkeit", "Jnschrift", "Jnschriften", "Jnsecten", "Jnsel", "Jnselbewohner", "Jnselfürst", "Jnseln", "Jnstinkt", "Jnstinkte", "Jnstrument", "Jnstrumente", "Jnstrumenten", "Jnstrumentes", "Jnteressen", "Jonier", "Jonierin", "Joniern", "Jonischen", "Jpsus", "Jrene", "Jrisstrom", "Jrrfahrten", "Jrrthum", "Jrrthümer", "Jrrthümern", "Jrrungen", "Jrrwahn", "Jrrwahne", "Jsis", "Jsisstern", "Jsrael", "Jsraelit", "Jsraeliten", "Jsraels", "Jst", "Jsthmos", "Jtalien", "Jto", "Jtys"]
j_dict = {}
for item in j_list:
    j_dict[item] = item.replace('J', 'I')

text = replace_multiple(text, j_dict)



# 'I' is often spelled as 'J'
pattern = r'\b\w*J\w*\b'
# Find all words with a J
wordlist = set(re.findall(pattern, text))
# Write the words to a file
output_file_path = 'new_words_with_j.txt'
with open(output_file_path, 'w') as output_file:
    for word in wordlist:
        output_file.write(word + '\n')




with open(os.path.join(output_dir, file_name), 'w') as f:
    f.write(text)


# %%
