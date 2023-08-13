# %%
import regex as re

separator = 'ƒ' # special separator that is not contained in to_replace strings

with open('simple-replacements.py') as f:
    original_code = f.read()

# Extract relevant parts using regular expressions
# pattern = r"if '([^']+)' in self.doc_path:\n\s+text = text.replace\('([^']+)', ('[^']+'|'')\)"
filenames_pattern = r"if ('[^']+'|'')"
pattern = r"('[^']+'|'')"


filenames = re.findall(filenames_pattern, original_code)
filenames = [x.replace("'", '') for x in filenames]


unique_set = set(filenames)
is_unique = len(unique_set) == len(filenames)
print("Contains only unique values:", is_unique)
duplicated_values = [item for item in unique_set if filenames.count(item) > 1]
print("Duplicated values:", duplicated_values)

matches = re.findall(pattern, original_code)
matches = [x.replace("'", '') for x in matches if x]

def is_odd(number):
    return number % 2 == 1

with open('simple-replacements.txt', 'w') as f:
    # Add priority to ensure replacement order
    for i in range(0, len(matches)-1):
        if matches[i] in filenames:
            fn = matches[i]
            i_fn = i
            priority = 0 
        elif is_odd(i-i_fn):
            f.write(f'{fn}{separator}{matches[i]}{separator}{matches[i+1]}{separator}{priority}\n')
            priority+=1
# %%
