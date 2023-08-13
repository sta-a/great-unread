# %%
from difflib import SequenceMatcher
path = '/home/annina/Downloads/JCLS2022_Modeling-and-Predicting-Lit-Reception/corpora/ENG/Forrester_Andrew_The-Female-Detective_1864.txt'
path = '/home/annina/scripts/great_unread_nlp/data/corpus_corrections/manually_corrected_texts/eng/Forrester_Andrew_The-Female-Detective_1864.txt'

def split_text_by_words(text, words_per_chunk, window_size):
    words = text.split()
    chunks = [words[i:i + words_per_chunk] for i in range(0, len(words), window_size)]
    return [' '.join(chunk) for chunk in chunks]

def find_repeated_passages(text, words_per_chunk, window_size, similarity_threshold):
    passages = []
    chunks = split_text_by_words(text, words_per_chunk, window_size)
    num_chunks = len(chunks)

    for i in range(num_chunks - 1):
        window = chunks[i]

        for j in range(i + 1, num_chunks):
            candidate = chunks[j]

            similarity = SequenceMatcher(None, window, candidate).ratio()

            if similarity >= similarity_threshold:
                print(similarity, '\n', window, '\n', candidate, '='*100)
                passages.append((window, candidate, similarity))

    return passages

def main():
    words_per_chunk = 5000  # Adjust the words per chunk as needed
    window_size = int(words_per_chunk/2)
    similarity_threshold = 0.55  # Adjust the similarity threshold as needed

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    # text = """This is a short example text to demonstrate how the code works. It contains repeated passages to test the passage finding algorithm. The goal is to identify similar passages with a specified similarity threshold."""

    passages = find_repeated_passages(text, words_per_chunk, window_size, similarity_threshold)

    for passage in passages:
        window, candidate, similarity = passage

main()

# # %%
# path = '/home/annina/Downloads/JCLS2022_Modeling-and-Predicting-Lit-Reception/corpora/ENG/Forrester_Andrew_The-Female-Detective_1864.txt'

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import Levenshtein

# # Define the chunking function
# def split_text_by_words(text, words_per_chunk):
#     window_size = words_per_chunk
#     words = text.split()
#     chunks = [words[i:i + words_per_chunk] for i in range(0, len(words), window_size)]
#     return [' '.join(chunk) for chunk in chunks]

# # Load your book text
# book_text = open(path, 'r').read()

# # Split the raw text into chunks
# chunked_text = split_text_by_words(book_text, words_per_chunk=10000)

# # Compare chunks with Levenshtein distance
# similarity_threshold = 0.2
# overlapping_passages = []
# num_chunks = len(chunked_text)

# for i in range(num_chunks):
#     for j in range(num_chunks):
#         if i != j:
#             levenshtein_distance = Levenshtein.distance(chunked_text[i], chunked_text[j])
#             normalized_distance = 1 - (levenshtein_distance / max(len(chunked_text[i]), len(chunked_text[j])))
            
#             if normalized_distance < similarity_threshold:
#                 overlapping_passages.append((chunked_text[i], chunked_text[j]))


# for i in overlapping_passages:
#     print(i[0])
#     print(i[1])
#     print('='*50)

# # %%

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity


# # Load and preprocess texts
# def preprocess_text(text):
#     sentences = sent_tokenize(text.lower())
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for sent in sentences for word in word_tokenize(sent) if word.isalnum() and word not in stop_words]
#     return ' '.join(tokens)

# # Load your book texts
# book1_text = open('book1.txt', 'r').read()
# book2_text = open('book2.txt', 'r').read()

# # Preprocess texts
# book1_preprocessed = preprocess_text(book1_text)
# book2_preprocessed = preprocess_text(book2_text)

# # Vectorize and calculate cosine similarity
# vectorizer = TfidfVectorizer()
# book1_tfidf = vectorizer.fit_transform([book1_preprocessed])
# book2_tfidf = vectorizer.transform([book2_preprocessed])
# similarity_matrix = cosine_similarity(book1_tfidf, book2_tfidf)

# # Define similarity threshold
# similarity_threshold = 0.7

# # Find overlapping passages
# overlapping_passages = []
# for i in range(similarity_matrix.shape[0]):
#     for j in range(similarity_matrix.shape[1]):
#         if similarity_matrix[i, j] > similarity_threshold:
#             overlapping_passages.append((i, j))

# # Print overlapping passages
# for i, j in overlapping_passages:
#     print("Overlapping Passage in Book 1 (Index {}):".format(i))
#     print(sent_tokenize(book1_text)[i])
#     print("\nOverlapping Passage in Book 2 (Index {}):".format(j))
#     print(sent_tokenize(book2_text)[j])
#     print("=" * 50)


# %%
