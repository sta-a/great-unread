import re
from utils import unidecode_custom


class Chunk(object):
    def __init__(self, doc_path, book_name, chunk_id, tokenized_sentences, sbert_sentence_embeddings, doc2vec_chunk_embedding):
        self.doc_path = doc_path
        self.book_name = book_name
        self.chunk_id = chunk_id
        self.tokenized_sentences = tokenized_sentences
        self.sbert_sentence_embeddings = sbert_sentence_embeddings
        self.raw_text = " ".join(tokenized_sentences)
        self.doc2vec_chunk_embedding = doc2vec_chunk_embedding
        self.unidecoded_raw_text = unidecode_custom(self.raw_text)
        self.processed_sentences = self.__preprocess_sentences()
        self.unigram_counts = self.__find_unigram_counts()
        self.bigram_counts = self.__find_bigram_counts()
        self.trigram_counts = self.__find_trigram_counts()
        self.char_unigram_counts = self.__find_char_unigram_counts()

    def __preprocess_sentences(self):
        def __preprocess_sentences_helper(text):
            text = text.lower()
            text = unidecode_custom(text)
            text = re.sub("[^a-zA-Z]+", " ", text).strip()
            text = text.split()
            text = " ".join(text)
            return text
        return [__preprocess_sentences_helper(sentence) for sentence in self.tokenized_sentences]

    def __find_unigram_counts(self):
        unigram_counts = {}
        for processed_sentence in self.processed_sentences:
            for unigram in processed_sentence.split():
                if unigram in unigram_counts.keys():
                    unigram_counts[unigram] += 1
                else:
                    unigram_counts[unigram] = 1
        return unigram_counts

    def __find_bigram_counts(self):
        processed_text = "<BOS> " + " <EOS> <BOS> ".join(self.processed_sentences) + " <EOS>"
        processed_text_split = processed_text.split()
        bigram_counts = {}
        for i in range(len(processed_text_split) - 1):
            current_bigram = processed_text_split[i] + " " + processed_text_split[i+1]
            if current_bigram in bigram_counts:
                bigram_counts[current_bigram] += 1
            else:
                bigram_counts[current_bigram] = 1
        return bigram_counts

    def __find_trigram_counts(self):
        processed_text = "<BOS> <BOS> " + " <EOS> <EOS> <BOS> <BOS> ".join(self.processed_sentences) + " <EOS> <EOS>"
        processed_text_split = processed_text.split()
        trigram_counts = {}
        for i in range(len(processed_text_split) - 2):
            current_trigram = processed_text_split[i] + " " + processed_text_split[i+1] + " " + processed_text_split[i+2]
            if current_trigram in trigram_counts.keys():
                trigram_counts[current_trigram] += 1
            else:
                trigram_counts[current_trigram] = 1
        return trigram_counts

    def __find_char_unigram_counts(self):
        char_unigram_counts = {}
        for character in self.unidecoded_raw_text:
            if character in char_unigram_counts.keys():
                char_unigram_counts[character] += 1
            else:
                char_unigram_counts[character] = 1
        return char_unigram_counts
