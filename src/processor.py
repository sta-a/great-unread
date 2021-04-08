import os
import re
import logging
logging.basicConfig(level=logging.DEBUG)
import ssl
from tqdm import tqdm
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import nltk
nltk.download("punkt")
from unidecode import unidecode


class DocProcessor(object):
    def __init__(self, lang, processed_chunk_sentence_count=500, stride=500):
        self.lang = lang
        self.processed_chunk_sentence_count = processed_chunk_sentence_count
        self.stride = stride

        # if self.lang == "eng":
        #     self.model_name = 'en_core_web_md'
        # elif self.lang == "ger":
        #     self.model_name = 'de_core_news_md'
        # else:
        #     raise Exception(f"Not a valid language {self.lang}")
        
        # try:
        #     self.nlp = spacy.load(self.model_name)
        # except OSError:
        #     logging.info(f"Downloading {self.model_name} for Spacy...")
        #     os.system(f"python3 -m spacy download {self.model_name}")
        #     logging.info(f"Downloaded {self.model_name} for Spacy.")
        #     self.nlp = spacy.load(self.model_name)
    
    def process(self, doc_paths):
        logging.info("Processing texts...")
        if self.processed_chunk_sentence_count is not None:
            os.makedirs("/".join(doc_paths[0].split("/")[:-1]).replace("raw_docs", f"/processed_docs_sc_{self.processed_chunk_sentence_count}_st_{self.stride}"), exist_ok=True)
        else:
            os.makedirs("/".join(doc_paths[0].split("/")[:-1]).replace("raw_docs", f"/processed_docs_full"), exist_ok=True)

        for doc_path in tqdm(doc_paths):
            with open(doc_path, "r") as doc_reader:
                doc = doc_reader.read()
            
            def _process_text(text):
                text = text.lower()
                text = unidecode(text)
                text = re.sub("[^a-zA-Z]+", " ", text).strip()
                text = " ".join(text.split())
                return text

            if self.processed_chunk_sentence_count is not None:
                if self.lang == "eng":
                    sentences = [_process_text(sentence) for sentence in nltk.tokenize.sent_tokenize(doc, "english")]
                elif self.lang == "ger":
                    sentences = [_process_text(sentence) for sentence in nltk.tokenize.sent_tokenize(doc, "german")]
                else:
                    raise Exception(f"Not a valid language {self.lang}")
                
                for i in range(0, len(doc), self.stride):
                    current_chunk = sentences[i:i+self.processed_chunk_sentence_count]
                    if (len(current_chunk) < self.processed_chunk_sentence_count) and i != 0:
                        break
                    processed_doc_path = doc_path.replace("/raw_docs", f"/processed_docs_sc_{self.processed_chunk_sentence_count}_st_{self.stride}").replace(".txt", f"_pt_{i}.txt")
                    with open(processed_doc_path, "w") as doc_writer:
                        doc_writer.write(" ".join(current_chunk))
            else:
                doc = _process_text(doc)
                processed_doc_path = doc_path.replace("/raw_docs", "/processed_docs_full")
                with open(processed_doc_path, "w") as doc_writer:
                    doc_writer.write(doc)
        logging.info("Processed texts.")