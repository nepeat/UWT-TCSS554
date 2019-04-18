import os
import sys
import logging
import operator
import json
import re

from collections import defaultdict

# XXX: IMPLEMENT THIS
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

if "DEBUG" in os.environ:
    logging.basicConfig(level=logging.DEBUG)

STOPWORDS_FILE = os.environ.get("STOPWORDS_FILE", "stopwords.txt")
TRANSCRIPT_LOCATION = os.environ.get("TRANSCRIPT_LOCATION", "propeitary/transcripts")

special_matcher = re.compile(r"(\W)")

class WordTokenizer:
    def __init__(self):
        self.log = logging.getLogger("tokenizer")
        self.words = defaultdict(lambda: 0)
        self.stopwords = set()
    
    # Stopword processing

    def load_stopwords(self):
        try:
            self._load_stopwords()
        except FileNotFoundError:
            print(f"Stopwords file {STOPWORDS_FILE} does not exisst.")
            sys.exit(1)

    def _load_stopwords(self):
        with open(STOPWORDS_FILE) as f:
            for word in f:
                word = word.lower().strip()
                if not word:
                    continue
                self.stopwords.add(word)

        self.log.info("Loaded %d stopwords.", len(self.stopwords))

    # File processing

    def load_file(self, filename: str):
        with open(filename, "r") as f:
            for line in f.readlines():
                # Nuke away capitalization and whitespace around the words.
                line = line.lower().strip()
                for word in line.split(" "):
                    self.words[word] += 1

    def load_files(self):
        loaded = 0

        for root, dirs, files in os.walk(TRANSCRIPT_LOCATION):
            for filename in files:
                if filename.endswith(".txt"):
                    self.load_file(os.path.join(root, filename))
                    loaded += 1
        
        self.log.info("Loaded %d files.", loaded)

    # Wordlist processing

    def stem_word(self, word: str) -> str:
        # XXX: IMPLEMENT THIS
        return stemmer.stem(word)

    def clean_tokens(self):
        self.log.info("%d word tokens before filtering.", len(self.words))

        for word in self.words.copy().keys():
            # Drop stopwords.
            if word.lower() in self.stopwords:
                del self.words[word]
                continue

            # Merge special characters with normal words.
            if special_matcher.search(word):
                cleaned_word = special_matcher.sub("", word)

                # Update the counts for words that exist and are not stopwords.
                if cleaned_word and cleaned_word.lower() in self.stopwords:
                    self.words[cleaned_word] += self.words[word]

                del self.words[word]
                continue

        # Stem all words
        stemmed = defaultdict(lambda: 0)

        for word in self.words.copy().keys():
            stemmed_word = self.stem_word(word)
            if (stemmed_word != word) and stemmed_word not in self.stopwords:
                stemmed[stemmed_word] += self.words[word]

        for word, value in stemmed.items():
            self.words[word] += value

        self.log.info("%d word tokens after filtering.", len(self.words))

    def run(self):
        self.load_stopwords()
        self.load_files()
        self.clean_tokens()
        with open("words.csv", "w") as f:
            f.write("word,count\n")
            for word in sorted(self.words.items(), key=operator.itemgetter(1), reverse=True):
                f.write(f"{word[0]},{word[1]}\n")