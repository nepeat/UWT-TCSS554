import os
import sys
import logging
import operator
import json
import re
import math

from typing import List
from collections import defaultdict

from nltk.stem.snowball import SnowballStemmer

# Logging
if "DEBUG" in os.environ:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

# Constants, change with environ.
STOPWORDS_FILE = os.environ.get("STOPWORDS_FILE", "stopwords.txt")
TRANSCRIPT_LOCATION = os.environ.get("TRANSCRIPT_LOCATION", "propeitary/transcripts")

# Regexes
special_matcher = re.compile(r"(\W)")

class WordTokenizer:
    def __init__(self):
        self.log = logging.getLogger("tokenizer")
        self.stemmer = SnowballStemmer("english")

        self.words = defaultdict(lambda: 0)
        self.document_words = defaultdict(lambda: defaultdict(lambda: 0))
        self.documents = set()
        self.stopwords = set()

    # Stopword processing

    def load_stopwords(self):
        try:
            self._load_stopwords(STOPWORDS_FILE)
        except FileNotFoundError:
            # Try again but in the script location.
            try:
                self._load_stopwords(
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), STOPWORDS_FILE)
                )
            except FileNotFoundError:
                print(f"Stopwords file {STOPWORDS_FILE} does not exisst.")
                sys.exit(1)

    def _load_stopwords(self, filename: str):
        added = 0

        with open(filename) as f:
            for word in f:
                word = word.lower().strip()

                if not word:
                    continue

                self.stopwords.add(word)

                added += 1

        self.log.info("Loaded %d stopwords.", added)

    # File processing

    def load_file(self, filename: str):
        self.documents.add(filename)

        with open(filename, "r") as f:
            for line in f.readlines():
                # Nuke away capitalization and whitespace around the words.
                line = line.lower().strip()
                for word in line.split(" "):
                    self.document_words[filename][word] = 1
                    self.words[word] += 1

    def load_files(self):
        loaded = 0

        if os.path.exists(TRANSCRIPT_LOCATION):
            transcript_path = TRANSCRIPT_LOCATION
        else:
            transcript_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), TRANSCRIPT_LOCATION)

        for root, dirs, files in os.walk(transcript_path):
            for filename in files:
                if filename.endswith(".txt"):
                    self.load_file(os.path.join(root, filename))
                    loaded += 1

        if loaded == 0:
            self.log.warning("Loaded 0 files. Are there .txt files in the path '%s'?", transcript_path)
        else:
            self.log.info("Loaded %d files.", loaded)

    # Wordlist processing

    def stem_word(self, word: str) -> str:
        return self.stemmer.stem(word)

    def process_tokens(self, add_stems: bool=True):
        # Stem out the big word list.
        self._process_tokens(self.words, add_stems, print_stats=True)

        # Stem the document frequency.
        for wordlist in self.document_words.values():
            self._process_tokens(wordlist, add_stems)

    def _process_tokens(self, wordlist: dict, add_stems: bool=True, print_stats: bool=False):
        if print_stats:
            self.log.info("%d word tokens before filtering.", self.token_count)

        for word in wordlist.copy().keys():
            # Drop stopwords.
            if word.lower() in self.stopwords:
                del wordlist[word]
                continue

            # Merge special characters with normal words.
            if special_matcher.search(word):
                cleaned_word = special_matcher.sub("", word)

                # Update the counts for words that exist and are not stopwords.
                if (
                    cleaned_word and  # Ignore blank/null words.
                    cleaned_word.lower() not in self.stopwords # Ignore stopwords.
                ):
                    wordlist[cleaned_word] += wordlist[word]

                del wordlist[word]
                continue

        # Stem all words
        if not add_stems:
            return

        stemmed = defaultdict(lambda: 0)

        for word in wordlist.copy().keys():
            stemmed_word = self.stem_word(word)
            if (
                stemmed_word != word and # Ignore words where the stemmed version is the same as the original.
                stemmed_word not in self.stopwords # Ignore stopwords.
            ):
                stemmed[stemmed_word] += wordlist[word]

        for word, value in stemmed.items():
            wordlist[word] += value

        if print_stats:
            self.log.info("%d word tokens after filtering.", self.token_count)

    # Homework requirements.
    @property
    def token_count(self):
        return sum(self.words.values())

    def count_words(self, count=1):
        result = 0

        for value in self.words.values():
            if value == count:
                result += 1

        return result
    
    def print_most_frequent(self, limit=30):
        printed = 0

        os.makedirs("output", exist_ok=True)
        with open("output/words.csv", "w") as f:
            f.write(",".join(self.term_to_dict(None).keys()))
            f.write("\n")

            for _ in self.sorted_words:
                word, count = _
                self.log.debug("%s [%d times]", word, count)

                # Save to csv file.
                word_dict = self.term_to_dict(word)
                f.write(",".join(str(x) for x in word_dict.values()))
                f.write("\n")
                

                # Loop bounds
                printed += 1
                if printed >= limit:
                    break

    def print_requirement_stats(self):
        self.log.info(f"{len(self.words)} unique words in the database.")
        self.log.info(f"{self.count_words(1)} words that occur only once.")
        self.log.info(
            "%.2f words on average between %d documents.",
            (self.token_count / len(self.documents)) if self.documents else 0,
            len(self.documents)
        )
        self.print_most_frequent(30)
    
    @property
    def sorted_words(self):
        return sorted(self.words.items(), key=operator.itemgetter(1), reverse=True)

    def calculate_idf(self, term: str, df: float):
        if not self.words[term]:
            return 0.0

        return math.log(len(self.documents) / df)

    def calculate_df(self, term: str):
        result = 0

        for docterms in self.document_words.values():
            if term in docterms:
                result += 1
        
        return result

    def term_to_dict(self, term: str):
        # | Term |  Tf | Tf(weight) | df  | IDF | tf*idf | p(term) |
        if term:
            tf = self.words[term]
            df = self.calculate_df(term)
            idf = self.calculate_idf(term, df)

            tf_idf = tf * idf
            tf_weighted = (1 + math.log(tf))
            probability = tf / self.token_count
        else:
            tf, df, idf, tf_idf, tf_weighted, probability = (0, 0, 0, 0, 0, 0)

        return dict(
            term=term,
            tf=tf,
            tf_weighted=tf_weighted,
            df=df,
            idf=idf,
            tf_idf=tf_idf,
            probability=probability,
        )

    def run(self):
        self.load_stopwords()
        self.load_files()
        self.process_tokens()
        self.print_requirement_stats()
