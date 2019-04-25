# Description
A dataset (614 KB on disk) of 404 transcripts is made available on canvas course page (Modules/Week3/transcripts.zip).
Write a program to gather information about word tokens in the sample database. You may use any programming language. 

## What to text processing steps you should do?
1. Remove stopwords ([using this file](stopwords.txt))
2. Remove special characters 
3. Use Porter or Snowball Stemming ([see python example](http://www.nltk.org/howto/stem.html))

## Use your program to generate the following information:
* The number of word tokens in the database (**before and after** text processing steps).
* The number of unique words in the database;
* The number of words that occur only once in the database;
* The average number of word tokens per document.
* For 30 most frequent words in the database, provide:
    - TF, scaled TF (1+log(tf), IDF, TF*IDF and  probabilities
    - in a tabular format (rows = terms, columns = values), see example table below.

### Table header
| Term |  Tf | Tf(weight) | df  | IDF | tf*idf | p(term) |
|------|:---:|:----------:|:---:|:---:|:------:|:--------|
