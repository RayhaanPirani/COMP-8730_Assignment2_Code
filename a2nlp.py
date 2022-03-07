# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 17:20:27 2022

Code for Performing Spelling Correction Using n-Gram Language Modelling 
for the course COMP-8730 for Winter 2022

@author: piranir
"""
# Step to download the Brown corpus manually if not already done
# import nltk
# nltk.download('brown') # Download Brown corpus
# nltk.download('punkt') # For tokenization

import datetime
import a2nlp_util
import numpy as np
import pandas as pd

SPELLING_ERROR_CORPUS_FILE = 'APPLING1DAT.643'

from nltk.corpus import brown
from nltk import word_tokenize, sent_tokenize

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE


# Read the sentences from the news genre of the Brown corpus, i.e., C
news_corpus_sents = list(brown.sents(categories='news'))
news_corpus_sents = [' '.join(sent) for sent in news_corpus_sents]
news_corpus = ' '.join(news_corpus_sents)

# Reading the Birkbeck spelling error corpus
with open(SPELLING_ERROR_CORPUS_FILE, 'r') as f:
    lines = f.read().splitlines()

lines = [word.lower() for word in lines]
lines_processed = list()

spelling_error_corpus = [(line.split()[0], line.split()[1],
                          ' '.join(line.split()[2:]))
                         for line in lines if line[0] != '$']
spelling_error_corpus = [[term[0], term[1], term[-1].replace('*', '').split()]
                         for term in spelling_error_corpus]

news_corpus_tokenized = [sent_tokenize(str.lower(sent))[0] for sent in news_corpus_sents]
news_corpus_tokenized = [word_tokenize(sent) for sent in news_corpus_tokenized]

# Tokenizing the news corpus
news_corpus = [list(map(str.lower, word_tokenize(sent))) 
                 for sent in sent_tokenize(news_corpus)]

# Constructing n-gram models where n = {1, 2, 3, 5, 10}
n_values = [1, 2, 3, 5, 10]
for n in n_values:
    # Preprocessing the tokenized news corpus
    train_news_corpus, padded_news_sents = padded_everygram_pipeline(n, news_corpus_tokenized)
    
    # Initializing an n-gram model
    ngram_model = MLE(n)
    
    # Fitting preprocessed corpus to the model
    ngram_model.fit(train_news_corpus, padded_news_sents)
    
    start_time = datetime.datetime.now()
    
    # Computing the minimum edit distance between misspelled words in C
    # and every word
    # spelling_error_word_med = pool.apply(a1nlp_util.calc_spelling_error_word_med, 
    #                                      args=(dictionary, spelling_error_corpus))
    ngram_probability = a2nlp_util.calc_word_probabilities_phrase(ngram_model, spelling_error_corpus)
    
    end_time = datetime.datetime.now()
    print("Time taken: " + str(end_time - start_time))
    
    # Converting processed data into a pandas DataFrame
    ngram_probability_df = pd.DataFrame(ngram_probability, columns = ['word',
                                                          'expected_word','phrase',
                                                          'probability'])
    
    # Selecting the top 10 words for every phrase
    top_ten_grouped = ngram_probability_df\
        .groupby(['phrase', 'expected_word'])\
        .apply(lambda x: x.nlargest(10, ['probability']))\
        .reset_index(drop=True)
    
    # Calculating success, i.e., if a predicted word matches the expected word
    top_ten_grouped['success'] = np.where(
        top_ten_grouped['word'] == top_ten_grouped['expected_word'], 1, 0)
    
    # Computing the success at k measure s@k
    top_ten_grouped['success_at_k'] = top_ten_grouped.groupby(
        ['phrase', 'expected_word'])['success'].transform(pd.Series.cumsum)
    
    # Selecting s@k for k = {1, 5, 10}
    s_at_k = top_ten_grouped.groupby(['phrase', 'expected_word'])\
        .nth([0,4,9])['success_at_k']
    
    # Computing the overall average and averages for s@k for k = {1, 5, 10}
    avg_s_at_k = s_at_k.groupby(['phrase', 'expected_word']).transform(np.mean)
    
    avg_s_at_k = avg_s_at_k.reset_index().drop_duplicates(ignore_index=True)
    
    avg_s_at_k.to_csv('avg_s_at_k.csv', index=False)
    
    s_at_1 = top_ten_grouped.groupby(['phrase', 'expected_word'])\
        .nth(0)['success_at_k']
    avg_s_at_1 = np.mean(s_at_1)
    
    s_at_5 = top_ten_grouped.groupby(['phrase', 'expected_word'])\
        .nth(4)['success_at_k']
    avg_s_at_5 = np.mean(s_at_5)
    
    s_at_10 = top_ten_grouped.groupby(['phrase', 'expected_word'])\
        .nth(9)['success_at_k']
    avg_s_at_10 = np.mean(s_at_10)
    
    # Summarizing the results
    avg_results = pd.DataFrame(
        [(avg_s_at_1, avg_s_at_5, avg_s_at_10)],
        columns=['Average s@1', 'Average s@5',
                 'Average s@10'])
    
    print("For n = " + str(n))
    print(avg_results)