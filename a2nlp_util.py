# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 22:51:59 2022

Python library containing the utility functions for Assignment 2 for
the course COMP-8730 for Winter 2022

@author: piranir
"""

# Uses a trained n-gram model and a spelling error corpus consisting of
# misspellings, actual word, and the phrase where this misspelling occurred
# to calculate the probability of predicting each word as a replacement
# for the misspelled word in the corpus.

# Returns a list of tuples containing the misspelling, expected word,
# phrase in split list form, and the probability of the prediction
# of the word based on the n-gram model.
def calc_word_probabilities_phrase(ngram_model, spelling_error_corpus):
    ngram_probability = list()
    
    phrase_list = [(term[1], term[-1]) for term in spelling_error_corpus]
    word_list = [term[1] for term in spelling_error_corpus]
    
    for phrase_group in phrase_list:
        for word in word_list:
            expected_word = phrase_group[0]
            phrase = phrase_group[1]
            prob = ngram_model.score(word, phrase)
            ngram_probability_tuple = (word, expected_word, ' '.join(phrase), prob)
            ngram_probability.append(ngram_probability_tuple)
    
    return ngram_probability