"""
Matt Shadish

This script defines several utility functions

1. tsvRead(tsv_file)
    - takes in a path to a tab-separated file
    - (presumably training data for the rotten tomatoes reviews)
    - returns a dictionary that maps each phrase to its given score
    
2. testPhraseSplit(test_phrase)
    - takes in any string
    - returns a dictionary of all of the possible sub-phrases
    - as well as each sub-phrase's start and stop indices
    
3. splitTrainHoldout(full data set, split_fraction = 0.1)
    - takes in a dictionary of full training data
    - that maps each phrase to its score
    - and randomly pulls out a holdout set, a list of just the phrases
    - (size determined by the split fraction)
    
4. computeAccuracy(predictions, full data set)
    - takes in 2 dictionaries:
    - the 1st maps each test phrase to our predicted sentiment
    - the 2nd is the full training set that maps every phrase to sentiment
    
5. computeOverPredict(predictions, full data set)
    - similar to computeAccuracy
    - except this function returns the ratio of over-guesses
    - to the total number of predictions we got wrong
    
6. computeWeightedScore(list of scores, list of phrase lengths)
    - takes in a list of matched scores
    - and a list of corresponding phrase lengths
    - computes a single, weighted score
"""

import re
import random
import time
import argparse
import sys
import numpy as np


def tsvRead(tsv_file):
    """
    Takes in a path to the tsv file (presumably of training data)
    
    Returns a mapping dictionary of training phrases to sentiment scores
    """
    # open the file
    infile = open(tsv_file, 'r')
    
    mapping_dict = {}
    
    for line in infile:
        # skip the first row
        if re.match(r'[a-zA-Z]', line):
            continue
        
        # remove newlines
        line = re.sub('\n', '', line)
        
        # split into four columns
        # and grab the 3rd (phrase) and 4th (score) columns
        line_columns = re.split('\t', line)
        mapping_dict[line_columns[2]] = line_columns[3]
        
    return mapping_dict


    
def testPhraseSplit(test_phrase):
    """
    Takes in a test phrase
    
    Returns a list of all of the possible sub-phrases
    that can be made by that test_phrase
    """
    # split the phrase by space
    phrase_split = re.split(r'\s', test_phrase)
    
    # note that we will need to keep track of the start and stop indices
    # of each sub-phrase, so we will use those as keys in our dict
    possible_phrases = {}
    
    # loop through all possible phrases
    phrase_length = len(phrase_split)
    
    for start_index in xrange(phrase_length):
        for stop_index in xrange(start_index + 1, phrase_length + 1):
            
            # add it to our list
            sub_phrase = ' '.join(phrase_split[start_index:stop_index])
            possible_phrases[sub_phrase] = (start_index, stop_index)
            
    return possible_phrases
    
    

def splitTrainHoldout(full_mapping_dict, split_fraction = 0.1):
    """
    Takes in a mapping dictionary of training data
    
    Returns a holdout set of the training data based on the split fraction
    Also returns a copy of the mapping dictionary
    with the remaining training data, minus the holdout set
    """
    # set the seed
    random.seed(time.time())
    
    # grab the holdout set
    sample_size = int(round(split_fraction * len(full_mapping_dict.keys())))
    holdout_set = random.sample(full_mapping_dict, sample_size)
    
    # create a copy of the full mapping_dict
    mapping_dict = full_mapping_dict.copy()
    
    # remove the holdout data from the mapping dictionary
    for phrase in holdout_set:
        del mapping_dict[phrase]
        
    return holdout_set, mapping_dict
    
    
    
def computeAccuracy(predicted_scores, full_mapping_dict):
    """
    This takes in a dictionary with our predictions on a validation set
    as well as the full dictionary
    
    And tests the accuracy of our predictions
    Returns the ratio of accurate predictions to total predictions
    """
    # compute the numerator
    numerator = 0.0
    
    # compare with the actual values
    for phrase in predicted_scores:
        if predicted_scores[phrase] == int(full_mapping_dict[phrase]):
            numerator += 1
            
    # compute denominator
    denominator = len(predicted_scores)
    
    return numerator / denominator
    
    
    
def computeOverPredict(predicted_scores, full_mapping_dict):
    """
    Takes in a dictionary with our predictions on a validation set
    as well as a dictionary that maps the phrases to their true scores

    This function gives us a number that tells us whether the model
    a) over-guesses
    b) under-guesses
    c) evenly over- and under-guesses
    
    Returns the ratio of predictions that were over-guesses
    to the number of total incorrect predictions
    = (number of over-guesses) / (number of wrong predictions_)
    """
    # initialize the numerator and denominators
    over_guesses = 0.0
    total_wrong = 0.0
    
    # compare with the actual values
    for phrase in predicted_scores:
        # check if our prediction was correct or not
        if predicted_scores[phrase] != int(full_mapping_dict[phrase]):
            total_wrong += 1
            
            # did we over- or under-guess?
            if predicted_scores[phrase] > int(full_mapping_dict[phrase]):
                over_guesses += 1
            
    # return the ratio of over-guesses to incrorrect predictions    
    return over_guesses / total_wrong
    
    
    
def computeWeightedScore(list_scores, list_phrase_lengths):
    """
    Takes in a list of matched scores
    as well as a corresponding list of phrase lengths
    that all represent some part of a single phrase
    
    Computes a single weighted score
    """
    # first, check to make sure both the input lists are of same length
    # if this is not the case, there is an issue
    if len(list_scores) != len(list_phrase_lengths):
        print 'Input lists of different size'
        sys.exit()
        
    # compute the weighted, non-normalized scores
    weightings = np.exp(list_phrase_lengths)
    weighted_scores = np.multiply(list_scores, weightings)
    
    # normalize
    normalized_total = float(np.sum(weightings))
    weighted_scores = weighted_scores / normalized_total
    
    # return the weighted score
    return int(round(np.sum(weighted_scores)))
    
    
def commandLineIntake():
    """
    Parses arguments, if any, from the command line
    """
    # take in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-neutrals",
                        help = "remove/keep neutrals",
                        type = str)
    parser.add_argument('-iter',
                        help = 'How many iterations?',
                        type = float)
    parser.add_argument('-holdout',
                        help = 'Percent of training set aside for validation',
                        type = float)
                        
    args = parser.parse_args()
                        
    # initialize defaults
    # neutrals
    rm_neutrals = False
    # iterations
    iterations = 10.0
    # holdout set size, as a fraction
    holdout_size = 0.1
        
    if args.neutrals:
        if args.neutrals.lower() == 'remove':
            rm_neutrals = True
            
    if args.iter:
        iterations = args.iter
        
    if args.holdout:
        if args.holdout > 0 and args.holdout < 1:
            holdout_size = args.holdout
            
    return rm_neutrals, iterations, holdout_size