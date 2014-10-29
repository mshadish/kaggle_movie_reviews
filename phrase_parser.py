# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 10:33:06 2014

@author: mshadish

This script defines several functions in order to
1. Create a "mapping" dictionary of phrases and their sentiments
2. Break apart each incoming test phrase into all of its possible sub-phrases
3. Match these sub-phrases to our mapping table
4. Compute a sentiment score for the given test phrase
"""

import re
import random
import time
import multiprocessing as mp
import sys
import numpy as np
import argparse


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
    
    
def splitTrainHoldout(full_mapping_dict, split_fraction = 0.05):
    """
    Takes in a mapping dictionary of training data
    
    Returns a holdout set of the training data based on the split fraction
    Also returns a copy of the mapping dictionary
    with the remaining training data, minus the holdout set
    """
    # grab the holdout set
    sample_size = int(round(split_fraction * len(full_mapping_dict.keys())))
    holdout_set = random.sample(full_mapping_dict, sample_size)
    
    # create a copy of the full mapping_dict
    mapping_dict = full_mapping_dict.copy()
    
    """
    # create a copy of the mapping dict for validation purposes
    validation_mapping = {key: full_mapping_dict[key]
                            for key in full_mapping_dict
                            if key not in holdout_set}
    """
    
    # remove the holdout data from the mapping dictionary
    for phrase in holdout_set:
        del mapping_dict[phrase]
        
    return holdout_set, mapping_dict
    
    
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
    
    
def joinWithMapping(mapping_dict, possible_phrases):
    """
    Takes in a mapping dict where key:value == phrase:score
    as well as a dictionary of potential phrases
    where key:value == sub-phrase:(start_index, stop_index)
    
    Returns a dictionary of the remaining possible phrases
    that matched with our mapping dict as keys
    and the associated sentiment score
    
    So key:value == sub-phrase:score
    
    Note that to avoid double-counting, we filter out any matched sub-phrases
    that are fully 'eclipsed' by matched sub-phrases
    ex/ suppose we start with the phrase 'this is a test of our function'
    and we have three matched sub-phrases,
    'test of our function', 'test', and 'function'
    -- we would filter out 'test' and 'function' so we are only left with
    'test of our function'
    """
    # first, grab the phrases that matched what we have in our mapping
    matched_phrases = set(possible_phrases).intersection(mapping_dict)
    match_dict = {key: possible_phrases[key] for key in matched_phrases}
    
    # now filter out any of the 'eclipsed' sub-phrases
    for phrase_1, indices_1 in match_dict.items():
        for phrase_2, indices_2 in match_dict.items():
            
            # skip if comparing with itself
            if phrase_1 == phrase_2:
                continue
            
            # compare start indices
            if indices_1[0] <= indices_2[0]:
                # compare stop indices
                if indices_1[1] >= indices_2[1]:
                    # this means our phrase_2 is a sub-phrase of phrase_1
                    del match_dict[phrase_2]
                    
    # end loops
    try:
        return_dict = {key: mapping_dict[key] for key in match_dict}
    except TypeError:
        print match_dict
        print type(mapping_dict)
        sys.exit()
    return return_dict
    
    
def randomBaseline(holdout_set):
    """
    Assigns random scores 0-4 to our holdout set
    
    This can help us establish a baseline
    """
    # set seed
    random.seed(time.time())
    
    predicted_scores = {phrase: random.randint(0,4) for phrase in holdout_set}
    
    return predicted_scores
    
    
def method1(mapping_dict, holdout_set, remove_neutrals = False):
    """
    Method 1:
    Average all of the scores from the matched sub-phrases
    for a given phrase from our holdout set
    
    Straight average, no weights applied
    
    Results: Can achieve approximately 56% to 59% accuracy
    Increases to 60% accuracy if we simply remove all of the neutral phrases
    """
    # we will store our results in a dictionary
    predicted_scores = {}
    
    count = 0

    for phrase in holdout_set:
        
        # split into all possible phrases
        sub_phrases = testPhraseSplit(phrase)
        
        # join with the mapping dictionary
        matches = joinWithMapping(mapping_dict, sub_phrases)
        
        # if we want to remove neutrals, do so here
        if remove_neutrals:
            matches = {key: float(matches[key]) for key in matches
                        if int(matches[key]) != 2}
        
        # if nothing is matched, default assignment is a neutral score
        if len(matches) == 0:
            predicted_scores[phrase] = 2
            continue
        
        # otherwise, compute the unweighted average
        scores = [float(num) for num in matches.values()]
        
        # and add to our scoring dictionary
        predicted_scores[phrase] = int(round(np.mean(scores)))
        
        """
        # to keep track of progress
        count += 1
        print 'completed: ' + str(count)
        """
                
    return predicted_scores
    
    
def method2(mapping_dict, holdout_set, remove_neutrals = False):
    """
    Method 2:
    Compute a weighted average score for each phrase
    Weightings are calculated based on sub-phrase length
    compared to the length of the entire phrase
    
    Also around 60% accuracy when we remove neutrals
    """
    # we will store our results in a dictionary
    predicted_scores = {}
    
    count = 0
    none_matched = 0
    
    for phrase in holdout_set:
        
        # grab the phrase length with which we will calculate our weighted avg
        phrase_length = float(len(re.split(r'\s', phrase)))
        
        # split into all possible phrases
        sub_phrases = testPhraseSplit(phrase)
        
        # join with the mapping dictionary
        matches = joinWithMapping(mapping_dict, sub_phrases)
        
        # if we want to remove neutrals, do so here
        if remove_neutrals:
            matches = {key: float(matches[key]) for key in matches
                        if int(matches[key]) != 0}
        
        # if nothing is matched, default assignment is a neutral score
        if len(matches) == 0:
            predicted_scores[phrase] = 0
            continue
                
        # bag of matched phrases
        bag_of_matches = ' '.join(matches.keys())
        bag_size = float(len(re.split(r'\s', bag_of_matches)))
        
        # otherwise, compute the weighted scores
        scores = []
        for matched_phrase in matches:
            
            # compute the length of our matched phrase
            matched_phrase_length = float(len(re.split(r'\s', matched_phrase)))
            
            # grab the score for the phrase
            num = float(matches[matched_phrase])

            # weight it
            num = num * matched_phrase_length / bag_size
            
            # add it to our list of scores
            scores.append(num)
        
        # and add our weighted average to the scoring dictionary
        predicted_scores[phrase] = int(round(np.sum(scores)))
        
        """
        # to keep track of progress
        count += 1
        print 'completed: ' + str(count)
        """
                
    return predicted_scores
    
    
def computeAccuracy(predicted_scores, full_mapping_dict):
    """
    This takes in a dictionary with our predictions on a validation set
    as well as the full dictionary
    
    And tests the accuracy of our predictions
    Returns a ratio
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
    
    
def testingWrapper(full_map_dict, size, weighted_scores = False,
                   remove_neutrals = False):
    """
    Runs a single iteration of
    1. generating a holdout set
    2. computing the model score
    
    Note: if we want weighted scores, we must shift the scores down by 2
    such that 0 is neutral, -2 is negative, and +2 is positive
    """
    # if we're using weighted scoring, we must convert the dict values
    # from strings to floats
    if weighted_scores:
        full_map_dict = {i: int(full_map_dict[i]) - 2 for i in full_map_dict}

    holdout_set, mapping_minus_holdout = splitTrainHoldout(full_map_dict, size)
    predictions = None
    if weighted_scores:
        predictions = method2(mapping_minus_holdout, holdout_set,
                              remove_neutrals)
    else:
        predictions = method1(mapping_minus_holdout, holdout_set,
                              remove_neutrals)
    accuracy = computeAccuracy(predictions, full_map_dict)
    
    return accuracy
    
    
def main():
    # take in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-method",
                        help = "method1 (unweighted avgs) or method2 (weighted)",
                        type = str)
    parser.add_argument("-neutrals",
                        help = "Remove neutrals, T or F",
                        type = str)
    parser.add_argument('-iter',
                        help = 'How many iterations?',
                        type = float)
    parser.add_argument('-size',
                        help = 'Percent of training set aside for validation',
                        type = float)

    args = parser.parse_args()
    
    # check inputs
    weight_scoring = False
    rm_neutrals = False
    iterations = 10.0
    validation_size = 0.001
    
    if args.method == 'method2':
        weight_scoring = True
    
    if args.neutrals:
        if args.neutrals.lower() == 't':
            rm_neutrals = True
            
    if args.iter:
        iterations = args.iter
        
    if args.size:
        if args.size > 0 and args.size < 1:
            validation_size = args.size
    
    full_map_dict = tsvRead('train.tsv')
    
    now = time.time()
    iter_count = 0
    model_scores = []

    
    # loop to compute average accuracies
    for i in xrange(int(iterations)):
        model_scores.append(testingWrapper(full_map_dict,
                                           size = validation_size,
                                           weighted_scores = weight_scoring,
                                           remove_neutrals = rm_neutrals))
        iter_count += 1
        print 'finished iteration ' + str(iter_count)
    # end loop

    
    print 'model average over ' + str(iter_count) + ' iterations'
    print str(np.mean(model_scores))
    print 'average time per run: ' + str((time.time() - now)/iterations)
    
    return None
    
    
if __name__ == '__main__':
    main()
    pass