# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 10:33:06 2014

@author: mshadish

This is called via the command line as follows
> python phrase_mapping.py -method (1 or 2) -neutrals (remove?)
...-iter (how many iterations, e.g. 10) -holdout (between 0 and 1)

Method 1: Unweighted scores used to compute phrase sentiment
Method 2: Weighted scores used to compute phrase sentiment

This script defines several functions in order to
1. Create a "mapping" dictionary of phrases and their sentiments
2. Break apart each incoming test phrase into all of its possible sub-phrases
3. Match these sub-phrases to our mapping table
4. Compute a sentiment score for the given test phrase
"""

import re
import time
import numpy as np
from utils import tsvRead
from utils import testPhraseSplit
from utils import computeAccuracy
from utils import splitTrainHoldout
from utils import commandLineIntake
from utils import computeOverPredict
from utils import computeWeightedScore

    
    
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
    return_dict = {key: mapping_dict[key] for key in match_dict}
    return return_dict
    
    
    
def method2(mapping_dict, holdout_set, remove_neutrals = False):
    """
    Method 2:
    Compute a weighted average score for each phrase
    Weightings are calculated on a non-linear scale based on the phrase length
    
    Also around 60% accuracy when we remove neutrals
    """
    # we will store our results in a dictionary
    predicted_scores = {}
    
    for phrase in holdout_set:
        
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

        # otherwise, compute the weighted scores
        scores = []
        phrase_lengths = []
        for matched_phrase in matches:
            # grab the score for the phrase
            num = float(matches[matched_phrase])
            # and add it to our list of scores
            scores.append(num)
            
            # compute the length of our matched phrase
            matched_phrase_length = float(len(re.split(r'\s', matched_phrase)))
            # store it in a list, we will reference it after looping
            # to assist with re-normalization after weighting
            phrase_lengths.append(matched_phrase_length)
            
        # end loop through matched phrases
        
        # compute the weighted score
        weighted_score = computeWeightedScore(scores, phrase_lengths)
        
        # and add our weighted average to the scoring dictionary
        predicted_scores[phrase] = weighted_score
        
    # end loop through holdout set
                
    return predicted_scores

    
    
def testingWrapper(full_map_dict, size, remove_neutrals = False):
    """
    Runs a single iteration of
    1. generating a holdout set
    2. computing the model prediction score on the holdout set
    """
    # we will convert the dict values
    # from strings to floats
    # will also set the neutral to 0 for ease of understandability
    full_map_dict = {i: int(full_map_dict[i]) - 2 for i in full_map_dict}

    # generate the holdout set
    holdout_set, mapping_minus_holdout = splitTrainHoldout(full_map_dict, size)
    
    # make predictions, using the user-specified model
    predictions = method2(mapping_minus_holdout, holdout_set, remove_neutrals)
                              
    # compute the accuracy of our model
    accuracy = computeAccuracy(predictions, full_map_dict)
    
    # compute the number of over-guesses
    over_guesses = computeOverPredict(predictions, full_map_dict)
    
    return accuracy, over_guesses
    

    
def main():
    
    rm_neutrals, iterations, holdout_size = commandLineIntake()
    if rm_neutrals:
        print 'Removing neutrals'
    full_map_dict = tsvRead('train.tsv')
    
    now = time.time()
    iter_count = 0
    model_scores = []
    over_guess_ratios = []
    
    # loop to compute average accuracies
    for i in xrange(int(iterations)):
        score, over_guess = testingWrapper(full_map_dict,
                                           size = holdout_size,
                                           remove_neutrals = rm_neutrals)
        
        # keep track of this run's model score and over-guess ratio
        model_scores.append(score)
        over_guess_ratios.append(over_guess)
        
        # increment our counter, report to the console
        iter_count += 1
        print 'finished iteration ' + str(iter_count)
        print 'running average: %f' % np.mean(model_scores)
    # end loop

    # report on the results of this model
    print 'model average over %d iterations: %f' % (iter_count,
                                                    np.mean(model_scores))
    print 'over-guesses to total errors: %f' % np.mean(over_guess_ratios)
    print 'average time per run: %f' % ((time.time() - now)/iterations)
    return None
    
    
    
if __name__ == '__main__':
    main()