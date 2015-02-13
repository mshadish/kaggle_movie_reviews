__author__ = 'mshadish'
"""
This is called via the command line as follows
> python phrase_mapping.py -method (1 or 2) -neutrals (remove?)
...-iter (how many iterations, e.g. 10) -holdout (between 0 and 1)

Method 1: Unweighted scores used to compute phrase sentiment
    - note: this has been deprecated, did not yield reliable results
    
Method 2: Weighted scores used to compute phrase sentiment

This script defines several functions in order to
1. Create a "mapping" dictionary of phrases and their sentiments
2. Break apart each incoming test phrase into all of its possible sub-phrases
3. Match these sub-phrases to our mapping table
4. Compute a sentiment score for the given test phrase

Note: this script can be thought of as an exercise in Python,
as much of the functionality defined here can be replicated
via a few function calls from pandas or scikit-learn
"""
# imports
import re
import sys
import time
import random
import argparse
import numpy as np
from utils import testPhraseSplit
from utils import computeOverPredict


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
    
    
    
def runPredictions(mapping_dict, holdout_set, remove_neutrals = False):
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
        
    # repeat for every phrase in the holdout set
    return predicted_scores
    
    
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

    
    
def testingWrapper(full_map_dict, size, remove_neutrals = False):
    """
    Runs a single iteration of
    1. generating a holdout set
    2. computing the model prediction score on the holdout set
    """
    # we will convert the dict values
    # from strings to floats
    # will also set the neutral to 0 for ease of understanding
    full_map_dict = {i: int(full_map_dict[i]) - 2 for i in full_map_dict}

    # generate the holdout set
    holdout_set, mapping_minus_holdout = splitTrainHoldout(full_map_dict, size)
    
    # make predictions, using the user-specified model
    predictions = runPredictions(mapping_minus_holdout, holdout_set, remove_neutrals)
                              
    # compute the accuracy of our model
    accuracy = computeAccuracy(predictions, full_map_dict)
    
    # compute the number of over-guesses
    over_guesses = computeOverPredict(predictions, full_map_dict)
    
    return accuracy, over_guesses
    
    
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
    

    
def main():
    """
    Main wrapper function to build and test the model
    """
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
    # repeat for however many specified iterations

    # report on the results of this model
    print 'model average over %d iterations: %f' % (iter_count,
                                                    np.mean(model_scores))
    print 'over-guesses to total errors: %f' % np.mean(over_guess_ratios)
    print 'average time per run: %f' % ((time.time() - now)/iterations)
    return None
    
    
    
if __name__ == '__main__':
    main()
