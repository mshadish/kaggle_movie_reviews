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
from joblib import Parallel, delayed
import multiprocessing


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
    
    
def splitTrainHoldout(mapping_dict, split_fraction = 0.2):
    """
    Takes in a mapping dictionary of training data
    
    Returns a holdout set of the training data based on the split fraction
    Also modifies (by reference) the mapping dictionary
    with the remaining training data, minus the holdout set
    """
    # grab the holdout set
    sample_fraction = int(round(split_fraction * len(mapping_dict.keys())))
    holdout_set = random.sample(mapping_dict, sample_fraction)
    
    # remove the holdout data from the mapping dictionary
    for phrase in holdout_set:
        del mapping_dict[phrase]
        
    return holdout_set
    
    
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
    return_dict = {key: mapping_dict[key] for key in match_dict}
    return return_dict
    
    
if __name__ == '__main__':
    
    num_cores = multiprocessing.cpu_count()
    
    now = time.time()
    map_dict = tsvRead('train.tsv')
    print time.time() - now
    now = time.time()
    
    count = 0
    holdout = splitTrainHoldout(map_dict, 0.1)
    
    """
    for phrase in holdout:
        x = joinWithMapping(map_dict, testPhraseSplit(phrase))
        print 'completed ' + str(count)
        count += 1
        
    print time.time() - now
    """
    
    func = lambda x: joinWithMapping(map_dict, testPhraseSplit(x))
    Parallel(n_jobs = num_cores)(delayed(func)(phrase) for phrase in holdout)
            
    print time.time() - now
