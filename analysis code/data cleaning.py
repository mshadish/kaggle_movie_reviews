"""
coding=utf-8
data cleaning.py
A component of: kaggle_movie_reviews
(C) Brendan J. Herger
Analytics Master's Candidate at University of San Francisco
13herger@gmail.com

Created on 10/28/14, at 6:48 PM

Available under MIT License
http://opensource.org/licenses/MIT
"""
# imports
# *********************************
import re
import sys

import numpy as np
import pandas as pd


# global variables
# *********************************

__author__ = 'bjherger'
__license__ = 'http://opensource.org/licenses/MIT'
__version__ = '1.0'
__email__ = '13herger@gmail.com'
__status__ = 'Development'
__maintainer__ = 'bjherger'


# functions
# *********************************
def create_train_df():

    df = pd.read_csv("../data/train.tsv", sep='\t')

    df.columns = ['phrase_id', 'sentence_id', 'phrase', 'sentiment']

    df_list = list()
    for (key, value) in df.groupby(['sentence_id']):

        local_dict = dict()

        # keep most of the data
        local_dict['sentence_id'] = value['sentence_id']
        local_dict['sentiment'] = value['sentiment']

        # keep only the longest phrase (aka the whole phrase)
        local_dict['phrase'] = max(list(value['phrase']), key=len)

        # add to the list
        df_list.append(local_dict)

    return_df = pd.DataFrame(df_list)
    return return_df

def create_test_df():

    df = pd.read_csv("../data/test.tsv", sep='\t')

    df.columns = ['phrase_id', 'sentence_id', 'phrase']

    df_list = list()
    for (key, value) in df.groupby(['sentence_id']):

        local_dict = dict()

        # keep most of the data
        local_dict['sentence_id'] = value['sentence_id']

        # keep only the longest phrase (aka the whole phrase)
        local_dict['phrase'] = max(list(value['phrase']), key=len)

        # add to the list
        df_list.append(local_dict)

    return_df = pd.DataFrame(df_list)
    return return_df

def main():
    train_df = create_train_df()
    test_df = create_test_df()

    


# main
# *********************************

if __name__ == '__main__':
    print 'Begin Main'
    main()
    print 'End Main'

