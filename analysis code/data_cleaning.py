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
import bhUtilities
import collections
import NaiveBayes
import re
import sys

import numpy as np
import pandas as pd

# pd.set_option('display.max_rows', 5)
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
    # get rid of sub-phrases
    for (key, value) in df.groupby(['sentence_id']):

        local_dict = dict()

        # keep most of the data
        local_dict['sentence_id'] = value['sentence_id'].max()
        local_dict['sentiment'] = value['sentiment'].max()

        # keep only the longest phrase (aka the whole phrase)
        local_dict['phrase'] = max(list(value['phrase']), key=len)

        # add to the list
        df_list.append(local_dict)

    return_df = pd.DataFrame(df_list)

    return_df['length'] = return_df['phrase'].apply(lambda text: len(text))
    return_df['word_list'] = return_df['phrase'].apply(bhUtilities.splitAndCleanString)
    return_df['word_counter'] = return_df['word_list'].apply(lambda text: collections.Counter(text))
    return_df['num_unique_words'] = return_df['word_list'].apply(lambda text: len(set(text)))
    return_df.to_csv('output.csv')
    return return_df

def create_test_df():

    df = pd.read_csv("../data/test.tsv", sep='\t')

    df.columns = ['phrase_id', 'sentence_id', 'phrase']

    df_list = list()
    # get rid of sub-phrases
    for (key, value) in df.groupby(['sentence_id']):

        local_dict = dict()

        # keep most of the data
        local_dict['sentence_id'] = value['sentence_id'].max()

        # keep only the longest phrase (aka the whole phrase)
        local_dict['phrase'] = max(list(value['phrase']), key=len)

        # add to the list
        df_list.append(local_dict)

    return_df = pd.DataFrame(df_list)

    return_df['length'] = return_df['phrase'].apply(lambda text: len(text))
    return_df['word_list'] = return_df['phrase'].apply(bhUtilities.splitAndCleanString)
    return_df['word_counter'] = return_df['word_list'].apply(lambda text_list: collections.Counter(text_list))
    return_df['num_unique_words'] = return_df['word_list'].apply(lambda text_list: len(set(text_list)))
    return_df.to_csv('output2.csv')
    return return_df


def main():
    train_df = create_train_df()
    test_df = create_test_df()

    msk = np.random.rand(len(train_df)) < 0.8
    train_df_1 = train_df[msk]
    train_df_2 = train_df[~msk]

    train_df_1 = train_df_1.reset_index()
    train_df_2 = train_df_2.reset_index()




    # try NB
    nb = NaiveBayes.NaiveBayes()
    nb.fit(train_df_1['phrase'], train_df_1['sentiment'])
    prediction =  nb.predict(train_df_2['phrase'])

    labels_2 = pd.DataFrame(train_df_2['sentiment'])
    check_df = pd.merge(prediction, labels_2, left_index=True, right_index=True)

    check_df['correct'] = (check_df['guess'] == check_df['sentiment'] - 0)
    print check_df
    print np.mean(check_df['correct'])

# main
# *********************************

if __name__ == '__main__':
    print 'Begin Main'
    main()
    print 'End Main'

