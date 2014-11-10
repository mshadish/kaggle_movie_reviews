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
import random
import bhUtilities
import collections
import NaiveBayes
import re
import sys
import functools

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
    # TODO remove

    random.shuffle(df_list)
    print len(df_list)

    df_list = df_list[:]
    return_df = pd.DataFrame(df_list)

    return_df['length'] = return_df['phrase'].apply(lambda text: len(text))
    return_df['word_list'] = return_df['phrase'].apply(bhUtilities.splitAndCleanString)
    return_df['word_counter'] = return_df['word_list'].apply(lambda text: collections.Counter(text))
    return_df['num_unique_words'] = return_df['word_list'].apply(lambda text: len(set(text)))
    return_df['unique_word_density'] = return_df['word_list'].apply(lambda text_list: len(set(text_list)) / float(max(len(text_list), 1)))

    return_df.to_csv('output.csv')

    return_df['guess'] = leave_one_out_test(return_df)

    return_df['guess-sentiment'] = return_df['guess'] - return_df['sentiment']

    return_df.to_csv('output.csv')
    print return_df

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
    return_df['unique_word_density'] = return_df['word_list'].apply(lambda text_list: len(set(text_list)) / max(len(text_list), 1))


    # partialed_loov = functools.partial(leave_one_out_test, df = return_df)
    # return_df['guess'] = return_df.apply(partialed_loov, axis = 1)


    # print return_df
    return return_df


def leave_one_out_test(df):
    guess_list = list()
    for (row_index, row_series) in df.iterrows():
        train_df = df.drop(row_index)
        test_df = row_series.to_frame().transpose()

        if (row_index % 25) == 0:
            print row_index, ' ',
        if (row_index % 200) == 0:
            print

        nb = NaiveBayes.NaiveBayes()
        nb.fit(train_df['phrase'], train_df['sentiment'])
        prediction = nb.predict(test_df['phrase'])

        guess_list.append(prediction['guess'][0])
    return_series = pd.Series(guess_list)
    return return_series


def main():
    train_df = create_train_df()
    # test_df = create_test_df()

    # msk = np.random.rand(len(train_df)) < 0.8
    # train_df_1 = train_df[msk]
    # train_df_2 = train_df[~msk]
    #
    # train_df_1 = train_df_1.reset_index()
    # train_df_2 = train_df_2.reset_index()

    # try NB
    # nb = NaiveBayes.NaiveBayes()
    # nb.fit(train_df_1['phrase'], train_df_1['sentiment'])
    # prediction =  nb.predict(train_df_2['phrase'])
    #
    # labels_2 = pd.DataFrame(train_df_2['sentiment'])
    # check_df = pd.merge(prediction, labels_2, left_index=True, right_index=True)
    #
    # check_df['correct'] = (check_df['guess'] == check_df['sentiment'] + 1)
    # print check_df
    # print np.mean(check_df['correct'])

# main
# *********************************

if __name__ == '__main__':
    print 'Begin Main'
    bhUtilities.timeItStart(printOff=False)
    main()
    bhUtilities.timeItEnd(1)
    print 'End Main'

