"""
coding=utf-8
NaiveBayes.py
A component of: hw6
(C) Brendan J. Herger
Analytics Master's Candidate at University of San Francisco
13herger@gmail.com

Created on 10/24/14, at 3:07 PM

Available under MIT License
http://opensource.org/licenses/MIT
"""
# imports
# *********************************
import bhUtilities
import collections

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


def aggregate_list_of_dicts(list_of_dicts):
    """
    flattens a list of dictionaries using update method
    :param list_of_dicts: list of dicts to flatten
    :return: one flattened dict
    :rtype: dict
    """
    to_return = collections.Counter()
    for local_dict in list_of_dicts:
        to_return.update(local_dict)
    return to_return


def length_function(list_of_values):
    """
    finds the combined length of all sub-lists
    :param list_of_values:
    :return: combined length of all sub-lists
    """
    counter = 0
    for local_list in list_of_values:
        counter += len(local_list)
    return counter


def train(train_df):
    """
    local helper method. see documentation for NaiveBayes.fit()
    :param train_df:
    :return:
    """

    # add files
    train_df["cleaned_text"] = train_df["text"].apply(lambda text: bhUtilities.splitAndCleanString(text))
    train_df["counter"] = train_df["cleaned_text"].apply(lambda text: collections.Counter(text))

    # create a new data frame with group by data
    combined_list = list()
    for (df_grouby_name, df_groupby_value) in train_df.groupby("label"):
        # combined counter for all documents of same label
        aggregrated_counter = aggregate_list_of_dicts(df_groupby_value["counter"])

        # create dict that contains pandas columns
        local_dict = dict()

        # counter for word frequency
        local_dict["counter"] = aggregrated_counter

        # number of non-unique words
        local_dict["num_non_unique_words"] = length_function(df_groupby_value["cleaned_text"])

        # number of unique words
        local_dict['num_unique_words'] = len(aggregrated_counter.keys())

        # label
        local_dict["label"] = df_grouby_name

        # add to list, which will later be converted to dataframe
        combined_list.append(local_dict)

    df = pd.DataFrame(combined_list)

    return df


def predict(test_data, trained_df):
    """
    local helper method. see documentation for NaiveBayes.predict()
    :param test_data:
    :param trained_df:
    :return:
    """

    # type check
    # test_data = pd.DataFrame(test_data)
    # trained_df = pd.DataFrame(trained_df)

    # variables
    total_non_unique_words = trained_df['num_unique_words'].sum()

    # set up test_data
    test_data["cleaned_text"] = test_data["text"].apply(lambda text: bhUtilities.splitAndCleanString(text))
    test_data["counter"] = test_data["cleaned_text"].apply(lambda text: collections.Counter(text))

    # iterate through test data rows (each row is a document)
    guess_list = list()
    for test_data_index, test_data_row in test_data.iterrows():

        # unpack variables
        local_test_counter = test_data_row['counter']

        # keep track of which is the best label so far
        best_label = None
        best_label_score = None

        # iterate through trained data rows (each row is a label), get score for each label.
        for trained_data_index, trained_data_row in trained_df.iterrows():

            # unpack variables
            label_num_non_unique_words = trained_data_row['num_non_unique_words']
            label_counter = trained_data_row['counter']

            # running counter. each entry is for a different word in the testing document
            label_score_list = []

            # iterate through words in test data
            for (word_i, n_i) in local_test_counter.iteritems():
                # number of times word occurs in label
                label_num_occurences = label_counter.get(word_i, 0)

                # probability of word, given label ( +1's for words that were not seen in training)
                p_i = (label_num_occurences + 1.0) / (label_num_non_unique_words + total_non_unique_words + 1.0)

                # create log-scaled label score for word. less negative scores are better
                label_word_score = n_i * np.log(p_i)
                label_score_list.append(label_word_score)

            # check if current label is best fit. if so, set to best fit.
            if sum(label_score_list) > best_label_score:
                best_label = trained_data_row['label']
                best_label_score = sum(label_score_list)

        # store to later add to dataframe
        local_dict = dict()
        local_dict['index'] = int(test_data_index)
        local_dict['guess'] = best_label
        guess_list.append(local_dict)

    # transform output to dataframe
    return_df = pd.DataFrame(guess_list)

    # return
    return return_df


# classes
# *********************************

class NaiveBayes(object):
    """
    A NaiveBayes implementation
    """

    def __init__(self):
        """
        itiates the NaiveBayes model
        :return: None
        :rtype: None
        """
        self.counter = collections.Counter()
        self.trained = None
        self.training_data = pd.DataFrame()

    def fit(self, data, labels):
        """
        Fits (trains) the NaiveBayes. Both data and labels should be a 1 dimensional arrays with multiple observations.
        :param data: 1-D array containing multiple observations. Each observation should be a cleaned string of text.
        :param labels: 1-D array containing multiple observations. Each observation a categorical value.
        :return: self
        :rtype: NaiveBayes
        """
        data = pd.DataFrame(data)
        labels = pd.DataFrame(labels)

        data.columns = ['text']
        labels.columns = ["label"]

        data["label"] = labels
        self.training_data = self.training_data.append(data)
        self.trained = train(self.training_data)

        return self

    def predict(self, test_data):
        """
        Predicts the categorical value for each observation of test_data. test_data should be a 1 dimensional array
        with multiple observations.
        :param test_data: 1-D array containing multiple observations. Each observation should be a cleaned string
        of text
        :return: predicted categorical values, based on previous fit. The observations will be in the same order as
        test_data
        """
        test_data = pd.DataFrame(test_data)
        test_data.columns = ['text']
        return predict(test_data, self.trained)
