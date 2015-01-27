__author__ = 'mshadish'
"""
This script defines the functions used to explore how SVM performs
and find the optimal training set size, loss function, and
regularization parameter to achieve the highest possible validation scores

1. learningLoops()
    - helper function to runLearningCurve()

2. runLearningCurve()
    - produces learning curves
    
3. validationLoops()
    - helper function to runValidationCurve()
    
4. runValidationCurve()
    - produces validation curves
    
5. predictOnTest()
    - helper function to createSubmissionFile()
    
6. createSubmissionFile()
    - produces CSV output ready for submission for kaggle competition

We achieved best results using as much of the data as possible
with L2 regularization, C = 14
"""

# standard imports
import random
import pandas as pd
import numpy as np
# sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
# utility imports
import multiprocessing
from utils import rebalanceSample
# plot imports
from utils import plotLearningCurve
from utils import plotValidationCurve


# define global variables
num_cores = multiprocessing.cpu_count()
    
    
def learningLoops(input_model, raw_x, raw_y, learning_space, rebalance = True):
    """
    Runs cross-validation for an input model
    given subsetted x and y
    """
    # initialize lists to keep track of learning scores
    train_set_sizes = []
    train_scores = []
    test_scores = []
    # loop through the different sets of training sizes
    # computing the training and test scores each in each loop
    for train_perc in learning_space:
        # extrapolate the training size
        # note: we multiply by 0.8 because we do 5-fold cross-validation
        train_size = round(len(raw_x) * train_perc * 0.67)
        print 'Starting %f Balanced' % train_size
        
        # now extract out x and y
        x = None
        y = None
        if rebalance:
            # initialize our x and y, with rebalancing
            x, y = rebalanceSample(raw_x, raw_y, sample_size = train_size)
        else:
            # initialize our x and y, wthout rebalancing
            # first, grab the idx numbers of a random sample
            sample_idx = random.sample(np.arange(len(raw_x)),
                                       k = int(train_size))
            # then, extract out the x and y
            x = np.asarray([raw_x[i] for i in sample_idx])
            y = np.asarray([raw_y[i] for i in sample_idx])
            
        # initialize the word vectorizer
        vec = CountVectorizer()
        
        # run 5-fold cross-validation
        kfolds = KFold(n = len(x), n_folds = 3, shuffle = True)
        train_accuracies = []
        test_accuracies = []
        for train_idx, test_idx in kfolds:
            # grab training data
            x_train = x[train_idx]
            y_train = y[train_idx]
            # grab testing data
            x_test = x[test_idx]
            y_test = y[test_idx]
            
            # now vectorize our training and testing x
            x_train = vec.fit_transform(x_train)
            x_train = x_train.toarray()
            x_test = vec.transform(x_test).toarray()
            
            # now we are ready to train the model
            input_model.fit(x_train, y_train)
            # generate predictions for training and test sets
            train_predictions = input_model.predict(x_train)
            test_predictions = input_model.predict(x_test)
            
            # compute accuracy
            acc = accuracy_score(y_test, test_predictions)
            print 'Accuracy: %f' % acc
            test_accuracies.append(acc)
            train_accuracies.append(accuracy_score(y_train,
                                                   train_predictions))
        # end loop for cross-validation
            
        print 'Average accuracy of %f: %f' % (train_size,
                                              np.mean(test_accuracies))
        print
        # append to lists
        train_set_sizes.append(train_size)
        train_scores.append(np.mean(train_accuracies))
        test_scores.append(np.mean(test_accuracies))
    # end loop through learning space
        
    # finally, plot it all in a learning curve
    save_string = ''
    if rebalance:
        save_string = 'RF_Balanced'
    else:
        save_string = 'RF_Unbalanced'
    plotLearningCurve(train_set_sizes, train_scores, test_scores, save_string)
    
    return None


def runLearningCurve(learning_space = np.linspace(start=0.05,stop=1.0,num=11),
                     balanced = True, unbalanced = True):
    """
    Wrapper function to run learning curve for SVM
    on both balanced and unbalanced datasets
    """
    # read in the data
    data_df = pd.read_table('train.tsv', header=0)
    # grab x, y
    raw_x = data_df['Phrase']
    raw_y = data_df['Sentiment']
    
    # initialize the model
    model = RandomForestClassifier(n_jobs = num_cores)
        
    # run for a rebalanced dataset
    if balanced:
        learningLoops(model, raw_x, raw_y, learning_space)
    # run for an unbalanced dataset
    if unbalanced:
        learningLoops(model, raw_x, raw_y, learning_space, rebalance = False)
    
    return None
    
    
def validationLoops(raw_x,raw_y,param_name,param_space,train_size,num_folds):
    """
    Runs through the different C's in the c-space for a given loss function
    and plots the validation curve
    also prints out the results
    """
    # initalize lists to keep track of results
    params = []
    train_scores = []
    test_scores = []
    # loop through the different sets of training sizes
    for p in param_space:
        print 'Starting parameter %f' % p
        # initialize the model
        #model = RandomForestClassifier(n_estimators = p, n_jobs = num_cores,
        #                               oob_score = True)
        model = RandomForestClassifier(n_estimators = p, n_jobs = num_cores)
        # initialize our x and y
        #x, y = rebalanceSample(raw_x, raw_y, sample_size = train_size)
        sample_idx = random.sample(np.arange(len(raw_x)), train_size)
        x = np.asarray([raw_x[i] for i in sample_idx])
        y = np.asarray([raw_y[i] for i in sample_idx])
        
        # initialize the word vectorizer
        vec = CountVectorizer()
        
        # run 5-fold cross-validation
        kfolds = KFold(n = len(x), n_folds = num_folds, shuffle = True)
        train_accuracies = []
        test_accuracies = []
        for train_idx, test_idx in kfolds:
            # grab training data
            x_train = x[train_idx]
            y_train = y[train_idx]
            # rebalance
            x_train, y_train = rebalanceSample(x_train, y_train)
            # grab testing data
            x_test = raw_x[test_idx]
            y_test = raw_y[test_idx]
            
            # now vectorize our training and testing x
            x_train = vec.fit_transform(x_train)
            x_train = x_train.toarray()
            x_test = vec.transform(x_test).toarray()
            
            # now we are ready to train the model
            model.fit(x_train, y_train)
            #print model.oob_score_
            # generate predictions for training and test sets
            train_predictions = model.predict(x_train)
            test_predictions = model.predict(x_test)
            
            # compute accuracy
            acc = accuracy_score(y_test, test_predictions)
            print 'Accuracy: %f' % acc
            test_accuracies.append(acc)
            train_accuracies.append(accuracy_score(y_train,
                                                   train_predictions))
        # end loop for cross-validation
            
        print 'Average accuracy of %f: %f' % (p, np.mean(test_accuracies))
        print
        # append to lists
        params.append(p)
        train_scores.append(np.mean(train_accuracies))
        test_scores.append(np.mean(test_accuracies))
    # end loop through parameter C space
    
    # compute cross-validation training size
    cv_train_size = int(round(train_size * (num_folds - 1.0) / num_folds))
    # plot and return
    plotValidationCurve(params, train_scores, test_scores,
                        'RF Balanced ' + param_name + ' ' + str(cv_train_size),
                        x_label = param_name)
    return None
    
    

def runValidationCurve(train_perc = 1.0, param_name = 'Unspec',
                       param_space = np.logspace(-2, 2, num = 16),
                       num_folds = 5):
    """
    Wrapper function to run validation curve for SVM
    for balanced dataset with maximal-size training set
    Again, we will have to do this manually b/c of how our vectorizer works
    """
    # read in the data
    data_df = pd.read_table('train.tsv', header=0)
    # grab x, y
    raw_x = data_df['Phrase']
    raw_y = data_df['Sentiment']
    
    # compute the training size
    if train_perc > 1.0 or train_perc <= 0.0:
        train_perc = 1.0
    train_size = int(round(train_perc * len(raw_x)))
    
    # generate validation curve
    validationLoops(raw_x,raw_y,param_name, param_space, train_size, num_folds)
    
    return None
    
    
    
def predictOnTest(input_model, word_vectorizer, test_file = 'test.tsv'):
    """
    Runs predictions on the test file given the input model
    Returns those predictions and their associated phrase ID's
    """
    # read in the test data
    test_df = pd.read_table(test_file, header = 0)
    # grab the phrase id and the phrase
    test_id = test_df['PhraseId']
    test_phrase = test_df['Phrase']
    
    # we must vectorize the test phrases
    test_phrase_vec = word_vectorizer.transform(test_phrase).toarray()
    
    # run predictions
    predictions = input_model.predict(test_phrase_vec)
    
    # create the return data frame
    return_df = pd.DataFrame(data = {'PhraseId': test_id,
                                     'Sentiment': predictions})
    return return_df
    
    
    
def createSubmissionFile(input_model, train_perc = 1.0):
    """
    Runs an input model with a specified training percentage size
    and writes out a CSV of the form:
    PhraseId,Sentiment
    156061,2
    156062,3
    156063,0
    ...etc
    """
    # read in the data
    data_df = pd.read_table('train.tsv', header=0)
    # grab x, y
    raw_x = data_df['Phrase']
    raw_y = data_df['Sentiment']
    
    # determine the training size based on the input training percent
    if train_perc > 1.0:
        print 'Invalid training percentage, defaulting to 100%'
        train_perc = 1.0
    train_size = int(round(len(raw_x) * train_perc))
    # rebalance the data
    x, y = rebalanceSample(raw_x, raw_y, sample_size = train_size)
    # try without rebalancing
    #x = raw_x
    #y = raw_y
    # vectorize the training data with a word vectorizer
    vec = CountVectorizer()
    x_train = vec.fit_transform(x)
    x_train = x_train.toarray()
    
    # fit the model
    input_model.fit(x_train, y)
    
    # run our predictions
    submission_df = predictOnTest(input_model = input_model,
                                  word_vectorizer = vec)
    # and write it to a csv
    submission_df.to_csv('solutions.csv', index = False)
    return None
    


def main():
    #runLearningCurve(balanced = False)
    #runValidationCurve(train_perc = 1.0, param_name = 'N Trees',
    #                   param_space = [100], num_folds = 3)
    model = RandomForestClassifier(n_estimators = 100, n_jobs = num_cores)
    createSubmissionFile(input_model = model)
    return None
    
    
if __name__ == '__main__':
    main()
    pass