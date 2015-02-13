__author__ = 'mshadish'
"""
This script imports much of its functionality
from the functionality written originally to serve SVM testing
but has been re-purposed to test the effectiveness and cross-validate
a random forest model
"""
# standard imports
import random
import pandas as pd
import numpy as np
# re-importing of functions i've already written
from svm import runLearningCurve
from svm import createSubmissionFile
# sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
# other imports
import multiprocessing as mp
from utils import rebalanceSample
from utils import plotValidationCurve


# define number of cores of this computer as a constant
num_cores = mp.cpu_count()


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
    # repeat for every parameter in the given parameter space
    
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


def main():
    # first, initialize our random forest model
    rf_model = RandomForestClassifier(n_estimators = 100, n_jobs = num_cores)
    
    # now run it through our tests
    runLearningCurve(model = rf_model, balanced = False)
    #runValidationCurve(param_name = 'N Trees', param_space = [100])
    createSubmissionFile(input_model = rf_model)
    return None
    
    
if __name__ == '__main__':
    main()
    pass