_author_ = 'mshadish'
"""
This script defines several utility functions, with the idea that
these utility functions can be useful not only for other models in this
competition, but also for other competitions

1. testPhraseSplit(phrase)
    - splits a given phrase into all of its potential sub-phrases
    - and provides a mapping of the phrases' relative positioning
    
2. computeOverPredict(predictions, full data set)
    - returns the ratio of over-guesses
    - to the total number of predictions we got wrong
    
3. rebalanceSample(x, y, desired sample size)
    - extracts a balanced sample of x and y
    
4. plotLearningCurve(training sizes, training scores, test scores, model name)
    - generates a plot of the learning curve
    
5. plotValidationCurve(x-axis values, training scores, testing scores,
                       model name, x-label)
    - generates a plot of the validation curve
"""
import re
import numpy as np
import matplotlib.pyplot as plt

    
def testPhraseSplit(test_phrase):
    """
    Takes in a test phrase
    
    Returns a dict of all of the possible sub-phrases
    that can be made by that test_phrase
    as well as their relative positions found in the phrase
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
    
    
    
def computeOverPredict(predicted_scores, full_mapping_dict):
    """
    Takes in a dictionary with our predictions on a validation set
    as well as a dictionary that maps the phrases to their true scores

    This function gives us a number that tells us whether the model
    a) over-guesses
    b) under-guesses
    c) evenly over- and under-guesses
    
    Returns the ratio of predictions that were over-guesses
    to the number of total incorrect predictions
    = (number of over-guesses) / (number of wrong predictions_)
    """
    # initialize the numerator and denominators
    over_guesses = 0.0
    total_wrong = 0.0
    
    # compare with the actual values
    for phrase in predicted_scores:
        # check if our prediction was correct or not
        if predicted_scores[phrase] != int(full_mapping_dict[phrase]):
            total_wrong += 1
            
            # did we over- or under-guess?
            if predicted_scores[phrase] > int(full_mapping_dict[phrase]):
                over_guesses += 1
            
    # return the ratio of over-guesses to incrorrect predictions    
    return over_guesses / total_wrong
    
    
    
def rebalanceSample(x_nparray, y_list, sample_size = 0):
    """
    Returns a sample x and corresponding y
    that has been rebalanced
    """
    if sample_size <= 0:
        sample_size = len(x_nparray)
        
    num_classes = len(set(y_list))
    # based on the number of classes,
    # we will sample from each class accordingly
    # to obtain the specified sample size
    subsample_size = np.ceil(float(sample_size) / num_classes)
    
    return_x = []
    return_y = []
    
    for y in set(y_list):
        # loop through each distinct class
        indices = []
        for index, item in enumerate(y_list):
            # check if the current item matches our selected class
            if item == y:
                # if so, grab the index
                indices.append(index)
        # end loop through y_list
        # now sample
        indices_sample = np.random.choice(indices, size = subsample_size,
                                          replace = True)
        # now obtain the values based on the indices sampled
        y_subsample = [y_list[i] for i in indices_sample]
        x_subsample = [x_nparray[i] for i in indices_sample]
        # and append
        return_x = return_x + x_subsample
        return_y = return_y + y_subsample
        
    # if our return list is larger than the specified sample size
    # we obtained too many values due to rounding error
    # and will chop off a few values at random
    while len(return_y) > sample_size:
        random_drop_index = np.random.randint(low = 0, high = len(return_y))
        return_x.pop(random_drop_index)
        return_y.pop(random_drop_index)
        
    return np.asarray(return_x), np.asarray(return_y)
    
    
    
def plotLearningCurve(train_sizes, training_avgs, validation_avgs, model_name):
    """                                                                         
    This function plots the learning curves                                     
    based on training averages and cross-validation averages                    
    against sample size                                                         
    """
    # clear figure                                                              
    plt.clf()

    # generate plots                                                            
    p1, = plt.plot(train_sizes, training_avgs, 'ro-', label = 'training')
    p2, = plt.plot(train_sizes, validation_avgs, 'go-',
                   label = 'cross-validation')

    # axis labels                                                               
    plt.xlabel('Training set size', fontsize = 16)
    plt.ylabel('Score', fontsize = 16)
    # create the title as a concat of our model and 'learning curves'
    plt.title(model_name + ' Learning Curves', fontdict = {'fontsize': 16})
    plt.legend(loc = 0)

    # save figure                                                               
    model_name = re.sub(r'\W', '_', model_name)
    plt.savefig(re.sub(r'\s', '_', model_name) + '_learning_curve.png',
                format = 'png')                                                

    return None
    
    
    
def plotValidationCurve(c, training_avgs, validation_avgs, model_name,
                        x_label = 'C'):
    """                                                                                                                                                       
    This function plots the training and validation averages                                                                                                  
    Against the parameter we are trying to optimize                                                                                                           
    """
    plt.clf()
    
    p1, = plt.plot(c, training_avgs, 'ro-', label = 'training')
    p2, = plt.plot(c, validation_avgs, 'go-', label = 'validation')

    plt.xlabel(x_label, fontsize = 16)
    plt.ylabel('Score', fontsize = 16)

    plt.title(model_name + ' Validation Curve',
              fontdict = {'fontsize': 16})
    plt.legend(loc = 0)
    plt.semilogx()
    
    model_name = re.sub(r'\W', '_', model_name)
    plt.savefig(model_name + '_validation.png',
                format = 'png')
    
    return None
