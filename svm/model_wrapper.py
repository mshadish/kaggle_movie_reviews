__author__ = 'mshadish'
"""
Wrapper for running SVM
56% out of the box on sample of 15k

77% with re-balancing on sample of 20k
"""


from sklearn import svm
#from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from utils import tsvReadXY
import sklearn.metrics as metrics
import numpy as np


# first, let's initialize our model
model = svm.SVC(kernel = 'linear')

# read in the file and grab the sentences + sentiment scores
x, y = tsvReadXY('train_sample.tsv')

# will try balancing the classes
indices_neg = []
indices_som_neg = []
indices_neut = []
indices_som_pos = []
indices_pos = []
for i, item in enumerate(y):
    if item == 0:
        indices_neg.append(i)
    elif item == 1:
        indices_som_neg.append(i)
    elif item == 2:
        indices_neut.append(i)
    elif item == 3:
        indices_som_pos.append(i)
    elif item == 4:
        indices_pos.append(i)
# sample
fifth_sample_length = int(round(len(y) / 5.0))
negatives = np.random.choice(indices_neg, size = fifth_sample_length,
                             replace = True)
som_neg = np.random.choice(indices_som_neg, size = fifth_sample_length,
                           replace = True)
neut = np.random.choice(indices_neut, size = fifth_sample_length,
                        replace = True)
som_pos = np.random.choice(indices_som_pos, size = fifth_sample_length,
                           replace = True)
pos = np.random.choice(indices_pos, size = fifth_sample_length, replace = True)

# select the data
neg_data = [x[i] for i in negatives]
som_neg_data = [x[i] for i in som_neg]
neut_data = [x[i] for i in neut]
som_pos_data = [x[i] for i in som_pos]
pos_data = [x[i] for i in pos]
# select the y's
neg_y = [y[i] for i in negatives]
som_neg_y = [y[i] for i in som_neg]
neut_y = [y[i] for i in neut]
som_pos_y = [y[i] for i in som_pos]
pos_y = [y[i] for i in pos]

new_data = np.asarray((neg_data + som_neg_data + neut_data + som_pos_data + pos_data))
new_y = np.asarray((neg_y + som_neg_y + neut_y + som_pos_y + pos_y))


# next, we will run cross-validation
# we'll use regular kfolds
kfolds = KFold(len(new_y), 5, shuffle = True)
# need to initialize lists to keep track of our scores
accuracy = []
precision = []
recall = []
count = 1
for train_idx, test_idx in kfolds:
    # create the training set
    x_train = new_data[train_idx]
    y_train = new_y[train_idx]
    # note that we have to vectorize our phrases
    # initialize the vectorizer
    train_vectorizer = CountVectorizer()
    # build the sparse matrix
    x_train = train_vectorizer.fit_transform(x_train).toarray()
    # extract the learned vocabulary
    train_vocab = train_vectorizer.vocabulary_    
    
    # create the testing set
    x_test = new_data[test_idx]
    y_test = new_y[test_idx]
    # use the learned vocabulary to vectorize our test data
    test_vectorizer = CountVectorizer(vocabulary = train_vocab)
    x_test = test_vectorizer.fit_transform(x_test).toarray()
    
    # train the model
    model.fit(x_train, y_train)
    # and make predictions
    predictions = model.predict(x_test)
    
    # score our predictions
    acc = metrics.accuracy_score(y_test, predictions)
    prec = metrics.precision_score(y_test, predictions)
    rec = metrics.recall_score(y_test, predictions)
    print acc
    print prec
    print rec
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    print 'completed iteration %d' % count
    print '\n'
    count += 1
# end loop

print 'Average Accuracy: %f' % np.mean(accuracy)
print 'Average Recall: %f' % np.mean(recall)
print 'Average Precision: %f' % np.mean(precision)