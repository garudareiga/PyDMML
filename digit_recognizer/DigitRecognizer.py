'''
Created on Nov 6, 2013

@author: raychen
'''

import os
from datetime import datetime
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklean.decomposition import PCA

class DigitRecognizer(object):
    """
    Kaggle Knowledge Competition - Digit Recognizer
    
    Classify handwritten digits using the famous MNIST data
    
    Algorithm to use:
        - K-Nearest Neighbor 
        - Random Forest
        - Logistic Regression
        - SVM
    '''    """

    def __init__(self):
        '''
        Constructor
        '''
        self.trainData = []
        self.trainLabels = []
        self.testData = []
        self.testLabels = [] 
    
    def parseTrain(self):
        ''' Parse train.csv for the training data '''
        print('Parse train.csv ...')
        with open(os.path.join('data', 'train.csv'), 'r') as csvfile:
            reader = csv.reader(csvfile)
            numrow = 0
            for row in reader:
                if numrow > 0:
                    self.trainLabels.append(row[0])
                    self.trainData.append(row[1:])
                numrow += 1
        self.trainData = np.array(self.trainData).astype(int)
        self.trainLabels = np.array(self.trainLabels).astype(int)
    
    def parseTest(self):
        ''' Parse test.csv for the testing data '''
        with open(os.path.join('data', 'test.csv'), 'r') as csvfile:
            reader = csv.reader(csvfile)
            numrow = 0
            for row in reader:
                if numrow > 0:
                    self.testData.append(row[1:])
                numrow += 1
        self.testData = np.array(self.testData).astype(int)     
        
    def dumpTestLabels(self, postfix=None):
        fname = 'benchmark'
        fname += '_{}.csv'.format(postfix) if postfix else '.csv'
        with open(fname, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ImageId', 'Label'])
            numrow = 1
            for label in self.testLabels:
                writer.writerow([numrow, label])
                numrow += 1
        
    def use_knn(self, num_neighbors=5, reserve_ratio=0.33):
        '''
        Use K-Nearest Neighbor algorithm.
        Reserve a portion of training data from the beginning for cross-validation
        '''
        reserve_num = int(self.trainData.shape[0]*reserve_ratio)
        X = self.trainData[reserve_num:]
        y = self.trainLabels[reserve_num:]
        
        #start = timeit.default_timer()
        print('Run KNN ...')
        start = datetime.now()
        neigh = KNeighborsClassifier(n_neighbors=num_neighbors, 
                                     weights='distance', 
                                     algorithm='kd_tree')
        neigh.fit(X, y)
        labels = neigh.predict(self.trainData[:reserve_num])
        #stop = timeit.default_timer()
        stop = datetime.now()
        print('KNN takes {}'.format(stop - start))
        
        error_rate = self.get_error_rate(labels)
        print('The total error rate is {}'.format(error_rate))
        return error_rate
        
    def use_pca_and_knn(self, reduce_ratio=0.25, num_neighbors=5, reserve_ratio=0.33):
        '''
        Use PCA and KNN algorithm.
        1. Reduce features from 28*28 by a ratio (0.25 by default)
        2. Reserve a portion of training data from the beginning for cross-validation
        '''
        reserve_num = int(self.trainData.shape[0]*reserve_ratio)
        X = self.trainData[reserve_num:]
        y = self.trainLabels[reserve_num:]
        
        print('Run PCA ...')
        start = datetime.now()
        pca = PCA(n_components=int(self.trainData.shape[0]*(1 - reduce_ratio)))
        X_r = pca.fit(X).transform(X)
        stop = datetime.now()
        print('PCA takes {}'.format(stop - start))
        
        print('Run KNN ...')
        start = datetime.now()
        neigh = KNeighborsClassifier(n_neighbors=num_neighbors, 
                                     weights='distance', 
                                     algorithm='kd_tree')
        neigh.fit(X_r, y)
        labels = neigh.predict(self.trainData[:reserve_num])
        stop = datetime.now()
        print('KNN takes {}'.format(stop - start))
    
        error_rate = self.get_error_rate(labels)
        print('The total error rate is {}'.format(error_rate))
        return error_rate
    
    def get_error_rate(self, labels):
        '''
        Use a portion of the training data 
        '''
        error_count = 0.0
        for num in xrange(labels.shape[0]):
            classifyLabel = labels[num]
            trainLabel = self.trainLabels[num]
            if classifyLabel != trainLabel:
                error_count += 1.0
        error_rate = error_count/labels.shape[0]
        #print('The total number of errors is {}'.format(error_count))
        return error_rate
    
if __name__ == '__main__':
    recognizer = DigitRecognizer()
    
    recognizer.parseTrain()
    error_rate_dict = {}
    #for num in range(3, 10):
    #    print('Choose n of neighbors: {}'.format(num))
    #    error_rate = recognizer.use_pca_and_knn(num)
    #    error_rate_dict[num] = error_rate
    for reduce_ratio in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        print('Choose number of components: {0.2f}'.format(1 - reduce_ratio))
        error_rate = recognizer.use_pca_and_knn(reduce_ratio)
    for num, error_rate in error_rate_dict.iteritems():
        print('K={0}: error_rate = {1:.3f}'.format(num, error_rate))
    
    #print('Parse test.csv ...')
    #recognizer.parseTest()
                        