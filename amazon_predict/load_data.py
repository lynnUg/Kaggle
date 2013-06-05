"""
Functions to load the dataset.
"""

import numpy as np

def read_data(file_name):
    f = open(file_name)
    #ignore header
    f.readline()
    samples = []
    target = []
    for line in f:
        line = line.strip().split(",")
        sample = [float(x) for x in line]
        samples.append(sample)
    return samples

def load():
    print "Loading data..."
    filename_train = 'train.csv'
    filename_test = 'test.csv'

    train = read_data("train.csv")
    #x = numpy.random.rand(100, 5)
    #numpy.random.shuffle(x)
    #training, test = x[:80,:], x[80:,:]
    y_train = np.array([x[0] for x in train])
    X_train = np.array([x[1:] for x in train])
    X_test = np.array(read_data("test.csv"))
    return X_train[:,(0,2,3,4,5,6,7)], y_train, X_test[:,(0,1,3,4,5,6,7,8)]

if __name__ == '__main__':

    X_train, y_train, X_test = load()
    print X_train[0,:]
    print X_test[0,:]
    #print y_train.shape
    #print X_test[:,1:]

    
