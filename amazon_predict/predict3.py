from __future__ import division
import numpy as np
import load_data
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import csv
if __name__ == '__main__':
   X, y, X_submission = load_data.load()
   n_folds = 10
   skf = list(StratifiedKFold(y, n_folds))
   forest=  RandomForestClassifier(n_estimators=100)
   dataset_blend_train = np.zeros((X.shape[0], 1))
   print dataset_blend_train.shape
   #for i, (train, test) in enumerate(skf):
   #print "Fold", i
   #print train.shape,test.shape
   X_train = X[:21513,:]
   y_train = y[:21513]
   X_test = X[21513:,:]
   y_test = y[21513:]
   forest.fit(X_train, y_train)
   p=forest.predict(X_test)
   print 'Train Accuracy:', (p == y_test).mean() * 100
            #break
            #print dataset_blend_train
   
   
