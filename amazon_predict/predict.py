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
   #X_train = X[train]
   #y_train = y[train]
   #X_test = X[test]
   #y_test = y[test]
   #forest.fit(X_train, y_train)
   #p=forest.predict(X_test)
   # print 'Train Accuracy:', (p == y_test).mean() * 100
            #break
            #print dataset_blend_train
   
   forest.fit(X, y)
   output=forest.predict(X_submission[:,1:])
   headers = ['id', 'ACTION']
   myfile=(open('firstforest12.csv','wb'))
   wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
   wr.writerow(headers)
   count=0
   for p in output:
       l=[]
       l.append(int(X_submission[count,0]))
       l.append(int(p))
       wr.writerow(l)
       count+=1
