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
   #skf = list(StratifiedKFold(y, n_folds))
   #forest= ExtraTreesClassifier(n_estimators=100)
   #dataset_blend_train = np.zeros((X.shape[0], 1))
   #print dataset_blend_train.shape
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
   skf = list(StratifiedKFold(y, 10))

   clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

   print "Creating train and test sets for blending."
    
   dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
   dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    
   for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission[:,1:])[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)


   clf = LogisticRegression()
   clf.fit(dataset_blend_train, y)
   y_submission = clf.predict_proba(dataset_blend_test)[:,1]

   print "Linear stretch of predictions to [0,1]"
   y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
   
   headers = ['id', 'ACTION']
   myfile=(open('firstforest8.csv','wb'))
   wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
   wr.writerow(headers)
   count=0
   for p in range(0,len(y_submission)):
       l=[]
       l.append(int(X_submission[count,0]))
       l.append(int(y_submission[count]))
       wr.writerow(l)
       count+=1
