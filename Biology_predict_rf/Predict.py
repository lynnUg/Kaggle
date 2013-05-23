import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

csv_file_object=csv.reader(open('train.csv','rb')) #load train csv file 
header=csv_file_object.next() #Skip first line as header
train_data=[] #Variable train data
for row in csv_file_object: #for every row in the csv_file object
    train_data.append(row) #add each row to train_data
train_data=np.array(train_data) #convert from list to array


#repeat for test file
test_file_object=csv.reader(open('test.csv','rb')) #load test csv file 
header=test_file_object.next() #Skip first line as header
test_data=[] #Variable train data
for row in test_file_object: #for every row in the csv_file object
    test_data.append(row) #add each row to train_data
test_data=np.array(test_data) #convert from list to array


print 'Training'
forest= RandomForestClassifier(n_estimators=100)

forest=forest.fit(train_data[0::,1::],train_data[0::,0])

print 'Prediciting'
print 'Writing1'
output=forest.predict(test_data)
#print output
print 'Writing'
open_file_object=csv.writer(open('firstforest.csv','wb'))

test_file_object= csv.reader(open('test.csv','rb'))
predicted_probs = [x[1] for x in forest.predict_proba(test_data)]
#print predicted_probs
test_file_object.next()
i=0
for row in output:
    open_file_object.writerow(row)
    i+=1


        

