import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

csv_file_object=csv.reader(open('train.csv','rb')) #load train csv file 
header=csv_file_object.next() #Skip first line as header
train_data=[] #Variable train data
for row in csv_file_object: #for every row in the csv_file object
    train_data.append(row) #add each row to train_data
train_data=np.array(train_data) #convert from list to array

#Convert all strings to interger classifers                           
train_data[train_data[0::,3]=='male',3]=1 #male =1
train_data[train_data[0::,3]=='female',3]=0#female =0

#same for embark
train_data[train_data[0::,10]=='C',10]=0
train_data[train_data[0::,10]=='S',10]=1
train_data[train_data[0::,10]=='Q',10]=2

#Fill gaps in data to make it complete
train_data[train_data[0::,4]=='',4]=np.median(train_data[train_data[0::,4]!='',4].astype(np.float)) #fill empty ages with median
train_data[train_data[0::,10]=='',10]=np.mean(train_data[train_data[0::,10]!='',10].astype(np.float))

train_data=np.delete(train_data,[2,7,9],1)

#repeat for test file
test_file_object=csv.reader(open('test.csv','rb')) #load test csv file 
header=test_file_object.next() #Skip first line as header
test_data=[] #Variable train data
for row in test_file_object: #for every row in the csv_file object
    test_data.append(row) #add each row to train_data
test_data=np.array(test_data) #convert from list to array

#Convert all strings to interger classifers                           
test_data[test_data[0::,2]=='male',2]=1 #male =1
test_data[test_data[0::,2]=='female',2]=0#female =0

#same for embark
test_data[test_data[0::,9]=='C',9]=0
test_data[test_data[0::,9]=='S',9]=1
test_data[test_data[0::,9]=='Q',9]=2

#Fill gaps in data to make it complete
test_data[test_data[0::,3]=='',3]=np.median(test_data[test_data[0::,3]!='',3].astype(np.float)) #fill empty ages with median
test_data[test_data[0::,9]=='',9]=np.mean(test_data[test_data[0::,9]!='',9].astype(np.float))

for i in xrange(np.size(test_data[0::,0])):
    if test_data[i,7]=='':
        test_data[i,7]=np.median(test_data[(test_data[0::,7]!='')&(test_data[0::,0]==test_data[i,0]),7].astype(np.float))
test_data=np.delete(test_data,[1,6,8],1)

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
print predicted_probs
test_file_object.next()
i=0
for row in test_file_object:
    row.insert(0,output[i].astype(np.uint8))
    open_file_object.writerow(row)
    i+=1


        

