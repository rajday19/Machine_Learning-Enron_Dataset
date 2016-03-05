#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from sklearn import cross_validation
from sklearn.svm import SVC
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report	
from sklearn import grid_search
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest

### FILE REASON - GRIDSEARCH CV

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','exercised_stock_options','long_term_incentive','fraction_emails_to_poi_from_person','fraction_emails_from_poi_to_person'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Defining number of people with valid financial fields
Valid_people_count = 0
Total_people = len(data_dict.keys())
for keys,values in data_dict.items():
    if values["salary"]!= "NaN" and values["restricted_stock"]!="NaN" and values["exercised_stock_options"]!= "NaN":

	Valid_people_count += 1

### Task 2: Remove outliers

# The outlier observed was the "TOTAL" key in the data_dict dictionary. Removing the same.

print "The outlier is: ", data_dict["TOTAL"]
data_dict.pop("TOTAL")


### Task 3: Create new feature(s)
# Adding a feature called Stocks_sold which is the sum of "exercised_stock_options" and "restricted_stock". The intuition is that, since most of the POI's knew that the company was making paper profits, they would have sold most of their stock when it was high. Hence a higher amount  made might be related to POI's

for keys,values in data_dict.items():
    if values["salary"]!= "NaN" and values["restricted_stock"]!="NaN" and values["exercised_stock_options"]!= "NaN":
	values["Stocks_sold"] = (values["exercised_stock_options"] + values["restricted_stock"])
    else:
	values["Stocks_sold"] = 0	

#Adding another feature from the mini-projects from Katie - fraction_emails_from_poi_to_person, fraction_emails_to_poi_from_person

for keys,values in data_dict.items():
    if values["from_poi_to_this_person"] != "NaN" and values["from_this_person_to_poi"]!="NaN":
	values["fraction_emails_from_poi_to_person"] = float(values["from_poi_to_this_person"])/float(values["from_messages"])*100
	values["fraction_emails_to_poi_from_person"] = float(values["from_this_person_to_poi"])/float(values["to_messages"])*100
    else:
	values["fraction_emails_from_poi_to_person"] = 0
	values["fraction_emails_to_poi_from_person"] = 0
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)




print ""
print ""
print "Length of feaures is: ", len(features)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline


max_leaf_nodes = 10
min_samples_split = 5
max_depth = None
Num_features=6
"""
clf = make_pipeline(RandomizedPCA(), GaussianNB())
clf.fit(features,labels)
pred = clf.predict(features)

acc = accuracy_score(labels,pred)
print "The accuracy of NB on same dataset is: ", acc

"""



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print "The size of training set is: ", len(features_train)
print "The size of test set is: ", len(features_test)

"""
clf = make_pipeline(RandomizedPCA(), GaussianNB())
print "Classifier is: ", clf
clf.set_params(randomizedpca__n_components=n_components,randomizedpca__whiten=True)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(labels_test,pred)

print "The accuracy of NB on different training and testing datasets is: ", acc

for i in range(len(pred)):
    print "The prediction and actual labels are: ", pred[i], labels_test[i]

"""
"""

clf = make_pipeline(PCA(), SVC())
clf.fit(features_train, labels_train)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(labels_test,pred)

for i in range(len(pred)):
    print "The prediction and actual labels are: ", pred[i], labels_test[i]

#print "The accuracy of SVM on different training and testing datasets is: ", acc

"""


from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import f_classif


select= SelectKBest(f_classif,k=4)
Ada = AdaBoostClassifier(algorithm="SAMME",base_estimator = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, min_samples_split = min_samples_split))

steps = [("Feature_selection",select),
	 ("Adaboost",Ada)]

pipeline = Pipeline(steps)


parameters=dict(Feature_selection__k = [3,4,5,6],
		Adaboost__n_estimators = [100,200],
		Adaboost__algorithm = ["SAMME"])


clf = grid_search.GridSearchCV(pipeline, param_grid=parameters)
#clf.set_params(Ada__base_estimator = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, min_samples_split = min_samples_split))

print "The classifier is: ", clf

print ""

clf.fit(features_train,labels_train)
print "The classifier parameters are: ",clf.best_estimator_

pred = clf.predict(features_test)
report = classification_report(labels_test,pred)


print ""
print "The metrics for Decision tree on different training and testing datasets is: "
print report



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)



