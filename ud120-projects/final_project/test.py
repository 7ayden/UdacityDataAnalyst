import sys
import pickle
from time import time
from pprint import pprint
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from feature_update import remove_outliers, calc_cash_received, calc_poi_messages_pctg 
from tester import dump_classifier_and_data

# Set Classifier choice
"""
gnb = GaussianNB
svc = SVC
dtc = DecisionTreeClassifier
knc = KNeighborsClassifier
abc = AdaBoostClassifier
rfc = RandomForestClassifier
"""
clf_choice = 'knc' 
# Set parameter to run cross-validation
perform_cv = False
# Set random_state
random_state = 42

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 #'bonus',
                 'cash_received',
                 #'deferred_income',
                 #'exercised_stock_options',
                 #'expenses',
                 #'loan_advances',
                 #'long_term_incentive',
                 'poi_messages_pctg',
                 #'restricted_stock',
                 #'salary',
                 #'shared_receipt_with_poi',
                 'total_payments',
                 'total_stock_value'
                 ]

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

### Task 2: Remove outliers
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
remove_outliers(data_dict, outliers)
### Task 3: Create new feature(s)
### Fields to be considered as cash received
cash_fields = ['salary','bonus','exercised_stock_options','loan_advances']
### Add cash received fields
calc_cash_received(data_dict, cash_fields)
### Add POI message percentage
calc_poi_messages_pctg(data_dict)
### Store to my_dataset for easy export below
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Create Min/Max Scaler
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
# Scale Features
features = scaler.fit_transform(features)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# GaussianNB
if clf_choice == 'gnb':
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

# SVC
if clf_choice == 'svc':
    from sklearn.svm import SVC
    #clf = SVC(random_state = random_state)
    clf = SVC(kernel = 'linear',
              max_iter = 10000,
              random_state = random_state)

# DecisionTreeClassifier
if clf_choice == 'dtc': 
    from sklearn.tree import DecisionTreeClassifier
    #clf = DecisionTreeClassifier(random_state = random_state)
    clf = DecisionTreeClassifier(criterion = 'gini',
                                 max_depth = 5,
                                 max_features = None,
                                 min_samples_leaf = 2,
                                 min_samples_split = 2,
                                 splitter = 'best',
                                 random_state = random_state)

# KNeighborsClassifier
if clf_choice == 'knc': 
    from sklearn.neighbors import KNeighborsClassifier
    #clf = KNeighborsClassifier()
    clf = KNeighborsClassifier(algorithm = 'auto',
                               leaf_size = 30,
                               metric = 'minkowski',
                               n_neighbors = 5,
                               p = 2,
                               weights = 'uniform')

# AdaBoostClassifier
if clf_choice == 'abc': 
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(random_state = random_state)

# RandomForestClassifier
if clf_choice == 'rfc': 
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state = random_state)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
##from sklearn.cross_validation import train_test_split
##features_train, features_test, labels_train, labels_test = \
##    train_test_split(features, labels, test_size = 0.3, random_state = random_state)

if perform_cv == True:
    # Split data using StratifiedShuffleSplit
    from sklearn.cross_validation import StratifiedShuffleSplit
    cv_split = StratifiedShuffleSplit(labels, n_iter = 1, test_size = 0.3, random_state = random_state)
    for train_idx, test_idx in cv_split: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])
        
    # Cross-validation for SVC
    if clf_choice == 'svc':       
        from sklearn.cross_validation import StratifiedKFold
        from sklearn.feature_selection import RFECV
        # Recursive feature elimination with cross-validation
        svc_clf = RFECV(estimator = clf,
                      step = 1,
                      cv = StratifiedKFold(labels_train, n_folds = 10),
                      scoring = 'accuracy')
        svc_clf = svc_clf.fit(features_train, labels_train)
        print("Optimal number of features: %d" % svc_clf.n_features_)
        print(svc_clf.support_)
        print(svc_clf.ranking_)
        
    # Cross-validation for DecisionTreeClassifier
    if clf_choice == 'dtc':
        from sklearn.cross_validation import StratifiedKFold
        from sklearn.grid_search import GridSearchCV
        param_grid = {'criterion': ['gini', 'entropy'],
                      'max_features': ['auto', 'sqrt', 'log2', None],
                      'max_depth': [4, 5, 6, 7, 8, None],
                      'min_samples_split': [2, 3, 4, 5],
                      'min_samples_leaf': [1, 2, 3, 4]}
        dtc_clf = GridSearchCV(clf,
                           param_grid,
                           cv = StratifiedKFold(labels_train, n_folds = 10),
                           verbose = 0)
        dtc_clf = dtc_clf.fit(features_train, labels_train)
        print dtc_clf.best_estimator_
        # Prediction for DecisionTreeClassifier
        dtc_pred = dtc_clf.predict(features_test)
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        accuracy = accuracy_score(labels_test, dtc_pred)
        precision = precision_score(labels_test, dtc_pred)
        recall = recall_score(labels_test, dtc_pred)
        print("DecisionTreeClassifier accuracy: %f, precision: %f, recall: %f" % (accuracy, precision, recall))
        ## Fit again for feature importances
        ## Do not use clf since that is for the dump
        dtc_clf2 = clf.fit(features_train, labels_train)
        print dtc_clf2.feature_importances_

    # Cross-validation for KNeighborsClassifier
    if clf_choice == 'knc':
        from sklearn.cross_validation import StratifiedKFold
        from sklearn.grid_search import GridSearchCV
        param_grid = {'n_neighbors': np.arange(2,11),
                      'weights': ['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'metric': ['minkowski','euclidean','manhattan'],
                      'leaf_size': [5, 10, 15, 20, 25, 30]}
        #cv = StratifiedShuffleSplit(labels, n_iter = 1, random_state = random_state)
        knc_clf = GridSearchCV(clf,
                               param_grid,
                               cv = StratifiedKFold(labels_train, n_folds = 10),
                               verbose = 0)
        knc_clf = knc_clf.fit(features_train, labels_train)
        print knc_clf.best_estimator_       
        # Prediction for KNeighborsClassifier
        knc_pred = knc_clf.predict(features_test)
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        accuracy = accuracy_score(labels_test, knc_pred)
        print("KNeighborsClassifier accuracy: %f" % (accuracy))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)