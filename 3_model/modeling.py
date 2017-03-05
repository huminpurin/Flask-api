import pandas as pd

###
### Load Data Set
###
df=pd.read_csv(
    'cs-training.csv', 
    sep=',',
    header=0)
data = df.drop(
    df.columns[0], 
    axis=1)

# Drop rows with missing column data
data = data.dropna()

###
### Convert Data Into List Of Dict Records
###
data = data.to_dict(orient='records')

###
### Seperate Target and Outcome Features
###
from sklearn.feature_extraction import DictVectorizer
from pandas import DataFrame
vec = DictVectorizer()

df_data = vec.fit_transform(data).toarray()
feature_names = vec.get_feature_names()
df_data = DataFrame(
    df_data,
    columns=feature_names)
    
outcome_feature = df_data['SeriousDlqin2yrs']
target_features = df_data.drop('SeriousDlqin2yrs', axis=1)

from sklearn import cross_validation

"""
    X_1: independent (target) variables for first data set
    Y_1: dependent (outcome) variable for first data set
    X_2: independent (target) variables for the second data set
    Y_2: dependent (outcome) variable for the second data set
"""
X_1, X_2, Y_1, Y_2 = cross_validation.train_test_split(
    target_features, outcome_feature, test_size=0.5, random_state=0)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_1,Y_1)

output = clf.predict(X_2)
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(output, Y_2)
score = clf.score(X_2, Y_2)
print("accuracy: {0}".format(score.mean()))
print(matrix)

from sklearn.externals import joblib
joblib.dump(clf, '../3_model/nb.pkl')

