"""
Authors: Mateusz Pioch s21331, Stanisław Dominiak s18864
The program first classifies wheat seeds (their type based on their area, perimeter, compactness,
                                          kernel length and width, asymmetry coefficient and the length
                                          of the kernel groove),
then the lenses (based on the age of the patient (1: young, 2: pre-presbyopic, 3: presbyopic),
                 spectacle prescription (1: myope, 2: hypermetrope),
                 astigmaticism status (1 for no, 2 for yes),
                 and if the tear production rate is reduced (1) or normal (2).)
"""

import pandas as pd
import pydotplus as pdp
from IPython.display import Image
from sklearn import tree
from sklearn import metrics
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from six import StringIO

"""
Load data from CSV file for the wheat seeds analysis
https://archive.ics.uci.edu/ml/datasets/seeds <- source
"""
df = pd.read_csv("wheat_seeds_dataset.csv", delimiter="\t")

"""
x axis is for input data, y for output (seed class)
"""
x = df.drop('class', axis=1)
y = df['class']

"""
Split the data to approx. 67% given to training and 33% to test purposes.
"""
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=33)

"""
Prepare classifier for the decision tree
"""
clasifier = DecisionTreeClassifier()
clasifier = clasifier.fit(X_train, Y_train)
y_pred = clasifier.predict(X_test)

"""
Export decision tree to PNG
"""
dot_file = StringIO()

export_graphviz(
    clasifier,
    out_file=dot_file,
    filled=True,
    rounded=True,
    feature_names=list(x.columns),
    class_names=np.array(y.unique()).astype('str').tolist()
)

graph = pdp.graph_from_dot_data(dot_file.getvalue())
graph.write_png('wheat_seeds_decision_tree.png')
Image(graph.create_png())

"""
Perform SVM classification with linear kernel
"""
svc = svm.SVC(kernel='linear').fit(x,y)
svc.fit(X_train, Y_train)
"""
Check accurancy of the model
"""
y_pred = svc.predict(X_test)
print("Wheat grain accuracy is: ",metrics.accuracy_score(Y_test, y_pred))



"""
Load data from CSV file for the lens analysis - basically the same, but for the lenses

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
https://archive.ics.uci.edu/ml/datasets/Lenses
"""
df2 = pd.read_csv("lenses.csv", delimiter=";")

"""
x axis is for input data, y for output (seed class)
"""
x2 = df2.drop('class', axis=1)
y2 = df2['class']

"""
Split the data to approx. 67% given to training and 33% to test purposes.
"""
X2_train, X2_test, Y2_train, Y2_test = train_test_split(x2, y2, test_size=0.33, random_state=33)

"""
Prepare classifier for the decision tree
"""
clasifier2 = DecisionTreeClassifier()
clasifier2 = clasifier.fit(X2_train, Y2_train)
y_pred2 = clasifier.predict(X2_test)

"""
Export decision tree to PNG
"""
dot_file2 = StringIO()

export_graphviz(
    clasifier2,
    out_file=dot_file2,
    filled=True,
    rounded=True,
    feature_names=list(x2.columns),
    class_names=np.array(y2.unique()).astype('str').tolist()
)

graph2 = pdp.graph_from_dot_data(dot_file2.getvalue())
graph2.write_png('lenses_decision_tree.png')
Image(graph2.create_png())

"""
Perform SVM classification with linear kernel
"""
svc = svm.SVC(kernel='linear').fit(x2,y2)
svc.fit(X2_train, Y2_train)
"""
Check accurancy of the model
"""
y2_pred = svc.predict(X2_test)
print("Lens accuracy is: ",metrics.accuracy_score(Y2_test, y2_pred))


