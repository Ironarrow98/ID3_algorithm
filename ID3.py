# Due to personal reason, I am unable to finish Full ID3 Algorithm
#     So I use sklearn.tree.DecisionTreeClassifier(criterion=’entropy’)
#     I sincerely apologize
#     the following contains the answer 
#     code is simple to run, if need set_b data, put it on line 91
#     but require graphviz and pydotplut module to show the decision tree graph

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import numpy as np
import pydotplus

# Read data from set_a.csv
data = pd.read_csv('./set_a.csv', header = None)
data.columns = ['sepal_length', "sepal_width", "petal_length", "petal_width", "class"]

######################################## Question 1 ########################################
# Build a decisiontree
classifier = DecisionTreeClassifier(criterion = "entropy")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
tree = DecisionTreeClassifier(criterion = 'entropy')
clf = tree.fit(X, y)
print("The accuracy is ", clf.score(X, y))

# Print out decision tree, and save it in tree.png in the same folder
# Require graphviz and pydotplut module 
dot_data = StringIO()
export_graphviz(clf, out_file = dot_data, filled = True, rounded = True, special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png('./tree.png')


######################################## Question 2 ########################################
# Modified version of decision tree algorithm which takes a maximum depth as input
# Choose Maximum Depth = 10
maximum_depth = 10
# Perform ten_fold cross_validation
k_fold = KFold(n_splits = 10, random_state = 66536)
# Recode accuracy
all_train_accuracy = []
all_val_accuracy = []
# Note: depth cannot be 0, so we cannot start i = 0
for i in range(1, 1 + maximum_depth):
    tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = i)
    train_accuracy = []
    val_accuracy = []
    for train_index, test_index in k_fold.split(X,):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = tree.fit(X_train, y_train)
        train_accuracy.append(clf.score(X_train, y_train))
        val_accuracy.append(clf.score(X_test, y_test))
    all_train_accuracy.append(np.mean(train_accuracy))
    all_val_accuracy.append(np.mean(val_accuracy))
    
# Plot average prediction accuracy on training set VS max_depth
#     and plot average prediction accuracy on validation set VS max_depth
plt.figure(figsize=(16, 9))
x_index = list(range(1, 11))
plt.plot(x_index, all_train_accuracy, label = "Training set average prediction accuracy")
plt.plot(x_index, all_val_accuracy, label = "Validation set average prediction accuracy")
plt.xlabel("Maximum Depth")
plt.ylabel("Average Prediction Accuracy")
plt.legend(loc = "upper left")
plt.show()
# From the graph it is obivious to observe that the maximum depth of decision tree 
#     that maximizes the average prediction accuracy of the generated tree on the validation set is 3
# So I choose max_depth = 3 to graph the decision tree

# Print out a decision tree with the best maximum depth using the entire data set A. 
# The image of the tree is saved to best_validation_accuracy_tree.png in the same folder
# Require graphviz and pydotplut module 
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
clf = tree.fit(X, y)
dot_data = StringIO()
export_graphviz(clf, out_file = dot_data, filled = True, rounded = True, special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png('./best_validation_accuracy_tree.png')
# From the graph it is easy to observe that 
#    the prediction accuracy of the decision tree with the best maximum depth on data set A is 81%


######################################## Question 3 ########################################
# if set_b data exist, load it and predict
# set_b has the same format of set_a
set_b_data = '' # put set_b here
try:
	assert len(set_b_data) != 0
	data_test = pd.read_csv(set_b_data, header = None)
	pred = clf.predict(data_test)
except:
	print("Need sepecific set_b data")

