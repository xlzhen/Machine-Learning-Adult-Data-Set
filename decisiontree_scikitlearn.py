# -------------------------------------------------------------
# 4220. Machine learning - Adult data set project - Spring 2017
# scikit-learning tool-box decision tree model
# Xiaoling Zheng
# date: 4/23/2017
# -------------------------------------------------------------

import numpy as np
import panda as pd
from sklearn import tree
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def main():
    # read in training set #
    with open("converted_adult_set_one.txt", "r") as infile:
        data = infile.read()
    # read in training set #
        
    file_list = data.splitlines() # convert file into lists by each line

    i = 0
    x_features = []
    y = []
    
    while (i < len(file_list)):
	input_list = file_list[i].split()
	input_list = [float(j) for j in input_list]
	y.append(input_list[-1])
	del input_list[-1]
	x_features.append(input_list)
	i += 1

    X = x_features
    Y = y
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    # read in testing set #
    with open("converted_test_adult_one.txt", "r") as infile:
        data = infile.read()
    # read in testing set #

    file_list = data.splitlines() # convert file into lists by each line    

    i = 0
    test_x_features = []
    result_y = []
    
    while (i < len(file_list)):
	input_list = file_list[i].split()
	input_list = [float(j) for j in input_list]
	result_y.append(input_list[-1])
	del input_list[-1]
	test_x_features.append(input_list)
	i += 1
	
    predict_y = clf.predict(test_x_features)
    
   # Load to File
   # i = 0
   # decision_tree = open('decision_tree.txt', "w")
    
   # while(i < len(predict_y)):
   #     decision_tree.write(predict_y[i])
   #     decision_tree.write("\n")
   #     i += 1
        
   # decision_tree.close()
   # Load to File

    fpr, tpr, _ = roc_curve(result_y, predict_y)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    
    lw = 2
    
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.6f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    plt.show()

    
        

    
if __name__ == "__main__":
    main()

#----------------#
#    References  #
#----------------#
#http:// stackoverflow.com/questions/5749195/how-can-i-split-and-parse-a-string-in-python
#http://stackoverflow.com/questions/10712002/create-an-empty-list-in-python-with-certain-size
#http://stackoverflow.com/questions/4455076/how-to-access-the-ith-column-of-a-numpy-multidimensional-array
