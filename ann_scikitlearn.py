# -------------------------------------------------------------
# 4220. Machine learning - Adult data set project - Spring 2017
# scikit-learning tool-box ann model
# Xiaoling Zheng
# date: 4/23/2017
# -------------------------------------------------------------
import numpy
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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
    
    # input dataset
    X = x_features
    # output dataset
  #  y = np.array([y])

    # apply ann #
    clf = MLPClassifier(hidden_layer_sizes=(7,4), activation='relu', solver='lbfgs', alpha=0.00001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=300, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    # apply ann #

    clf.fit(X,y)

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

 #   print n_classes

 #   print sum(predict)
  #  print sum(y)
    


if __name__ == "__main__":
    main()
