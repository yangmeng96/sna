import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os


def ROC(Y_test, Y_scores, method, LR = True):
    file_path = 'ROCfigs/'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if LR == True:
       type = "logistic regression"
    elif LR == False:
       type = "random forest"
    
    
    plt.figure(figsize=(10, 6))

    for i, preds in enumerate(Y_scores):
      fpr, tpr, _= roc_curve(Y_test, preds)
      roc_auc = auc(fpr, tpr)
      plt.plot(fpr, tpr, label=f'{method[i]} (AUC = {roc_auc:.2f})')

   
    # plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic'+type)
    plt.legend(loc="lower right")
    

    #plt.savefig(file_path + type + "_" + iter + ".png") 
    plt.show()

