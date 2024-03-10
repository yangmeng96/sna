import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os

def PR(Y_test, Y_scores, method, clf, seed):
    file_path = 'figs/pr/'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.figure(figsize=(6, 4))
    colors = ['purple', 'blue', 'green', 'orange', 'red']


    for i, preds in enumerate(Y_scores[:len(method)]):
      precision, recall, _ = precision_recall_curve(Y_test, preds)
      pr_auc = average_precision_score(Y_test, preds)
      plt.plot(recall, precision, label=f'{method[i]} (AUC = {pr_auc:.3f})', color=colors[i])
      
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if clf == "LR":
        plt.title('Precison-Recall curve - Logistic Regression')
    elif clf == "RF":
        plt.title('Precison-Recall curve - Random Forest')


    plt.legend(loc="lower left")
    print(f'Seed: {seed}')
    plt.savefig(f'{file_path}{clf}_{seed}_{len(method)}.png') 
    plt.show()



