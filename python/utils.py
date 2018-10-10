import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from sklearn.metrics import confusion_matrix

def match(_id, pos=1):
    """
    Helper function for selecting dataframe columns
    """
    def inner(s):
        try:
            if s.split("^")[pos] == _id:
                return True
            return False
        except IndexError:
            return False
    
    return inner

def normalized_confusion_matrix(ground_truth, predictions):
    cm = confusion_matrix(ground_truth, predictions)
    return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

def plot_confusion_matrix(ground_truth, predictions, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = normalized_confusion_matrix(ground_truth, predictions) if normalize else \
         confusion_matrix(ground_truth, predictions)
         
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return cm