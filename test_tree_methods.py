from tree_methods import TreeMethodClassifiers
from neural_network import train, test
import shutup
import sys

if __name__ == '__main__':
    shutup.please()
    tree = TreeMethodClassifiers()
    xgb_acc, ada_acc, bagg_acc, rf_acc = tree.train()
    nn_acc = test()
    accuracies = {
        'XGBoost': xgb_acc,
        'AdaBoost': ada_acc,
        'Bagging': bagg_acc,
        'Random Forest': rf_acc,
        'Convolutional Neural Network': nn_acc
    }
    tree.plot_relative_accuracies(accuracies)