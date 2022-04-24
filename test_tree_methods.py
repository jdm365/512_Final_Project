from tree_methods import TreeMethodClassifiers
from neural_network import train, test
import shutup
import json
import sys

if __name__ == '__main__':
    shutup.please()
    tree = TreeMethodClassifiers()
    retrain = False
    if retrain:
        xgb_acc, ada_acc, bagg_acc, rf_acc = tree.train()
        nn_acc = test()
        accuracies = {
            'XGBoost': [xgb_acc],
            'AdaBoost': [ada_acc],
            'Bagging': [bagg_acc],
            'Random Forest': [rf_acc],
            'Convolutional Neural Network': [nn_acc]
        }
        obj = json.dumps(accuracies)
        file = open('accuracies.json', 'w')
        file.write(obj)
        file.close()
    with open('accuracies.json') as json_file:
        accuracies = json.load(json_file)
    tree.plot_relative_accuracies(accuracies)