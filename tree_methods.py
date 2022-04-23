import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
import time
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hyperopt import Trials, fmin, hp, tpe
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix, \
                            precision_recall_curve, auc, roc_curve, recall_score, \
                            classification_report \

class TreeMethodClassifiers:
    def __init__(self, training_filename='official_data/train.csv', 
                        testing_filename='official_data/test.csv', sample=True):
        self.training_filename = training_filename
        self.testing_filename = testing_filename
        self.sample = sample
        self.train_data, self.test_data = self.get_train_test()
        self.x_train, self.y_train = self.train_data
        self.x_test, self.y_test = self.test_data

        self.xgb_config = {
            'n_estimators': 10,
            'max_depth': hp.quniform('max_depth', 3, 18, 1),
            'gamma': hp.uniform('gamma', 1, 9),
            'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1)
        }

        self.methods = {
            'xgboost': self.xgboost
        }

    def upsample(self, df):
        df_minority = df[df['Response'] == 1]
        df_majority = df[df['Response'] == 0]
        df_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority)
        )
        return pd.concat([df_majority, df_upsampled])


    def transform(self, df):
        df['Gender'] = np.where(df["Gender"] == "Male", 1, 0)
        new_names = {
            '1-2 Year': 'one_to_two_years', 
            '> 2 Years': 'more_than_two_years', 
            '< 1 Year': 'less_than_one_year'
            }
        for cat in pd.unique(df['Vehicle_Age']):
            df[new_names[cat]] = np.where(df["Vehicle_Age"] == cat, 1, 0)
        df.drop(columns=['Vehicle_Age'], inplace=True)
        df['Vehicle_Damage'] = np.where(df["Vehicle_Damage"] == "Yes", 1, 0)

        cols_for_std_scl = ['Age', 'Vintage']
        cols_for_min_max_scl = ['Annual_Premium']

        std_scl = StandardScaler()
        min_max_scl = MinMaxScaler()

        df[cols_for_std_scl] = std_scl.fit_transform(df[cols_for_std_scl])
        df[cols_for_min_max_scl] = min_max_scl.fit_transform(df[cols_for_min_max_scl])
        return df

    def get_train_test(self):
        train_df = pd.read_csv(self.training_filename)
        test_df = pd.read_csv(self.testing_filename)
        train_df.set_index('id', inplace=True)
        test_df.set_index('id', inplace=True)

        df_train = self.transform(train_df)
        if self.sample:
            df_train = self.upsample(df_train)
        df_test = self.transform(test_df)

        x_train = df_train.drop(columns=['Response'])
        y_train = df_train['Response']

        x_test = df_test.drop(columns=['Response'])
        y_test = df_test['Response']

        order = x_train.columns.tolist()
        x_test = x_test[order]
        return (x_train, y_train), (x_test, y_test)

    def xgboost(self, config):
        clf = xgb.XGBClassifier(n_estimators=10,
                                max_depth=int(config['max_depth']),
                                gamma=int(config['gamma']),
                                min_child_weight=int(config['min_child_weight']),
                                colsample_bytree=int(config['colsample_bytree']))
        eval = [self.train_data, self.test_data]
        clf.fit(self.x_train, self.y_train, eval_set=eval, eval_metric='auc',
                verbose=False)
        preds = clf.predict(self.x_test)
        y_score = clf.predict_proba(self.x_test)[:, 1]
        accuracy = accuracy_score(self.y_test, preds > 0.5)
        Roc_Auc_Score = roc_auc_score(self.y_test, y_score)
        loss = -Roc_Auc_Score
        #print ("ROC-AUC Score:", Roc_Auc_Score)
        #print ("Accuracy:", accuracy)
        return loss

    def adaboost(self):
        print('..Training adaboost...')
        n_estimators = [10, 50, 100, 500, 1000]
        accuracy_n_est = []
        for N in n_estimators:
            clf = AdaBoostClassifier(n_estimators=N)
            clf.fit(self.x_train, self.y_train)
            preds = clf.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, preds > 0.5)
            accuracy_n_est.append(accuracy)
            print (f'Accuracy {N} trees:', accuracy)
        return dict(zip(n_estimators, accuracy_n_est))

    def bagging(self):
        print('..Training bagging...')
        n_estimators = [25]
        accuracy_n_est = []
        for N in n_estimators:
            start = time.time()
            clf = BaggingClassifier(base_estimator=SVC(), n_estimators=N, max_samples=0.01)
            clf.fit(self.x_train, self.y_train)
            preds = clf.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, preds > 0.5)
            accuracy_n_est.append(accuracy)
            print (f'Accuracy {N} trees:', accuracy, f'Time taken {int(time.time()-start)} seconds')
        return dict(zip(n_estimators, accuracy_n_est))

    def randomForest(self):
        print('..Training Random Forest...')
        max_depths = [1, 2, 3, 4, 5]
        accuracy_n_est = []
        for depth in max_depths:
            clf = RandomForestClassifier(max_depth=depth)
            clf.fit(self.x_train, self.y_train)
            preds = clf.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, preds > 0.5)
            accuracy_n_est.append(accuracy)
            print (f'Accuracy depth {depth}:', accuracy)
        return dict(zip(max_depths, accuracy_n_est)), clf

    def tune_hyperparameters(self, method='xgboost'):
        trials = Trials()
        optimal_hyperparameters = fmin(fn = self.methods[method],
                        space = self.xgb_config,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)
        self.xgb_config = optimal_hyperparameters
        return self.xgboost(self.xgb_config)

    def generate_feature_imp_plot(self, model):
        feature_names = [f'feature {i}' for i in range(self.x_train.shape[1])]
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        forest_importances = pd.Series(importances, index=feature_names)

        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        fig.savefig('plots/feature_importances.png')

    def plot_relative_accuracies(self, accuracies:dict):
        fig = plt.bar(*zip(*accuracies.items()))
        fig.savefig('plots/relative_accuracies.png')


    def train(self, method='xgboost'):
        print('..Training xgboost...')
        xgb_best_accuracy = self.tune_hyperparameters(method)
        adaboost_accuracies = self.adaboost()
        bagging_accuracies = self.bagging()
        randomForest_accuracies, rf_model = self.randomForest()

        self.generate_feature_imp_plot(rf_model)
        return xgb_best_accuracy, \
                max(adaboost_accuracies, key=adaboost_accuracies.get), \
                max(bagging_accuracies, key=bagging_accuracies.get), \
                max(randomForest_accuracies, key=randomForest_accuracies.get)


