#! /bin/env/python3

# import necessary libraries
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import norm, skew
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

class regModel:

    def def_model(self, model):
        """
        Define regression models and set parameters
        :param model:
        :return:
        """

        if model == "lasso":
            return make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
        elif model == "ENet":
            return make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
        elif model == "KRR":
            return KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
        elif model == "GBoost":
            return GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                             min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5)

        elif model == "Lgb":
            return lgb.LGBMRegressor(objective='regression',num_leaves=5, learning_rate=0.05, n_estimators=720,
                                     max_bin=55, bagging_fraction=0.8, bagging_freq=5, feature_fraction=0.2319,
                                     feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf=6,
                                     min_sum_hessian_in_leaf=1)
        else:
            raise NameError('The model is not included or the name is incorrect')

    def K_fold_CV(self, model, modeling, data_tr, y_tr):
        """
        Using K-fold cross validation to compare different models
        :param model: Here we select five different models: LASSO Regression; Elastic Net Regression;
        Kernel Ridge Regression; Gradient Boosting Regression; LightGBM

        :return:
        """
        n_folds = 5  # 5 runs of cross validation
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(data_tr.values)
        rmse = np.sqrt(-cross_val_score(modeling, data_tr.values, y_tr.values, scoring="neg_mean_squared_error", cv=kf))
        print(model + "\tscore: {:.4f} ({:.4f})\n".format(rmse.mean(), rmse.std()))
        return rmse


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    # Meta-model: the basic idea is using K-folds to train several base models and then treat the predictions as input
    # and the ground truth as output, using meta-model to train
    # More details please check the following website: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

    def __init__(self, base_models, meta_model, n_folds = 5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        """
        For all regression algorithm class in sklearn, fit function is to fit paramters of certain models
        :return:
        """
        # Initialization
        self.base_models_ = [[] for x in self.base_models] # self.base_models = [[],...,[]]
        self.meta_model_ = clone(self.meta_model)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # train the cloned models and give the prediction which is the input of meta-model
        prediction_matrix = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_idx, test_idx in kf.split(X):
                model_ = clone(model)  # for every iteration, model_ is different
                model_.fit(X[train_idx], y[train_idx])
                self.base_models_[i].append(model_)
                prediction_matrix[test_idx, i] = model_.predict(X[test_idx])

        self.meta_model_.fit(prediction_matrix, y)
        return self

    def predict(self, X):
        """
        For all regression algorithm class in sklearn, predict function is to predict output considering certain input.
        Here we get the input by averaging over different base models in one iteration, so the shape of input is (X.shape[0], num_iteration)
        :return:
        """
        meta_features = np.column_stack(
            np.column_stack([model.predict(X) for model in models]).mean(axis=1) for models in self.base_models_)
        return self.meta_model_.predict(meta_features)



