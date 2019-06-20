#! /bin/env/python3

# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
import feature_engineering
from feature_engineering import Feature
import modeling
from modeling import regModel
from modeling import StackingAveragedModels

if __name__ == "__main__":
    #### Read data from csv files
    data_tr = pd.read_csv('./input_data/train.csv')
    data_tt = pd.read_csv('./input_data/test.csv')
    Id_tt = data_tt["Id"]
    # Giving the info of data
    print("Training set info: ")
    print(data_tr.info())
    print("\n")
    print("Test set info: ")
    print(data_tt.info())
    print("\n")


    #### Visualization of training data
    # e.p. 'GrLivArea'
    #feature = Feature(data_tr, "GrLivArea")
    feature_tr = Feature(data_tr)
    feature_tr.visualization("GrLivArea")
    feature_tr.corr_map() # correlation matrix between different features(only numerical values)
    feature_tr.y_distribution()


    #### Feature engineering of all data
    # Drop useless ID
    data_tr.drop("Id", axis=1, inplace=True)
    data_tt.drop("Id", axis=1, inplace=True)
    # Deleting outliers
    data_tr = data_tr.drop(data_tr[(data_tr['GrLivArea'] > 4000) & (data_tr['SalePrice'] < 300000)].index)
    # transform the skewed sale prices into Gaussian distribution
    data_tr = feature_tr.log_transformation("SalePrice")
    y_tr = data_tr["SalePrice"]
    all_data = pd.concat((data_tr, data_tt)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)


    feature = Feature(all_data)
    for column in all_data.columns:
        # Dealing with the missing value
        all_data = feature.missing_value(column)
        # Transfer certain numerical values which are actually categorical
        all_data = feature.num2cat(column)
        # Transfer label into continuous categories
        all_data = feature.labelencoder(column)
    # Dealing with skewed features
    numerical_idx = all_data.dtypes[all_data.dtypes != "object"].index
    skewness = all_data[numerical_idx].skew(axis=0)
    skewed_idx = skewness.index
    print("Skewed feature info:\n")
    print(skewness)
    for idx in skewed_idx:
        if abs(skewness[idx]) > 0.75:
            all_data = feature.skewed_feature(idx)
    # add one more important feature: the total area of the house
    all_data["TotalSF"] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    # Getting dummy categorical feature
    all_data = pd.get_dummies(all_data)
    print(all_data)
    # Split train dataset and test dataset
    ntrain = data_tr.shape[0]
    train = all_data[:ntrain]
    test = all_data[ntrain:]

    #### Modeling
    Modeling = regModel()
    # LASSO Regression
    lasso = Modeling.def_model("lasso")
    score1 = Modeling.K_fold_CV("lasso", lasso, train, y_tr)
    # Elastic Net Regression
    ENet = Modeling.def_model("ENet")
    score2 = Modeling.K_fold_CV("ENet", ENet, train, y_tr)
    # Kernel Ridge Regression
    KRR = Modeling.def_model("KRR")
    score3 = Modeling.K_fold_CV("KRR", KRR, train, y_tr)
    # Gradient Boosting Regression
    GBoost = Modeling.def_model("GBoost")
    score4 = Modeling.K_fold_CV("GBoost",GBoost, train, y_tr)
    # "Lgb"
    Lgb = Modeling.def_model("Lgb")
    score5 = Modeling.K_fold_CV("Lgb", Lgb, train, y_tr)


    #### Stacking averaged regression
    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR),
                                                     meta_model=lasso)
    score6 = Modeling.K_fold_CV("stacked_averaged_models", stacked_averaged_models, train, y_tr)


    #### Submission
    stacked_averaged_models.fit(train.values, y_tr.values)
    stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
    sub = pd.DataFrame()
    sub['Id'] = Id_tt
    sub['SalePrice'] = stacked_pred
    sub.to_csv('submission.csv', index=False)



