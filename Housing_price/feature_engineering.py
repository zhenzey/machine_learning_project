#！ /bin/env/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
import sklearn
from sklearn import preprocessing


class Feature:
    def __init__(self, df):
        self.data_frame = df


    def visualization(self, column):
        """
        Visualize the scattering plot of certain feature
        :return:
        """
        x = self.data_frame[column]
        y = self.data_frame["SalePrice"]
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        plt.ylabel("SalePrice", fontsize=13)
        plt.xlabel(column, fontsize=13)
        plt.show()

    def corr_map(self):
        """
        Plot the heatmap showing the correlation of different features
        :return:
        """

        corr_mat = self.data_frame.corr()
        plt.subplots(figsize=(12, 11))
        sns.heatmap(corr_mat, vmax=1.0, square=True)
        plt.show()

    def y_distribution(self):
        """
        Plot the distribution of y to make sure that it observes Gaussian distribution, if not, transform the data using log

        :return:
        """
        y = self.data_frame["SalePrice"]
        sns.distplot(y, fit=norm)
        plt.show()

    def log_transformation(self, column):
        """
        Transform the data y(in our case, sale prices) with the form: y -> log(y + c)
        :return: c which leads to the best fit of normal distribution
        """
        # #r_square_max = 0
        # c_max = 0
        # print(self.data_frame[column])
        # for c in np.logspace(0, int(np.log10(self.data_frame[column].max())), 50):
        #     temp = np.log(c + self.data_frame[column])
        #     r_square = stats.probplot(temp)[1][2]
        #     if r_square > r_square_max:
        #         c_max = c
        #         r_square_max = r_square
        # self.data_frame[column] = np.log(self.data_frame[column] + c_max)
        # res = stats.probplot(self.data_frame[column], plot=plt)
        # plt.show()
        # return self.data_frame
        self.data_frame[column] = np.log1p(self.data_frame[column])
        return self.data_frame


    def missing_value(self, column):
        """
        Input the missing values
        rule:
        :return:
        """

        # Case 1
        # Features for which "NaN" means "None", e.p. df["Fence"] = "NaN" means the house does not have fence
        cate = np.array(
            ["Alley", "Fence", "FireplaceQu", "MiscFeature", "PoolQC", "GarageType", "GarageFinish", "GarageQual",
             "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType",
             "MSSubClass"])
        if any(cate == column):
            self.data_frame[column] = self.data_frame[column].fillna("None")

        # Case 2
        # Feature for which "NaN" means 0, e.p. df["GarageArea"] = "NaN" means the area of the garage is 0
        # Though it seems the same as case 1, 0 is a numerical value while "None" is a categorical value
        cate = np.array(
            ["GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
             "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"]
        )
        if any(cate == column):
            self.data_frame[column] = self.data_frame[column].fillna(0)

        # Case 3
        # Feature which is skewed, so we can give the value which is most frequent to replace "NaN"
        cate = np.array(
            ["MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType"]
        )
        if any(cate == column):
                self.data_frame[column] = self.data_frame[column].fillna(self.data_frame[column].mode()[0])

        # Case 4, 5, 6, 7
        # Feature --- Utilities, for nearly all examples,  categorical values are Allpub, so we can drop it
        # Feature --- Functional, according to data description, "NaN" means typical
        # Feature --- LotFrontage, the houses in the same neighborhood have similar lot frontage, we assume the lot
        # frontage observes Gaussian distribution
        if column == "Utilities":
            self.data_frame = self.data_frame.drop(["Utilities"], axis=1)
        elif column == "Functional":
            self.data_frame[column] = self.data_frame[column].fillna("Typ")
        elif column == "LotFrontage":
            self.data_frame[column] = self.data_frame.groupby("Neighborhood")[column].transform(
                lambda x: x.fillna(np.random.normal(x.mean(), x.std()))
            )
        return self.data_frame

    def num2cat(self, column):
        """
        Transfer the numerical feature into categorical feature for those features that are really categorical
        The features includes MSSubClass, overall rank(optional), year and month build/reconstructed
        e.p. MSSubClass, details can be found in data_description.txt

        :return:
        """

        cate = np.array(["MSSubClass", "OverallQual", "OverallCond", "YearRemodAdd", "YearBuilt", "GarageYrBlt",
                         "MoSold", "YrSold"
        ])
        #cate = np.array(["MSSubClass", "OverallQual", "MoSold", "YrSold"])
        if any(cate == column):
            self.data_frame[column] = self.data_frame[column].astype(str)

        return self.data_frame

    def labelencoder(self, column):
        """
        Using sklearn.preprocessing.labelEncoder to transfer categorical value into numerical values
        Sklearn.preprocessing.labelEncoder reference:https://scikit-learn.org/stable/modules/generated/
        sklearn.preprocessing.LabelEncoder.html

        :return:
        """
        label = preprocessing.LabelEncoder()

        cate = np.array(["FireplaceQu", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", 
        "ExterQual", "ExterCond","HeatingQC", "PoolQC", "KitchenQual", "BsmtFinType1", 
        "BsmtFinType2", "Functional", "Fence", "BsmtExposure", "GarageFinish", "LandSlope",
        "LotShape", "PavedDrive", "Street", "Alley", "CentralAir", "MSSubClass", "OverallCond", 
        "YrSold", "MoSold"])
        if any(cate == column):
            label.fit(list(self.data_frame[column].values))
            self.data_frame[column] = label.transform(list(self.data_frame[column].values))
        return self.data_frame

    def skewed_feature(self, column):
        """
        Dealing with the skewed features using Box Cox transformation
        Don't worry about the math
        Ps: log transformation is also working, but when dealing with negative skewed feature, the data should
        be reflected first

        :return:
        """

        lam = 0.15
        self.data_frame[column] = boxcox1p(self.data_frame[column], lam)
        return self.data_frame












