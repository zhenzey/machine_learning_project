#! /bin/bash/python 3

## Import necessary libraries
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
#import keras
#from keras.utils.np_utils import to_categorical
import feature_engineering
from feature_engineering import filling_missing_values



if __name__ == "__main__":
    ## Read csv files
    train = pd.read_csv("./input_data/train.csv")
    test = pd.read_csv("./input_data/test.csv")
    Y_train = train["label"].values

    ## Initialize some constants may be useful
    m = int(math.sqrt(len(test.columns))) # Number of columns and rows for images
    ntrain = train.values.shape[0]
    ntest = test.values.shape[0]

    ## Visualization of digits images
    sample = 0
    img_matrix = train.iloc[sample][1:].values.reshape(m, m)
    plt.imshow(img_matrix, cmap="Greys")
    plt.title(train.iloc[sample][0])
    plt.show()

    ## Feature engineering
    # Checking missing value
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['label'], axis=1, inplace=True)
    miss_data = all_data.isna().values.reshape(all_data.shape[0], m, m)
    # Reshape vector representation of images into matrices representation
    all_data = all_data.values.reshape(all_data.shape[0], m, m)
    miss_rate = sum(sum(sum(np.where(miss_data == True, 1, 0)))) / (all_data.shape[0] * m * m)
    if miss_rate == 0.0:
        print("Congratulation! No missing values")
    else:
        print("Missing rate is %f" % miss_rate)
        all_data = filling_missing_values(all_data, miss_data)

    all_data = all_data.reshape(-1, m**2)
    train = all_data[:ntrain]
    test = all_data[ntrain:]

    ## Normalization
    train = train / 255.0
    test = test / 255.0

    ## One hot encoding of labels
    #Y_train = to_categorical(Y_train, num_classes=10)

    ## Neural network implementation
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1)
    clf.fit(train, Y_train)
    Y_test = clf.predict(test)
    print(Y_test)
    sub = pd.DataFrame()
    sub['ImageId'] = np.arange(ntest) + 1
    sub['Label'] = Y_test
    sub.to_csv('submission.csv', index=False)







