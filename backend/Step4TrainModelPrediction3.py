## Importing the libraries
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from joblib import dump, load
import warnings

import regression_multiLinear
import regression_polynomial
import regression_knn
import regression_decisionTree
import regression_randomForest
import regression_svrLinear
import regression_svrNonLinear
import regression_xgb



def train_model_prediction3(): 
    path_fig = "./Figure/prediction3/"
    path_model = "./Model/prediction3/"

    df = pd.read_csv('./Data/game_info_cleaned.csv')


    raw_data2 = pd.read_csv('./Data/game_info.csv')
    raw_data2

    df2 = raw_data2.copy()
    df2['Total Sales'] = df2['Total Sales'].fillna(0)
    df2 = df2.drop(['Units Sold'], axis = 1)
    df2 = df2[df2['Total Sales'] != 0]
    df2 = df2[df2['Total Sales'] != '0.00m']
    df2

    # Đổi tên cột 'Publishers(s)' thành 'Publishers' trong DataFrame df2
    df2.rename(columns={'Publishers(s)': 'Publishers'}, inplace=True)

    # Áp dụng các biến đổi cho toàn bộ DataFrame mà không lọc bất kỳ cột nào
    filtered_df = df2.pipe(pd.melt, id_vars=['Console', 'Publishers', 'Genre', 'Release Date'], 
                        value_vars=['NA Sales','PAL Sales','JP Sales','Other Sales'], 
                        var_name='Region', value_name='Sales')\
                    .assign(Region=lambda df2: df2.Region.str.strip(' _Sales'))

    filtered_df

    filtered_df['Sales'] = filtered_df['Sales'].fillna(0)
    filtered_df = filtered_df[filtered_df['Sales'] != 0]
    filtered_df = filtered_df[filtered_df['Sales'] != '0.00m']
    filtered_df

    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Define the function to convert sales data
    def convert_to_float(s):
        return float(s.replace('m', ''))

    # Assuming filtered_df is already defined
    filtered_df['Sales'] = filtered_df['Sales'].apply(convert_to_float)


    #third linear model
    model_no_publishers = smf.ols('Sales~Genre+Region+Console',data=filtered_df).fit()
    model_no_publishers.summary()


    path_model_no_publishers = os.path.join(path_model, 'model_no_publishers.joblib')
    # path_model_publishers = os.path.join(path_model, 'model_publishers.joblib')

    dump(model_no_publishers, path_model_no_publishers) 
    

    # # return
    print("Done train models of prediction 3")



# train_model_prediction3()
