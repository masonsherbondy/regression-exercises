import warnings
warnings.filterwarnings('ignore')
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import mason_functions as mf

def acquire_zillow():
    #define my sql query into the relational database
    sql = '''
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017
    WHERE propertylandusetypeid = 261 or propertylandusetypeid = 279
    '''

    #define my url
    url = mf.get_db_url('zillow')

    #read the information from the db into a df
    #I also don't want to keep querying the codeup rdbms (in case the kernel gotta go)
    if os.path.isfile('properties_2017.csv'):
        df = pd.read_csv('properties_2017.csv', index_col = 0)
    else:
        df = pd.read_sql(sql, url)
        df.to_csv('properties_2017.csv')

    return df


#remove outliers
def remove_outliers(df, k, col_list):
    ''' 
    Removes outliers from a list of columns in a dataframe and returns the dataframe.
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df



def prep_zillow():
    
    #acquire data
    df = acquire_zillow()

    df = df.rename(columns = {'bedroomcnt': 'bedroom_count',
                         'bathroomcnt': 'bathroom_count',
                         'calculatedfinishedsquarefeet': 'square_footage',
                         'taxvaluedollarcnt': 'tax_value',
                         'yearbuilt': 'year_built',
                         'taxamount': 'taxes',
                         'fips': 'fips_id'
                         })

    #define numeric/ continuous columns
    quant_vars = ['bedroom_count', 'bathroom_count', 'square_footage', 'tax_value', 'taxes']

    #remove outliers
    df = remove_outliers(df, 1.3, quant_vars)

    #I can afford to drop 900 rows in a dataset of over a million records
    df = df[df.year_built.notnull()]

    #engineer feature (this is target leakage)
    df['tax_ratio'] = df.tax_value / df.taxes

    #let's get dummies
    dummy_df = pd.get_dummies(df['fips_id'], dummy_na = False, drop_first = False)

    dummy_df = dummy_df.rename(columns = {6037: 'LA_county',
                                      6059: 'orange_county',
                                      6111: 'ventura_county'
                                     })

    df = pd.concat([df, dummy_df], axis = 1)

    return df


def wrangle_zillow():

    #prep data
    df = prep_zillow()

    #split data
    train, validate, test = mf.split_data(df)

    #return exploration dfs
    return train, validate, test