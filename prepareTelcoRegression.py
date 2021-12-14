#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings('ignore')

# import acquire module
import acquireTelco


#------------------------------------- TELCO DATA -------------------------------------#


def prep_telco():
    '''
    This function acquires the telco data through my acquire module, and it drops any seemingly unhelpful
    columns and one-hot encodes my categorical variables for modeling purposes.
    '''
    
    #acquire data from database
    df = acquireTelco.get_telco_data()
    
    #total_charges column holds object-type values. I need workable numeric values
    df.total_charges = df.total_charges.str.strip()
    df = df[df.total_charges != '']
    df.total_charges = df.total_charges.astype(float)
    
    #customer_id is not a useful feature, but it should serve as an index
    df = df.set_index('customer_id')
    
    #now I want to one-hot encode my categorical variables for later exploration
    df = df.replace('No', 0).replace('Yes', 1)
    
    #one-hot encode gender
    dummy_1 = pd.get_dummies(df['gender'], dummy_na = False, drop_first = True)
    
    #rename column
    dummy_1.columns = dummy_1.columns.str.replace('Male', 'is_male')
    
    #more one-hot encoding (get dummy variables for categorical columns with more than 2 unique values)
    #I don't drop the first because I'm picky on categorical columns with more than 2 unique values
    categ_col = df.columns[df.nunique() > 2].to_list()
    dummy_2 = pd.get_dummies(df[categ_col], dummy_na = False, drop_first = False)
    
    #clean up column names
    dummy_2.columns = dummy_2.columns.str.lower().str.replace(' ', '_').str.replace('_1', '').str.replace('_type', '').str.replace('no_internet_service', 'no_internet')
    
    #I do not want these particular first dummy columns
    dummy_drop = dummy_2.columns[dummy_2.columns.str.contains('_0')].to_list()
    dummy_drop_again = dummy_2.columns[dummy_2.columns.str.contains('_no_')].to_list()
    dummy_drop.extend(dummy_drop_again)
    dummy_2 = dummy_2.drop(columns = dummy_drop)
    
    #I want easier column names to work with
    dummy_2 = dummy_2.rename(columns = {
        'contract_month-to-month': 'm2m_contract',
        'contract_one_year': 'one_year_contract',
        'contract_two_year': 'two_year_contract',
        'internet_service_dsl': 'DSL_internet',
        'internet_service_fiber_optic': 'Fiber_internet',
        'internet_service_none': 'no_internet',
        'payment_bank_transfer_(automatic)': 'bank_auto_payment',
        'payment_credit_card_(automatic)': 'card_auto_payment',
        'payment_electronic_check': 'electronic_check_payment',
        'payment_mailed_check': 'mailed_check_payment'
    })
    
    #drop dummied categorical columns 
    df = df.drop(columns = categ_col)
    df = df.drop(columns = 'gender')
    
    #concatenate the dataframes
    telco = pd.concat([dummy_1, df, dummy_2], axis = 1)
    
    #return the resulting dataframe
    return telco


def split_telco(df):
    '''
    Takes in the telco dataset and returns the train, validate, and test subset dataframes.
    Dataframe size for my test set is .2 or 20% of the original data. 
    Validate data is 30% of my training set, which is 24% of the original data. 
    Training data is 56% of the original data.
    '''
    
    #get my training and test data sets defined, stratify my target variable
    train, test = train_test_split(df, test_size = .2, random_state = 421)
    
    #get my validate set from the training set, stratify target variable again
    train, validate = train_test_split(train, test_size = .3, random_state = 421)
    
    #return the 3 dataframes
    return train, validate, test


def prep_telco_data():
    '''
    This function acquires the telco data through my acquire module, preps the data, and returns my train, validate and test
    training sets.
    '''
    #prep the data
    telco = prep_telco()
    
    #split the data
    train, validate, test = split_telco(telco)
    
    #return the end of prep ((t, v, t)) dataframes
    return train, validate, test