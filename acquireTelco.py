#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mason_functions as mf
import warnings
warnings.filterwarnings('ignore')


#This function accepts a string as a parameter, the database name I want to access, and imports valid credentials to access the database and returns and runs the code (credentials involved) necessary to access the database

#necessary to access this database in a python environment
#TL; DR: this function provides access to the database
def get_db_url(db_name):

    #import credentials
    from env import host, user, password
    
    #run necessary code to access RDBMS
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'


    
#This function takes 0 parameters and ouputs all of the information I will need to explore so I can turn the data into actionable intelligence as a dataframe

#TL; DR: this function acquires the necessary data from the database as a dataframe
def telco_data():

    #assign a variable to represent my SQL query
    sql = '''
    SELECT *
    FROM customers
    JOIN contract_types USING(contract_type_id)
    JOIN internet_service_types USING(internet_service_type_id)
    JOIN payment_types USING(payment_type_id)
    '''

    #assign a variable to represent the url necessary to access the telco_churn database
    url = get_db_url('telco_churn')
    
    #use the pd.read_sql function to convert the data into a workable dataframe
    df = pd.read_sql(sql, url)

    #write the dataframe to a readable csv file to easily access throughout exploration
    df.to_csv('telco.csv')

    #return the data as a dataframe
    return df




#this function takes 0 parameters and returns a dataframe
def get_telco_data():

    #set up an if conditional to see if there is a .csv readily available
    if os.path.isfile('telco.csv'):

        #if there is, render a workable dataframe from the .csv
        df = pd.read_csv('telco.csv', index_col = 0)
    else:

        #if not, access the relational database, and then write a .csv for later ease of access
        df = telco_data()
        df.to_csv('telco.csv')

    #return the dataframe
    return df



