import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
from env import user, host, password
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

#**************************************************Acquire*******************************************************


def get_connection(db, username=user, host=host, password=password):
    '''
    Creates a connection URL
    '''
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'
    

    
    
def new_zillow_data():
    '''
    Returns zillow data as a dataframe
    '''
    sql_query = '''SELECT
                    bedroomcnt,
                    bathroomcnt, 
                    calculatedfinishedsquarefeet,
                    taxvaluedollarcnt,
                    yearbuilt,
                    taxamount,
                    fips
                    FROM properties_2017
                    join propertylandusetype using(propertylandusetypeid)
                    where propertylandusetypeid = 261'''
    df = pd.read_sql(sql_query, get_connection('zillow'))
    return df 




def get_zillow_data():
    '''get connection, returns Zillow into a dataframe and creates a csv for us'''
    if os.path.isfile('zillow.csv'):
        df = pd.read_csv('zillow.csv', index_col=0)
    else:
        df = new_zillow_data()
        df.to_csv('zillow.csv')
    return df



#**************************************************Remove Outliers*******************************************************

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#**************************************************Distributions*******************************************************

def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if col not in ['fips', 'yearbuilt']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()
        
        
def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = ['bedrooms', 'bathrooms', 'sqft', 'taxvalue']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()

   #**************************************************Scale*******************************************************

def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    
    # new column names
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    
    # Fit the scaler on the train
    scaler.fit(train[columns_to_scale])
    
    # transform train validate and test
    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    
    
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    return train, validate, test

 
    
#**************************************************Prepare*******************************************************

def prepare_zillow():
    '''acquires and prepares the zillow data for us. Drops all null values (.005% of the data) then drops all 
    outliers'''
    
    # Get zillow file
    df = get_zillow_data()
    
    # Drop the null in every column(less than .006% of the data)
    df = df.dropna()
    
    # drop tax taxamount due to not actually having this knowledge until the following year. Data leak
    df = df.drop(columns=['taxamount'])
    
    # Remove the outliers
    k=1.5
    col_list=df.columns
    df = remove_outliers(df, k, col_list)
    
    # reset index
    df.reset_index(inplace=True,drop=True)
    
    # rename my columns
    df = df.rename(columns={"bedroomcnt": "bedrooms", "bathroomcnt": "bathrooms", "calculatedfinishedsquarefeet": "sqft", "taxvaluedollarcnt": "taxvalue"})
    
    # converts float whole numbers to int
    slc =['bedrooms','sqft','taxvalue','yearbuilt','fips']
    df[slc] = df[slc].astype(int)
    
    # converting column data thats categorical into objects so that tgey wont be used in mathmetics
    df.fips = df.fips.astype(object)
    df.year_built = df.yearbuilt.astype(object)
    
    # get distributions of numeric data
    get_hist(df)
    get_box(df)
    
  
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=617)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=617)
    return train, validate, test 



#**************************************************Wrangle*******************************************************


def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow()
    train, validate, test = add_scaled_columns(train,
                                                   validate,
                                                   test,
                                                   scaler=sklearn.preprocessing.MinMaxScaler(),
                                                   columns_to_scale=['bedrooms', 'bathrooms', 'sqft'])
    
    return train, validate, test

















#******************************Telco wrangle********************************************

def new_telco_charge_data():
    '''
    Returns telco_charge into a dataframe
    '''
    sql_query = '''select customer_id, monthly_charges, tenure, total_charges from customers
    join internet_service_types using(internet_service_type_id)
    join contract_types using(contract_type_id)
    join payment_types using(payment_type_id)
    where contract_type_id = 3'''
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    return df 

def get_telco_charge_data():
    '''get connection, returns telco_charge into a dataframe and creates a csv for us'''
    if os.path.isfile('telco_charge.csv'):
        df = pd.read_csv('telco_charge.csv', index_col=0)
    else:
        df = new_telco_charge_data()
        df.to_csv('telco_charge.csv')
    return df

def clean_telco(df):
    '''cleans our data'''
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    df.total_charges = pd.to_numeric(df.total_charges, errors='coerce').astype('float64')
    df.total_charges = df.total_charges.fillna(value=df.total_charges.mean()).astype('float64')
    df = df.set_index("customer_id")
    return df 


def split_telco(df):
    '''
    Takes in a cleaned df of telco data and splits the data appropriatly into train, validate, and test.
    '''
    
    train_val, test = train_test_split(df, train_size =  0.8, random_state = 123)
    train, validate = train_test_split(train_val, train_size =  0.7, random_state = 123)
    return train, validate, test

def wrangle_telco():
    '''acquire and our dataframe, returns a df'''
    df = clean_telco(get_telco_charge_data())
    train, validate, test = split_telco(df)
    return train, validate, test
