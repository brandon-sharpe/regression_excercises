import pandas as pd
def wrangle_student_math():
    basepath = './mathfiles/'
    df = pd.read_csv(basepath+'df.csv', index_col=0)
    X_train_exp = pd.read_csv(basepath+'X_train_exp.csv', index_col=0)
    X_train = pd.read_csv(basepath+'X_train.csv', index_col=0)
    y_train = pd.read_csv(basepath+'y_train.csv', index_col=0)
    X_validate = pd.read_csv(basepath+'X_validate.csv', index_col=0)
    y_validate = pd.read_csv(basepath+'y_validate.csv', index_col=0)
    X_test = pd.read_csv(basepath+'X_test.csv', index_col=0)
    y_test = pd.read_csv(basepath+'y_test.csv', index_col=0)
    return df,X_train_exp, X_train, y_train,\
X_validate, y_validate, X_test, y_test