

# X = predictors
#y = target
#k = number of features
def select_kbest(X, y, k):
    from sklearn.feature_selection import SelectKBest, f_regression, RFE
    best_feat = SelectKBest(f_regression, k = k)
    best_feat.fit(X_train, y_train)
    feat_mask = best_feat.get_support()
    best_features = X_train.iloc[:,feat_mask].columns.to_list()
    return best_features

select_kbest(X_train, y_train, 2)


def rfe(X, y, k):
    from sklearn.feature_selection import SelectKBest, f_regression, RFE
    from sklearn.linear_model import LinearRegression
    #crank it
    lm = LinearRegression()

    #pop it
    rfe = RFE(lm, k)
    
    #bop it
    rfe.fit(X_train, y_train)  
    
    #twist it
    feat_mask = rfe.support_
    
    #pull it 
    best_rfe = X_train.iloc[:,feat_mask].columns.tolist()
    
    return best_rfe

rfe(X_train, y_train, 2)