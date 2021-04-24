from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def get_feature_removal():
    def removeFeatures(X):
        return X.drop(["Cabin", "Name", "Ticket"], axis=1)

    return FunctionTransformer(removeFeatures)

def get_col_transf():
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('poly', 'passthrough')
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder()), 
    ])
        
    col_t = ColumnTransformer([
        ('num', num_pipe, make_column_selector(dtype_include=['int64', 'float64'])),
        ('cat', cat_pipe, make_column_selector(dtype_include='object'))
    ])

    return col_t