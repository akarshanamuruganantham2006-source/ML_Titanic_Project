import joblib
import pandas as pd
import seaborn as sns
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.linear_model import LogisticRegression

#Load Dataset
#------------------------------
df=sns.load_dataset("titanic")
df=df[["pclass","sex","parch","age","fare","sibsp","survived"]]
x=df.drop("survived",axis=1)
y=df["survived"]

numeric_features=["age","fare"]
categorical_features=["sex","pclass"]
#----------------------------------
#Numeric Pipeline
#----------------------------------
numeric_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="mean")),
    ("scaler",StandardScaler())
])


#-----------------------------------
#categorical pipeline
#-----------------------------------
categorical_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("onehot",OrdinalEncoder())
])
#------------------------------------
#Column
#------------------------------------
preprocessor=ColumnTransformer([
    ("numeric",numeric_pipeline,numeric_features),
    ("categorical",categorical_pipeline,categorical_features)
])

#---------------------
#final pipeline
#---------------------
pipeline=Pipeline([
    ("preprocessor",preprocessor),
    ("classifier",LogisticRegression())
])
#---------------------
#fit
#---------------------
pipeline.fit(x,y)

#Create the 'model' directory if it doesn't exist
os.makedirs("model",exist_ok=True)

#Save the model
joblib.dump(pipeline,"model/model.pk1")
print("Pipeline saved successfully.")