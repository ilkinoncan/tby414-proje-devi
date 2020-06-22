import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df_train = pd.read_csv("train.csv")
df_train.dtypes
df_train.describe()



df_train[["Pclass", "Sex", "Embarked","Survived"]] = df_train[["Pclass", "Sex", "Embarked","Survived"]].astype("category")
df_train["Title"] = df_train["Name"].map(lambda s:s.split(",")[1].split(".")[0])
df_train["Surname"] = df_train["Name"].map(lambda s:s.split(",")[0])
df_train["Familysize"] = df_train["Parch"]+df_train["SibSp"]+1
bins = [0, 1, 4, 7, np.inf]
names = ["Single", "Small", "Medium", "Large"]
df_train["Fs"] = pd.cut(df_train["Familysize"], bins, labels=names)
df_train["Title"].unique()
df_train = df_train.drop(["Name", "Cabin","SibSp","Parch","Familysize"], axis=1)
df_train["Title"] = df_train["Title"].astype("category")
df_train["Title"] = df_train["Title"].replace(regex =['Jonkheer','Don','Sir','the Countess','Lady'],value ="Royalty")
df_train["Title"] = df_train["Title"].replace(regex =['Capt','Col','Major','Dr','Rev'],value ="Officer")
df_train["Title"] = df_train["Title"].replace(regex =['Mme','Ms'],value ="Mrs")
df_train["Title"] = df_train["Title"].replace(regex =['Mlle'],value ="Miss")



bins_1 = [0, 14, 25, 55, np.inf]
names_1 = ["Children", "Youth" , "Adult", "Senior"]
df_train["Age_c"] = pd.cut(df_train["Age"], bins_1, labels=names_1)



df_train.groupby("Embarked").count()
df_train["Embarked"] = df_train["Embarked"].fillna("S")
df_train_nona = df_train.dropna(subset=["Age"])
df_train_wna = df_train[df_train["Age"].isna()]
x = df_train_nona[["Survived","Pclass","Sex","Fare","Embarked","Title","Fs"]]
y = df_train_nona["Age_c"]
y_n = df_train_nona["Age"]



x_r = pd.get_dummies(data=x, drop_first=True)
y_r = pd.get_dummies(data=y, drop_first=True)



X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(x_r, y_r, test_size=0.2, random_state=0)


clas_arf_r = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
clas_arf_r.fit(X_train_r, y_train_r)
y_pr_arf_r = clas_arf_r.predict(X_test_r)
accuracy_score(y_test_r, y_pr_arf_r)
