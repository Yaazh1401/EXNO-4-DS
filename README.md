# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("C:\\Users\\admin\\Downloads\\bmi.csv")
df
```

<img width="536" height="447" alt="image" src="https://github.com/user-attachments/assets/fdcd0edb-557f-44b9-8c35-39a1f39e6b86" />

```
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,Normalizer,RobustScaler
df2=df.copy()
enc=StandardScaler()
df2[['new_height','new_weight']]=enc.fit_transform(df2[['Height','Weight']])
df2
```


<img width="637" height="445" alt="image" src="https://github.com/user-attachments/assets/5c5e2909-1377-42ec-bd9a-326f4f15282c" />

```
df3=df.copy()
enc=MinMaxScaler()
df3[['new_height','new_weight']]=enc.fit_transform(df3[['Height','Weight']])
df3
```


<img width="788" height="456" alt="image" src="https://github.com/user-attachments/assets/eca2cb7f-dfc8-4761-970e-a76c8cf3ebf8" />

```
df5=df.copy()
enc=MaxAbsScaler()
df5[['new_height','new_weight']]=enc.fit_transform(df5[['Height','Weight']])
df5
```


<img width="581" height="433" alt="image" src="https://github.com/user-attachments/assets/496c2710-c700-403e-9935-b383858d391e" />

```
df6=df.copy()
enc=Normalizer()
df6[['new_height','new_weight']]=enc.fit_transform(df6[['Height','Weight']])
df6
```


<img width="602" height="456" alt="image" src="https://github.com/user-attachments/assets/c4abba23-959a-474f-92e5-27fa3ec8d8d2" />

```
df7=df.copy()
enc=RobustScaler()
df7[['new_height','new_weight']]=enc.fit_transform(df7[['Height','Weight']])
df7
```


<img width="609" height="428" alt="image" src="https://github.com/user-attachments/assets/221c88dd-9ca9-4a6a-b4bd-93abaf6bae2f" />

```
df=pd.read_csv("C:/Users/admin/Downloads/income(1) (1).csv")
df
```


<img width="1283" height="776" alt="image" src="https://github.com/user-attachments/assets/e39bc911-152b-48fc-8b5e-d486acac13bd" />

```
from sklearn.preprocessing import LabelEncoder

df_enc=df.copy()
le=LabelEncoder()

for col in df_enc.select_dtypes(include="object").columns:
    df_enc[col]=le.fit_transform(df_enc[col])
    
x=df_enc.drop("SalStat",axis=1)
y=df_enc["SalStat"]

x
```


<img width="1140" height="444" alt="image" src="https://github.com/user-attachments/assets/5b43693f-5d8c-47f8-b29a-1c05022b492b" />


```
y
```

<img width="493" height="283" alt="image" src="https://github.com/user-attachments/assets/5afe3c81-4530-49b7-a78e-9d748549545e" />

```
from sklearn.feature_selection import SelectKBest,chi2
chi2_selector =SelectKBest(chi2,k=5)
chi2_selector.fit(x,y)

selected_features_chi2=x.columns[chi2_selector.get_support()]
print("selected features(chi-square):",list(selected_features_chi2))

mi_scores=pd.Series(chi2_selector.scores_,index=x.columns)
print(mi_scores.sort_values(ascending=False))
```



<img width="1162" height="323" alt="image" src="https://github.com/user-attachments/assets/13031625-07d5-4bb3-b029-5f6d82b81f63" />

```
from sklearn.feature_selection import f_classif
anova_selector =SelectKBest(f_classif,k=5)
anova_selector.fit(x,y)

selected_features_chi2=x.columns[anova_selector.get_support()]
print("selected features(Anova F-test):",list(selected_features_chi2))

mi_scores=pd.Series(anova_selector.scores_,index=x.columns)
print(mi_scores.sort_values(ascending=False))
```



<img width="1052" height="334" alt="image" src="https://github.com/user-attachments/assets/4f921096-31c0-4640-be69-fb86cc84a7b9" />

```
from sklearn.feature_selection import mutual_info_classif
mi_selector =SelectKBest(mutual_info_classif,k=5)
mi_selector.fit(x,y)

selected_features_mi=x.columns[mi_selector.get_support()]
print("selected features(Mutual info):",list(selected_features_mi))
mi_scores=pd.Series(mi_selector.scores_,index=x.columns)
print(mi_scores.sort_values(ascending=False))
```


<img width="1054" height="318" alt="image" src="https://github.com/user-attachments/assets/b5e0b038-65d7-4d82-9141-970e17551ef1" />

```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model=LogisticRegression(max_iter=100)
rfe=RFE(model,n_features_to_select=5)
rfe.fit(x,y)

sel_feat_rfe=x.columns[rfe.support_]
print("Selected features (RFE):",list(sel_feat_rfe))
```


<img width="886" height="47" alt="image" src="https://github.com/user-attachments/assets/2eb93d50-3613-4e52-a438-91c44a603b18" />

```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

model=LogisticRegression(max_iter=100)
rfe=SequentialFeatureSelector(model,n_features_to_select=5)
rfe.fit(x,y)

sel_feat_rfe=x.columns[rfe.get_support()]
print("Selected features (SF):",list(sel_feat_rfe))
```


<img width="1064" height="93" alt="image" src="https://github.com/user-attachments/assets/078ada33-44cd-4768-8ddd-b6b3addcbef9" />

```
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x, y)

importances = pd.Series(rf.feature_importances_, index=x.columns)

selected_features_rf = importances.sort_values(ascending=False).head(5).index
print(importances)
print("Top 5 features (Random Forest Importance):", list(selected_features_rf))

```


<img width="1168" height="315" alt="image" src="https://github.com/user-attachments/assets/6f9286fb-dd7a-4341-b761-c3457e3196cb" />

```
from sklearn.linear_model import LassoCV 
import numpy as np

lasso = LassoCV(cv=5).fit(x, y)
importance = np.abs(lasso.coef_)

selected_features_lasso = x.columns[importance > 0]

print("Selected features (Lasso):", list(selected_features_lasso))
```


<img width="872" height="67" alt="image" src="https://github.com/user-attachments/assets/b6ab4313-c94e-4e1b-a134-285442b5e5a6" />

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("C:\\Users\\admin\\Downloads\\income(1) (1).csv")

df_encoded = df.copy()

le = LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop("SalStat", axis=1)
y = df_encoded["SalStat"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=3)  # you can tune k
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```



<img width="778" height="357" alt="image" src="https://github.com/user-attachments/assets/cb9be9d4-8231-4168-a56d-316017244dde" />



# RESULT:
      thus,Feature selection and Feature scaling has been used on the given dataset.
