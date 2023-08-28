# Ex.No.1---Data-Preprocessing
## AIM:
To perform Data preprocessing in a data set downloaded from Kaggle.

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
 import pandas as pd
df=pd.read_csv('data.csv')
df

import pandas as pd
df=pd.read_csv("/content/data.csv")
df.head()

df.duplicated()
df.describe()
df.isnull().sum()
x=df.iloc[:, :-1].values
print(x)

y=df.iloc[:, -1].values
print(y)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)

from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
print(xtrain)
print(len(xtrain))
print(xtest)
print(len(xtest))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df1 = sc.fit_transform(df)
print(df1)

x=df.iloc[:, :-1].values
x
y=df.iloc[:, -1].values
y
print(df.isnull().sum)

df.fillna(df.mean().round(1), inplace=True)
print(df.isnull().sum())
y=df.iloc[:, -1].values
print(y)

df.duplicated()
print(df['Calories'].describe())

from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler()
df1=pd.DataFrame(Scaler.fit_transform(df))
df1

from sklearn.model_selection import train_test_split
x_train, x_test ,y_train, y_test = train_test_split(x, y, test_size= 0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```
## OUTPUT:
![image](https://github.com/Archana2003-Jkumar/Ex.No.1---Data-Preprocessing/assets/93427594/15713192-5dd5-4030-87e9-c316afdb9260)
![image](https://github.com/Archana2003-Jkumar/Ex.No.1---Data-Preprocessing/assets/93427594/35213399-4a01-41b1-9a92-8267bc90d3f3)
![image](https://github.com/Archana2003-Jkumar/Ex.No.1---Data-Preprocessing/assets/93427594/175f8bc7-252b-4f6c-9cb0-f3a09f5d45f5)
![image](https://github.com/Archana2003-Jkumar/Ex.No.1---Data-Preprocessing/assets/93427594/7b6294e5-99cf-43a7-8dd6-dc04373e18ee)
![image](https://github.com/Archana2003-Jkumar/Ex.No.1---Data-Preprocessing/assets/93427594/4a52b76b-4394-4a70-92b6-f57e23ceb2a0)
![image](https://github.com/Archana2003-Jkumar/Ex.No.1---Data-Preprocessing/assets/93427594/dd8bad03-d56c-49f9-b2ff-f6a2981c265f)
![image](https://github.com/Archana2003-Jkumar/Ex.No.1---Data-Preprocessing/assets/93427594/e2045dd8-620c-4533-bd8a-ea00779818ef)
![image](https://github.com/Archana2003-Jkumar/Ex.No.1---Data-Preprocessing/assets/93427594/ff2a8e21-05f7-44f4-8d3e-a3777a4bef1e)
![image](https://github.com/Archana2003-Jkumar/Ex.No.1---Data-Preprocessing/assets/93427594/f54a2ef2-26b7-4834-857b-af0ad7a85f5c)
## RESULT
Hence Data preprocessing in a data set downloaded from Kaggle has been performed successfully.
