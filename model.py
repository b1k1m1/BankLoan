# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:54:31 2020

@author: USHA
"""
# First import main libraries\n",
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

# Now import other libraries which we need to complete this project\n",
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from imblearn.over_sampling import SMOTE
from keras import Sequential
from keras.layers import Dense
warnings.filterwarnings('ignore')

# Check whether any NULL values is there in dataset
df=pd.read_csv('bankloan.csv')
df.isnull().sum()

# Remove all the rows which has nukll values
df=df.dropna()
df.isna().any()

# We don't need first columns Loan_ID, we need to drop this column
df=df.drop('Loan_ID', axis =1)

# Now we have to convert LoanAmount to integer and convert into 1000's 
df['LoanAmount'] = (df['LoanAmount']*1000).astype(int)

# Count how many \"N\" and \"Y\" in Loan_status field\n",
Counter(df['Loan_Status'])

# Now figure out the relationship/ratio of Y in Loan Status Field (how many % of \"Y\" are there in Loan_status field\n",
Counter(df["Loan_Status"])['Y']/df["Loan_Status"].size

pre_y = df["Loan_Status"]
pre_X = df.drop("Loan_Status", axis = 1)
dm_X = pd.get_dummies(pre_X)
#dm_X.to_csv('tempfile.csv', index=False)
dm_y = pre_y.map(dict(Y=1, N=0))
pre_y.shape

# Now we will SMOTE the data\n",
smote = SMOTE(sampling_strategy="minority")
X1, y = smote.fit_sample(dm_X, dm_y)
sc = MinMaxScaler()
X = sc.fit_transform(X1)

# Now Split the dataset into train and test 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, shuffle=True)

# Now we will build ANN using Sequential
classifier = Sequential()
classifier.add(Dense(200, activation='relu', kernel_initializer='random_normal', input_dim =X_train.shape[1]))
classifier.add(Dense(400, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

# Now we will compile the model\n",
classifier.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics=['accuracy'])

# Now we will fit the model
classifier.fit(X_train, y_train, batch_size=20, epochs=50, verbose=0)
eval_model=classifier.evaluate(X_train, y_train)
eval_model

# Now we will predict the model
y_pred=classifier.predict(X_test)
y_pred
# Now convert the predictive value to True and False
y_pred=(y_pred>0.5)

# Now create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (10,10))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(["No","Yes"])
ax.yaxis.set_ticklabels(["No", "Yes"])

# Now create the model in pickle file"
import pickle
from sklearn.externals import joblib
filename = 'model_new.pkl'
joblib.dump(classifier, filename)

model = joblib.load('model_new.pkl')
df=pd.read_csv('testfile.csv')
df.info()
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler()
x_test1 = sc.fit_transform(df)
y_pred112 = model.predict(x_test1)
y_pred113= (y_pred112 > 0.5)
#df.info()
df_np = df.to_numpy()
df_np
df_sc = sc.fit_transform(df_np.reshape(17,-1))
df_sc
df_sc1 = df_sc.reshape(-1,17)
df_sc1
y_pred9 = model.predict(df_sc1)
y_pred9
y_pred1=(y_pred9>0.3)
y_pred1
