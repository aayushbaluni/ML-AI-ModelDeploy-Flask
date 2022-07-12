import pandas as pd
import numpy as np

df=pd.read_csv("hiring.csv")

df=df.fillna(0)

for i in range(0,len(df)):
    if(df['experience'][i] == 0):
        df['experience'][i]='zero'

df.columns=['experience','test_score','interview_score','salary']


y=np.array(df['salary'])


x=df.iloc[:,:3]

from sklearn.linear_model import LinearRegression



lr=LinearRegression()

dic={'zero':0,'two':2,'three':3,'five':5,'seven':7,'ten':10,'eleven':11}


for i in range(0,len(x)):
    x['experience'][i]=dic[x['experience'][i]]


x=np.asarray(x)


lr.fit(x,y)

import pickle


pickle.dump(lr,open('model.pkl','wb'))