# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:40:26 2020

@author: Rajath
"""

import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


Train_data = pd.read_csv('train_set.csv')
Train_data.shape

ps = PorterStemmer()
corpus = []

for i in range(0, Train_data.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', Train_data['Item_Description'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(corpus)
print(x.shape)
vocabulary = vectorizer.vocabulary_

feature_names = vectorizer.get_feature_names()
first_document_vector=x[0]

#Print the TF-IDF values of the first row
df = pd.DataFrame(first_document_vector.T.todense(),index=feature_names,columns=['Tf-Idf'])
df.sort_values(by=['Tf-Idf'],ascending = False)

x = pd.DataFrame.sparse.from_spmatrix(x)
Inv_Df = pd.DataFrame(Train_data['Inv_Amt'])
X = pd.concat([x, Inv_Df] , axis = 1)



class_encodingmap ={
                    'CLASS-1758': 0,  'CLASS-1274': 1,  'CLASS-1522': 2,  'CLASS-1250': 3,  'CLASS-1376': 4,  
                    'CLASS-1963': 5,  'CLASS-1249': 6,  'CLASS-1828': 7,  'CLASS-2141': 8,  'CLASS-1721': 9,  
                    'CLASS-1567': 10, 'CLASS-1919': 11, 'CLASS-2112': 12, 'CLASS-1850': 13, 'CLASS-1477': 14,  
                    'CLASS-2241': 15, 'CLASS-1870': 16, 'CLASS-2003': 17, 'CLASS-1309': 18, 'CLASS-1429': 19, 
                    'CLASS-1322': 20, 'CLASS-1964': 21, 'CLASS-1294': 22, 'CLASS-1770': 23, 'CLASS-1983': 24,  
                    'CLASS-1867': 25, 'CLASS-1652': 26, 'CLASS-2038': 27, 'CLASS-1805': 28, 'CLASS-2152': 29,
                    'CLASS-1688': 30, 'CLASS-1248': 31, 'CLASS-2146': 32, 'CLASS-1957': 33,  'CLASS-1838': 34,  
                    'CLASS-2015': 35 
                    }
Y = Train_data['Product_Category'].map(class_encodingmap)


#Dividng he dataset into the Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
Product_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=Product_detect_model.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
CM = confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
CR = classification_report(y_test,y_pred)

##############################################################################################
#TestDataSet
##############################################################################################
Test_data = pd.read_csv('train_set.csv')

corpus = []

for i in range(0, Test_data.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', Train_data['Item_Description'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)



X_TestData = vectorizer.fit_transform(corpus)

X_TestData = pd.DataFrame.sparse.from_spmatrix(X_TestData)
Inv_Df_TestData = pd.DataFrame(Test_data['Inv_Amt'])
X_TestData = pd.concat([X_TestData, Inv_Df_TestData] , axis = 1)

Y_TestData = Product_detect_model.predict(X_TestData)

Y_TestData = pd.DataFrame(Y_TestData,columns = ['Model_Output'])



















