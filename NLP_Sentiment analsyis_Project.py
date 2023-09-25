#!/usr/bin/env python
# coding: utf-8

# # **Sentiment Based Product Recommendation System** 
# 
# ---
# 
# 

# ## **Import Packages**

# In[ ]:


# Import the following basic libraries
import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter
import csv
import re 
import string

# To change date to datetime
from datetime import datetime
import time


# Visualization libraries
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


# To show all the columns
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 300)

# Avoid warnings
import warnings
warnings.filterwarnings("ignore")

# Enable logging for gensim - optional but important
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

from IPython.display import clear_output
clear_output()


# In[ ]:


# NLTK libraries
import nltk
nltk.download('all')
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet


# In[ ]:


# Modelling
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics.pairwise import cosine_similarity


# ## **Load the Dataset**

# In[ ]:


df = pd.read_csv('sample30.csv')


# In[ ]:


df.head(2)


# ## **Data Cleaning**

# In[ ]:


# Print the shape of the dataset.
print("Shape :", df.shape)


# In[ ]:


# Print the columns of the dataset.
print("Columns :")
print(df.columns)


# In[ ]:


# Print the information about the dataset columns.
print("Info :")
print(df.info())


# In[ ]:


# Finding the count of missing values in each columns.
print("Missing Value Count :")
print(df.isnull().sum())


# In[ ]:


#Finding the percentage of missing values in each columns.
print("Percentage of missing values :")
print(df.isna().mean().round(4) * 100)


# In[ ]:


# Drop the columns with more than 50% of missing values.
missing_val_threshold = len(df) * .5
df.dropna(thresh = missing_val_threshold, axis = 1, inplace = True)


# In[ ]:


# Drop the 'reviews_doRecommend'& 'reviews_didPurchase' column also as this is of no use for our analysis.
df= df.drop(columns=['reviews_doRecommend'])
df= df.drop(columns=['reviews_didPurchase'])


# In[ ]:


#Finding the percentage of missing values in remaining columns.
print("Percentage of missing values :")
print(df.isna().mean().round(4) * 100)


# In[ ]:


# Drop the NULL value rows of reviews_text, reviews_title, reviews_username, user_sentiment, reviews_date, manufacture. 
df = df[df['reviews_text'].notna()]
df = df[df['reviews_title'].notna()]
df = df[df['reviews_username'].notna()]
df = df[df['user_sentiment'].notna()]
df = df[df['reviews_date'].notna()]
df = df[df['manufacturer'].notna()]


# In[ ]:


#Finding the percentage of missing values in remaining columns.
print("Percentage of missing values :")
print(df.isna().mean().round(4) * 100)


# In[ ]:


df.columns


# In[ ]:


# Renaming the columns
df.rename(columns={'reviews_username' : 'userID','name':'prod_name'}, inplace=True)


# In[ ]:


df.columns


# In[ ]:


# Shape of Dataset: Total of 14 columns are there now.
df.info()


# Converting the Target Variable (user_sentiment) into Binary Numerical Value for Modelling Purposes

# In[ ]:


# Convert the user sentiment column into binary values: Positive to 1 and Negative to 0.

def get_sentiment_binary(x):
    if(x== 'Positive'):
        return 1
    else:
        return 0

#Convert user_sentiment string into binary.
df['user_sentiment']=df['user_sentiment'].apply(get_sentiment_binary)


# In[ ]:


# Shape of Dataset
df.info()


# ## **Text Processing**

# ### **Combine Review Text and Title into one**

# In[ ]:


df[['reviews_title','reviews_text']].head(n=2)


# In[ ]:


# Joining Review Text and Title.
df['Review'] = df['reviews_title'] + " " + df['reviews_text'] 


# In[ ]:


df['Review'].head()


# ### **Lowercasing**

# In[ ]:


# Lowercasing the reviews and title column
df['Review'] = df['Review'].apply(lambda x : x.lower())


# In[ ]:


# Print the first review from row1
df['Review'].head()


# ### **Remove Punctuation**

# In[ ]:


# Remove punctuation 
df['Review'] = df['Review'].str.replace('[^\w\s]','')


# In[ ]:


# Print the first review from row1
df['Review'].head()


# ### **Remove Stopwords**

# In[ ]:


# Remove Stopwords
stop = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[ ]:


# Print the first review from row1
df['Review'].head()


# ### **Lemmatization**

# In[ ]:


lemmatizer = nltk.stem.WordNetLemmatizer()
wordnet_lemmatizer = WordNetLemmatizer() 


# In[ ]:


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# In[ ]:


def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


# In[ ]:


# Apply Lemmetisation
df['Review']=df['Review'].apply(lambda x: lemmatize_sentence(x))


# In[ ]:


# Print the first review from row1
df['Review'][0]


# ### **Noise Removal**

# In[ ]:


def scrub_words(text):
    """Basic cleaning of texts."""
    
    # remove html markup
    text=re.sub("(<.*?>)","",text)
    
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    
    #remove whitespace
    text=text.strip()
    return text


# In[ ]:


df['Review']=df['Review'].apply(lambda x: scrub_words(x))


# In[ ]:


# Print the first review from row1
df['Review'][0]


# ## **Defining Features & Target Variables, Train Test Split, Build the TF-IDF Vectorizer**

# ### **Defining features and target variables**

# In[ ]:


x=df['Review'] 
y=df['user_sentiment']


# In[ ]:


#Distribution of the target variable data.
print(pd.Series(y).value_counts())


# In[ ]:


# #Distribution of the target variable data in terms of proportions.
print("Percent of 1s: ", 100*pd.Series(y).value_counts()[1]/pd.Series(y).value_counts().sum(), "%")
print("Percent of 0s: ", 100*pd.Series(y).value_counts()[0]/pd.Series(y).value_counts().sum(), "%")


# As you can see there are 89% positives and only 11% negative values in the dataset hence, it is an imbalanced dataset. You need to build the model using the original dataset and then you need do use different sampling techniques also to this dataset to make the ML model efficient.

# **Train Test Split**

# In[ ]:


# Split the dataset into test and train
seed = 50 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)


# **TF-IDF Vectorizer**

# In[ ]:


word_vectorizer = TfidfVectorizer(
    strip_accents='unicode',    # Remove accents and perform other character normalization during the preprocessing step. 
    analyzer='word',            # Whether the feature should be made of word or character n-grams.
    token_pattern=r'\w{1,}',    # Regular expression denoting what constitutes a “token”, only used if analyzer == 'word'
    ngram_range=(1, 3),         # The lower and upper boundary of the range of n-values for different n-grams to be extracted
    stop_words='english',
    sublinear_tf=True)

word_vectorizer.fit(X_train)    # Fiting it on Train
train_word_features = word_vectorizer.transform(X_train)  # Transform on Train


# In[ ]:


## transforming the train and test datasets
X_train_transformed = word_vectorizer.transform(X_train.tolist())
X_test_transformed = word_vectorizer.transform(X_test.tolist())

# # Print the shape of each dataset.
print('X_train_transformed', X_train_transformed.shape)
print('y_train', y_train.shape)
print('X_test_transformed', X_test_transformed.shape)
print('y_test', y_test.shape)


# ## **ML Models**

# ### **Logistic Regression**

# #### **Logistic Regression Model- Without any sampling techniques**

# In[ ]:


# Build the Logistic Regression model.
time1 = time.time()

logit = LogisticRegression()
logit.fit(X_train_transformed,y_train)

time_taken = time.time() - time1
print('Time Taken: {:.2f} seconds'.format(time_taken))


# **Model Performance Metrics**

# In[ ]:


# Prediction Train Data
y_pred_train= logit.predict(X_train_transformed)

#Model Performance on Train Dataset
print("Logistic Regression accuracy", accuracy_score(y_pred_train, y_train))
print(classification_report(y_pred_train, y_train))


# In[ ]:


# Prediction Test Data
y_pred_test = logit.predict(X_test_transformed)

#Model Performance on Test Dataset
print("Logistic Regression accuracy", accuracy_score(y_pred_test, y_test))
print(classification_report(y_pred_test, y_test))


# In[ ]:


# Create the confusion matrix for Logistic regression.

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


print("Confusion matrix for train and test set")

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

# confusion matrix for train set
cm_train = metrics.confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm_train/np.sum(cm_train), annot=True , fmt = ' .2%')

plt.subplot(1,2,2)

# confusion matrix for the test data
cm_test = metrics.confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_test/np.sum(cm_test), annot=True , fmt = ' .2%')

plt.show()


# In[ ]:


# storing the values in variables  

#For train set: TN: True Negative, FP: False Positive, FN: False Negative, TP: True Positive
TN_tr = cm_train[0, 0] 
FP_tr = cm_train[0, 1]
FN_tr = cm_train[1, 0]
TP_tr = cm_train[1, 1]

#for test set
TN = cm_test[0, 0]
FP = cm_test[0, 1]
FN = cm_test[1, 0]
TP = cm_test[1, 1]


# Sensitivity (True Positive Rate) is a measure of the proportion of actual positive cases that got predicted as positive.
# 
# Specificity (True Negative Rate) is defined as the proportion of actual negatives, which got predicted as the negative (or true negative).
# 
# 
# True Positive Rate = True Positives / (True Positives + False Negatives)
# 
# False Positive Rate = False Positives / (False Positives + True Negatives)
# 
# The metrics that have been chosen are sensitivity and specificity. As you need to have more sensitivity as well as specificity.

# In[ ]:


#Calculating the Sensitivity for train and test set
sensitivity_tr = TP_tr / float(FN_tr + TP_tr)
print("sensitivity for train set: ",sensitivity_tr)
sensitivity = TP / float(FN + TP)
print("sensitivity for test set: ",sensitivity)


# In[ ]:


#specificity for test and train set. 
specificity_tr = TN_tr / float(TN_tr + FP_tr)
print("specificity for train set: ",specificity_tr)
specificity = TN / float(TN + FP)
print("specificity for test set: ",specificity)


# Since the distribution of the dataset is imbalanced with more positives, many reviews which were negative were incorrectly classified as positives and hence, low specificity. From a business point of view, this is not a very good model as you will miss out on the negatives.

# ####**Logistic Regression Model : With Sampling Techniques**
# 
# We can use different Sampling Techniques like -
# 
# 1. Oversampling
# 2. Smote

# In[ ]:


# Split test and train
seed = 50 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)


# In[ ]:


#from imblearn import over_sampling
from imblearn import over_sampling
ros = over_sampling.RandomOverSampler(random_state=0)


# In[ ]:


# Oversampling the dataset.
X_train, y_train = ros.fit_resample(pd.DataFrame(X_train), pd.Series(y_train))


# In[ ]:


pd.Series(y_train).value_counts()


# In[ ]:


type(X_train)


# In[ ]:


# The word vectorizer takes a list of string as an argument. to get a list of string from a 2D array,
# we convert the 2D array to a dataframe and then convert it to a list.

X_train = pd.DataFrame(X_train).iloc[:,0].tolist()


# In[ ]:


# transforming the train and test datasets

X_train_transformed = word_vectorizer.transform(X_train)
X_test_transformed = word_vectorizer.transform(X_test.tolist())


# In[ ]:


# Building the Logistic Regression model
time1 = time.time()

logit = LogisticRegression()
logit.fit(X_train_transformed,y_train)

time_taken = time.time() - time1
print('Time Taken: {:.2f} seconds'.format(time_taken))


# In[ ]:


# Prediction Train Data
y_pred_train= logit.predict(X_train_transformed)

#Model Performance on Train Dataset
print("Logistic Regression accuracy", accuracy_score(y_pred_train, y_train))
print(classification_report(y_pred_train, y_train))


# In[ ]:


# Prediction Test Data
y_pred_test = logit.predict(X_test_transformed)

print("Logistic Regression accuracy", accuracy_score(y_pred_test, y_test))
print(classification_report(y_pred_test, y_test))
print(confusion_matrix(y_pred_test, y_test))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


print("Confusion matrix for train and test set")

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
# confusion matrix for train set
cm_train = metrics.confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm_train/np.sum(cm_train), annot=True , fmt = ' .2%')
# help(metrics.confusion_matrix)

plt.subplot(1,2,2)
# confusion matrix for the test data
cm_test = metrics.confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_test/np.sum(cm_test), annot=True , fmt = ' .2%')

plt.show()


# In[ ]:


# storing the values in variables  

#For train set
TN_tr = cm_train[0, 0] 
FP_tr = cm_train[0, 1]
FN_tr = cm_train[1, 0]
TP_tr = cm_train[1, 1]

#for test set
TN = cm_test[0, 0]
FP = cm_test[0, 1]
FN = cm_test[1, 0]
TP = cm_test[1, 1]


# In[ ]:


#Calculating the Sensitivity for train and test set
sensitivity_tr = TP_tr / float(FN_tr + TP_tr)
print("sensitivity for train set: ",sensitivity_tr)
sensitivity = TP / float(FN + TP)
print("sensitivity for test set: ",sensitivity)


# In[ ]:


#specificity for test and train set. 
specificity_tr = TN_tr / float(TN_tr + FP_tr)
print("specificity for train set: ",specificity_tr)
specificity = TN / float(TN + FP)
print("specificity for test set: ",specificity)


# As you can see here that the value of specificity has increased hence, you need to use oversampling method.

# ####**Logistic Regression Model: Smote**
# 

# In[ ]:


# Split test and train
seed = 50 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)


# In[ ]:


# Print the type of X_train.
type(X_train)


# In[ ]:


pd.Series(y_train).value_counts()


# In[ ]:


type(X_train)


# In[ ]:


# The word vectorizer takes a list of string as an argument. to get a list of string from a 2D array,
# we convert the 2D array to a dataframe and then convert it to a list.

X_train = pd.DataFrame(X_train).iloc[:,0].tolist()


# In[ ]:


type(X_train)


# In[ ]:


# transforming the train and test datasets

X_train_transformed = word_vectorizer.transform(X_train)
X_test_transformed = word_vectorizer.transform(X_test.tolist())


# In[ ]:


from imblearn.over_sampling import SMOTE

counter = Counter(y_train)
print('Before',counter)

# oversampling the train dataset using SMOTE
smt = SMOTE()
X_train_transformed, y_train = smt.fit_resample(X_train_transformed, y_train)

counter = Counter(y_train)
print('After',counter)


# In[ ]:


# Building the Logistic Regression model
time1 = time.time()

logit = LogisticRegression()
logit.fit(X_train_transformed,y_train)

time_taken = time.time() - time1
print('Time Taken: {:.2f} seconds'.format(time_taken))


# In[ ]:


# Prediction Train Data
y_pred_train= logit.predict(X_train_transformed)

#Model Performance on Train Dataset
print("Logistic Regression accuracy", accuracy_score(y_pred_train, y_train))
print(classification_report(y_pred_train, y_train))


# In[ ]:


# Prediction Test Data
y_pred_test = logit.predict(X_test_transformed)

print("Logistic Regression accuracy", accuracy_score(y_pred_test, y_test))
print(classification_report(y_pred_test, y_test))
print(confusion_matrix(y_pred_test, y_test))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


print("Confusion matrix for train and test set")

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
# confusion matrix for train set
cm_train = metrics.confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm_train/np.sum(cm_train), annot=True , fmt = ' .2%')
# help(metrics.confusion_matrix)

plt.subplot(1,2,2)
# confusion matrix for the test data
cm_test = metrics.confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_test/np.sum(cm_test), annot=True , fmt = ' .2%')

plt.show()


# In[ ]:


# storing the values in variables  

#For train set
TN_tr = cm_train[0, 0] 
FP_tr = cm_train[0, 1]
FN_tr = cm_train[1, 0]
TP_tr = cm_train[1, 1]

#for test set
TN = cm_test[0, 0]
FP = cm_test[0, 1]
FN = cm_test[1, 0]
TP = cm_test[1, 1]


# In[ ]:


#Calculating the Sensitivity for train and test set
sensitivity_tr = TP_tr / float(FN_tr + TP_tr)
print("sensitivity for train set: ",sensitivity_tr)
sensitivity = TP / float(FN + TP)
print("sensitivity for test set: ",sensitivity)


# In[ ]:


#specificity for test and train set. 
specificity_tr = TN_tr / float(TN_tr + FP_tr)
print("specificity for train set: ",specificity_tr)
specificity = TN / float(TN + FP)
print("specificity for test set: ",specificity)


# ![image.png](data:image/png;base64,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)

# ### **Naive Bayes**

# ####**Naive Bayes Model: Oversampling**
# 

# In[ ]:


from imblearn import over_sampling
ros = over_sampling.RandomOverSampler(random_state=0)


# In[ ]:


# Split test and train
seed = 50 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)


# In[ ]:


# Print the type of X_train.
type(X_train)


# In[ ]:


# Oversampling the dataset.
X_train, y_train = ros.fit_resample(pd.DataFrame(X_train), pd.Series(y_train))


# In[ ]:


pd.Series(y_train).value_counts()


# In[ ]:


type(X_train)


# In[ ]:


# The word vectorizer takes a list of string as an argument. to get a list of string from a 2D array,
# we convert the 2D array to a dataframe and then convert it to a list.

X_train = pd.DataFrame(X_train).iloc[:,0].tolist()


# In[ ]:


type(X_train)


# In[ ]:


# transforming the train and test datasets

X_train_transformed = word_vectorizer.transform(X_train)
X_test_transformed = word_vectorizer.transform(X_test.tolist())


# In[ ]:


# Build the Naive Bayes model.

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

time1 = time.time()

mnb = MultinomialNB()
mnb.fit(X_train_transformed,y_train)


time_taken = time.time() - time1
print('Time Taken: {:.2f} seconds'.format(time_taken))


# In[ ]:


# Prediction Train Data
y_pred_train = mnb.predict(X_train_transformed)

print("Naive Bayes accuracy", accuracy_score(y_pred_train , y_train))
print(classification_report(y_pred_train , y_train))


# In[ ]:


# Prediction Test Data
y_pred_test  = mnb.predict(X_test_transformed)

print("Naive Bayes accuracy", accuracy_score(y_pred_test , y_test))
print(classification_report(y_pred_test , y_test))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


print("Confusion matrix for train and test set")

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
# confusion matrix for train set
cm_train = metrics.confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm_train/np.sum(cm_train), annot=True , fmt = ' .2%')
# help(metrics.confusion_matrix)

plt.subplot(1,2,2)
# confusion matrix for the test data
cm_test = metrics.confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_test/np.sum(cm_test), annot=True , fmt = ' .2%')

plt.show()


# In[ ]:


# storing the values in variables  

#For train set
TN_tr = cm_train[0, 0] 
FP_tr = cm_train[0, 1]
FN_tr = cm_train[1, 0]
TP_tr = cm_train[1, 1]

#for test set
TN = cm_test[0, 0]
FP = cm_test[0, 1]
FN = cm_test[1, 0]
TP = cm_test[1, 1]


# In[ ]:


#Calculating the Sensitivity for train and test set
sensitivity_tr = TP_tr / float(FN_tr + TP_tr)
print("sensitivity for train set: ",sensitivity_tr)
sensitivity = TP / float(FN + TP)
print("sensitivity for test set: ",sensitivity)


# In[ ]:


#specificity for test and train set. 
specificity_tr = TN_tr / float(TN_tr + FP_tr)
print("specificity for train set: ",specificity_tr)
specificity = TN / float(TN + FP)
print("specificity for test set: ",specificity)


# ### **Random Forest Classifier**

# #### **Random Forest Model: Oversampling** 
# 

# In[ ]:


#from imblearn import over_sampling
from imblearn import over_sampling
ros = over_sampling.RandomOverSampler(random_state=0)


# In[ ]:


# Split test and train
seed = 50 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)


# In[ ]:


# Print the type of X_train.
type(X_train)


# In[ ]:


# Oversampling the dataset.
X_train, y_train = ros.fit_resample(pd.DataFrame(X_train), pd.Series(y_train))


# In[ ]:


pd.Series(y_train).value_counts()


# In[ ]:


type(X_train)


# In[ ]:


# The word vectorizer takes a list of string as an argument. to get a list of string from a 2D array,
# we convert the 2D array to a dataframe and then convert it to a list.

X_train = pd.DataFrame(X_train).iloc[:,0].tolist()


# In[ ]:


type(X_train)


# In[ ]:


# transforming the train and test datasets

X_train_transformed = word_vectorizer.transform(X_train)
X_test_transformed = word_vectorizer.transform(X_test.tolist())


# In[ ]:


# Building Random Forest Model.
time1 = time.time()

classifier = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
classifier.fit(X_train_transformed,y_train)

time_taken = time.time() - time1
print('Time Taken: {:.2f} seconds'.format(time_taken))


# In[ ]:


# Prediction Train Data
y_pred_train= classifier.predict(X_train_transformed)

print("Random Forest Model accuracy", accuracy_score(y_pred_train, y_train))
print(classification_report(y_pred_train, y_train))


# In[ ]:


# Prediction Test Data
y_pred_test = classifier.predict(X_test_transformed)

print("Random Forest Model accuracy", accuracy_score(y_pred_test, y_test))
print(classification_report(y_pred_test, y_test))


# In[ ]:


# Create the confusion matrix for Random Forest.

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


print("Confusion matrix for train and test set")

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

# confusion matrix for train set
cm_train = metrics.confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm_train/np.sum(cm_train), annot=True , fmt = ' .2%')
# help(metrics.confusion_matrix)

plt.subplot(1,2,2)

# confusion matrix for the test data
cm_test = metrics.confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_test/np.sum(cm_test), annot=True , fmt = ' .2%')

plt.show()


# In[ ]:


# storing the values in variables  

#For train set: TN: True Negative, FP: False Positive, FN: False Negative, TP: True Positive
TN_tr = cm_train[0, 0] 
FP_tr = cm_train[0, 1]
FN_tr = cm_train[1, 0]
TP_tr = cm_train[1, 1]

#for test set
TN = cm_test[0, 0]
FP = cm_test[0, 1]
FN = cm_test[1, 0]
TP = cm_test[1, 1]


# In[ ]:


#Calculating the Sensitivity for train and test set
sensitivity_tr = TP_tr / float(FN_tr + TP_tr)
print("sensitivity for train set: ",sensitivity_tr)
sensitivity = TP / float(FN + TP)
print("sensitivity for test set: ",sensitivity)


# In[ ]:


#specificity for test and train set. 
specificity_tr = TN_tr / float(TN_tr + FP_tr)
print("specificity for train set: ",specificity_tr)
specificity = TN / float(TN + FP)
print("specificity for test set: ",specificity)


# ### **XGBoost**

# ####**XGBoost Model: Oversampling**
# 

# In[ ]:


#from imblearn import over_sampling
from imblearn import over_sampling
ros = over_sampling.RandomOverSampler(random_state=0)


# In[ ]:


# Split test and train
seed = 50 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)


# In[ ]:


# Print the type of X_train.
type(X_train)


# In[ ]:


# Oversampling the dataset.
X_train, y_train = ros.fit_resample(pd.DataFrame(X_train), pd.Series(y_train))


# In[ ]:


pd.Series(y_train).value_counts()


# In[ ]:


type(X_train)


# In[ ]:


# The word vectorizer takes a list of string as an argument. to get a list of string from a 2D array,
# we convert the 2D array to a dataframe and then convert it to a list.

X_train = pd.DataFrame(X_train).iloc[:,0].tolist()


# In[ ]:


type(X_train)


# In[ ]:


# transforming the train and test datasets

X_train_transformed = word_vectorizer.transform(X_train)
X_test_transformed = word_vectorizer.transform(X_test.tolist())


# In[ ]:


# Building the XGBoost model
import xgboost as xgb
time1 = time.time()

xgb = xgb.XGBClassifier(n_jobs=-1)
xgb.fit(X_train_transformed,y_train)

time_taken = time.time() - time1
print('Time Taken: {:.2f} seconds'.format(time_taken))


# In[ ]:


# Prediction Train Data
y_pred_train  = xgb.predict(X_train_transformed)

print("XGBoost Model accuracy", accuracy_score(y_pred_train, y_train))
print(classification_report(y_pred_train, y_train))


# In[ ]:


# Prediction Test Data
y_pred_test  = xgb.predict(X_test_transformed)

print("XGBoost Model accuracy", accuracy_score(y_pred_test, y_test))
print(classification_report(y_pred_test, y_test))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


print("Confusion matrix for train and test set")

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
# confusion matrix for train set
cm_train = metrics.confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm_train/np.sum(cm_train), annot=True , fmt = ' .2%')
# help(metrics.confusion_matrix)

plt.subplot(1,2,2)
# confusion matrix for the test data
cm_test = metrics.confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_test/np.sum(cm_test), annot=True , fmt = ' .2%')

plt.show()


# In[ ]:


# storing the values in variables  

#For train set
TN_tr = cm_train[0, 0] 
FP_tr = cm_train[0, 1]
FN_tr = cm_train[1, 0]
TP_tr = cm_train[1, 1]

#for test set
TN = cm_test[0, 0]
FP = cm_test[0, 1]
FN = cm_test[1, 0]
TP = cm_test[1, 1]


# In[ ]:


#Calculating the Sensitivity for train and test set
sensitivity_tr = TP_tr / float(FN_tr + TP_tr)
print("sensitivity for train set: ",sensitivity_tr)
sensitivity = TP / float(FN + TP)
print("sensitivity for test set: ",sensitivity)


# In[ ]:


#specificity for test and train set. 
specificity_tr = TN_tr / float(TN_tr + FP_tr)
print("specificity for train set: ",specificity_tr)
specificity = TN / float(TN + FP)
print("specificity for test set: ",specificity)


# ### **Hyper Parameter Tuning On Oversampled Data** 
# 
# 
# 

# #### Random Forest

# In[ ]:


from imblearn import over_sampling
ros = over_sampling.RandomOverSampler(random_state=0)


# In[ ]:


# Split test and train
seed = 50 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)


# In[ ]:


# Print the type of X_train.
type(X_train)


# In[ ]:


# Oversampling the dataset.
X_train, y_train = ros.fit_resample(pd.DataFrame(X_train), pd.Series(y_train))


# In[ ]:


pd.Series(y_train).value_counts()


# In[ ]:


type(X_train)


# In[ ]:


# The word vectorizer takes a list of string as an argument. to get a list of string from a 2D array,
# we convert the 2D array to a dataframe and then convert it to a list.

X_train = pd.DataFrame(X_train).iloc[:,0].tolist()


# In[ ]:


type(X_train)


# In[ ]:


# transforming the train and test datasets

X_train_transformed = word_vectorizer.transform(X_train)
X_test_transformed = word_vectorizer.transform(X_test.tolist())


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
X_train_transformed,y_train
# Building Random Forest Model.
time1 = time.time()

n_estimators = [10,20,30] 
max_features = ['auto', 'sqrt']
max_depth = [4,5,6]
max_depth.append(None) # If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf_classifier = RandomForestClassifier(random_state=42)

rf_final = RandomizedSearchCV(estimator=rf_classifier, param_distributions=random_grid, n_iter=5, cv=3, 
                               verbose=2, random_state=42, n_jobs=-1)


rf_final.fit(X_train_transformed,y_train)

time_taken = time.time() - time1
print('Time Taken: {:.2f} seconds'.format(time_taken))


# In[ ]:


rf_final.best_estimator_


# In[ ]:


# Prediction Train Data
y_pred_train= rf_final.predict(X_train_transformed)

print("Random Forest Model accuracy", accuracy_score(y_pred_train, y_train))
print(classification_report(y_pred_train, y_train))


# In[ ]:


# Prediction Test Data
y_pred_test = rf_final.predict(X_test_transformed)

print("Random Forest Model accuracy", accuracy_score(y_pred_test, y_test))
print(classification_report(y_pred_test, y_test))


# In[ ]:


# Create the confusion matrix for Random Forest.

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


print("Confusion matrix for train and test set")

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

# confusion matrix for train set
cm_train = metrics.confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm_train/np.sum(cm_train), annot=True , fmt = ' .2%')
# help(metrics.confusion_matrix)

plt.subplot(1,2,2)

# confusion matrix for the test data
cm_test = metrics.confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_test/np.sum(cm_test), annot=True , fmt = ' .2%')

plt.show()


# In[ ]:


# storing the values in variables  

#For train set: TN: True Negative, FP: False Positive, FN: False Negative, TP: True Positive
TN_tr = cm_train[0, 0] 
FP_tr = cm_train[0, 1]
FN_tr = cm_train[1, 0]
TP_tr = cm_train[1, 1]

#for test set
TN = cm_test[0, 0]
FP = cm_test[0, 1]
FN = cm_test[1, 0]
TP = cm_test[1, 1]


# In[ ]:


#Calculating the Sensitivity for train and test set
sensitivity_tr = TP_tr / float(FN_tr + TP_tr)
print("sensitivity for train set: ",sensitivity_tr)
sensitivity = TP / float(FN + TP)
print("sensitivity for test set: ",sensitivity)


# In[ ]:


#specificity for test and train set. 
specificity_tr = TN_tr / float(TN_tr + FP_tr)
print("specificity for train set: ",specificity_tr)
specificity = TN / float(TN + FP)
print("specificity for test set: ",specificity)


# #### Xgboost

# In[ ]:


#from imblearn import over_sampling
from imblearn import over_sampling
ros = over_sampling.RandomOverSampler(random_state=0)


# In[ ]:


# Split test and train
seed = 50 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)


# In[ ]:


# Print the type of X_train.
type(X_train)


# In[ ]:


# Oversampling the dataset.
X_train, y_train = ros.fit_resample(pd.DataFrame(X_train), pd.Series(y_train))


# In[ ]:


pd.Series(y_train).value_counts()


# In[ ]:


type(X_train)


# In[ ]:


# The word vectorizer takes a list of string as an argument. to get a list of string from a 2D array,
# we convert the 2D array to a dataframe and then convert it to a list.

X_train = pd.DataFrame(X_train).iloc[:,0].tolist()


# In[ ]:


type(X_train)


# In[ ]:


# transforming the train and test datasets

X_train_transformed = word_vectorizer.transform(X_train)
X_test_transformed = word_vectorizer.transform(X_test.tolist())


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
# Building Xgboost Model.
time1 = time.time()

import xgboost as xgb 
n_estimators = [10,15,20,25,30] 
max_features = ['auto', 'sqrt']
max_depth = [4,5,6]
max_depth.append(None) # If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

xgb = xgb.XGBClassifier(n_jobs=-1)

xgb_final = RandomizedSearchCV(estimator=xgb, param_distributions=random_grid, n_iter=5, cv=3, 
                               verbose=2, random_state=42, n_jobs=-1)


xgb_final.fit(X_train_transformed,y_train)

time_taken = time.time() - time1
print('Time Taken: {:.2f} seconds'.format(time_taken))


# In[ ]:


xgb_final.best_estimator_


# In[ ]:


# Prediction Train Data
y_pred_train= xgb_final.predict(X_train_transformed)

print("Xgboost Forest Model accuracy", accuracy_score(y_pred_train, y_train))
print(classification_report(y_pred_train, y_train))


# In[ ]:


# Prediction Test Data
y_pred_test = xgb_final.predict(X_test_transformed)

print("Xgboost Model accuracy", accuracy_score(y_pred_test, y_test))
print(classification_report(y_pred_test, y_test))


# In[ ]:


# Create the confusion matrix for Random Forest.

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


print("Confusion matrix for train and test set")

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

# confusion matrix for train set
cm_train = metrics.confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm_train/np.sum(cm_train), annot=True , fmt = ' .2%')
# help(metrics.confusion_matrix)

plt.subplot(1,2,2)

# confusion matrix for the test data
cm_test = metrics.confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_test/np.sum(cm_test), annot=True , fmt = ' .2%')

plt.show()


# In[ ]:


# storing the values in variables  

#For train set: TN: True Negative, FP: False Positive, FN: False Negative, TP: True Positive
TN_tr = cm_train[0, 0] 
FP_tr = cm_train[0, 1]
FN_tr = cm_train[1, 0]
TP_tr = cm_train[1, 1]

#for test set
TN = cm_test[0, 0]
FP = cm_test[0, 1]
FN = cm_test[1, 0]
TP = cm_test[1, 1]


# In[ ]:


#Calculating the Sensitivity for train and test set
sensitivity_tr = TP_tr / float(FN_tr + TP_tr)
print("sensitivity for train set: ",sensitivity_tr)
sensitivity = TP / float(FN + TP)
print("sensitivity for test set: ",sensitivity)


# In[ ]:


#specificity for test and train set. 
specificity_tr = TN_tr / float(TN_tr + FP_tr)
print("specificity for train set: ",specificity_tr)
specificity = TN / float(TN + FP)
print("specificity for test set: ",specificity)


# #### Machine Learning - Sentiment Model Evaluation Summary

# ![image.png](data:image/png;base64,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)

# ## **Recommendation System**

# ### **Test and Train Split**

# In[ ]:


df.info()


# In[ ]:


# Getting only relevant columns
df1 = df[["userID","reviews_rating","prod_name"]]


# In[ ]:


#Check the NULL count:
df1.isnull().any()


# In[ ]:


df1.info()


# In[ ]:


df1.head(2)


# #### Dividing the dataset into train and test

# In[ ]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df1, test_size=0.30, random_state=31)


# In[ ]:


print(train.shape)
print(test.shape)


# #### Dummy Train creation for end prediciton.

# These dataset will be used for prediction and evaluation.
# 
# Dummy train will be used later for prediction of the movies which has not been rated by the user. To ignore the movies rated by the user, we will mark it as 0 during prediction. The movies not rated by user is marked as 1 for prediction.

# In[ ]:


dummy_train = train.copy()


# In[ ]:


# Remove this part
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)


# In[ ]:


dummy_train.head()


# In[ ]:





# In[ ]:


# The products not rated by user is marked as 1 for prediction. And make the user- item matrix representaion.
dummy_train = dummy_train.pivot_table(index='userID', columns='prod_name', values='reviews_rating').fillna(1)


# ### **User Similarity Matrix**

# #### **Using adjusted Cosine** 

# We are not removing the NaN values and calculating the mean only for the movies rated by the user

# In[ ]:


# Make the user- item matrix representaion of train dataset.
user_based_matrix = train.pivot_table(index='userID', columns='prod_name', values='reviews_rating')


# In[ ]:


user_based_matrix.shape


# Normalising the rating of the products for each user around 0 mean

# In[ ]:


mean = np.nanmean(user_based_matrix, axis=1)
df_subtracted = (user_based_matrix.T-mean).T


# In[ ]:


pd.options.display.max_columns = None
df_subtracted.sample()


# Finding cosine similarity

# In[ ]:


from sklearn.metrics.pairwise import pairwise_distances

# User Similarity Matrix: The correlation matrix of users.
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# In[ ]:


np.shape(user_correlation)


# #### **Prediction**

# Doing the prediction for the users which are positively related with other users, and not the users which are negatively related as we are interested in the users which are more similar to the current users. So, ignoring the correlation for values less than 0. 

# In[ ]:


user_correlation[user_correlation<0]=0
user_correlation


# In[ ]:


user_correlation.shape


# Rating predicted by the user (for movies rated as well as not rated) is the weighted sum of correlation with the movie rating (as present in the rating dataset). 

# In[ ]:


user_predicted_ratings = np.dot(user_correlation, user_based_matrix.fillna(0))
user_predicted_ratings


# In[ ]:


user_predicted_ratings.shape


# #### Since we are interested only in the products not rated by the user, we will ignore the products rated by the user by making it zero. 

# In[ ]:


user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()


# ### **Item Based Similarity**

# #### **Using adjusted Cosine**

# Using Correlation
# 
# Taking the transpose of the rating matrix to normalize the rating around the mean for different movie ID. In the user based similarity, we had taken mean for each user intead of each movie.

# In[ ]:


item_based_matrix = train.pivot_table(index='userID', columns='prod_name', values='reviews_rating').T


# In[ ]:


item_based_matrix.head(2)


# In[ ]:


item_based_matrix.shape


# Normalising the movie rating for each movie

# In[ ]:


mean = np.nanmean(item_based_matrix, axis=1)
df_subtracted = (item_based_matrix.T-mean).T


# In[ ]:


df_subtracted.head()


# Finding the cosine similarity.

# In[ ]:


from sklearn.metrics.pairwise import pairwise_distances

item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)


# Filtering the correlation only for which the value is greater than 0. (Positively correlated)
# 

# In[ ]:


item_correlation[item_correlation<0]=0
item_correlation


# #### **Prediction**

# In[ ]:


item_predicted_ratings = np.dot((item_based_matrix.fillna(0).T),item_correlation)
item_predicted_ratings


# In[ ]:


item_predicted_ratings.shape


# In[ ]:


dummy_train.shape


# #### Filtering the rating only for the products not rated by the user for recommendation

# In[ ]:


item_final_rating = np.multiply(item_predicted_ratings,dummy_train)
item_final_rating.head()


# ### **Evaluation: User vs Item Based System**

# Evaluation will we same as you have seen above for the prediction. The only difference being, you will evaluate for the movie already rated by the user insead of predicting it for the movie not rated by the user.

# #### **Using User Similarity**

# In[ ]:


common = test[test.userID.isin(train.userID)]
common.shape


# In[ ]:


common.head()


# In[ ]:


common_user_based_matrix = common.pivot_table(index='userID', columns='prod_name', values='reviews_rating')


# In[ ]:


common_user_based_matrix.shape


# In[ ]:


user_correlation_df = pd.DataFrame(user_correlation)


# In[ ]:


user_correlation_df['reviews_username'] = df_subtracted.index
user_correlation_df.set_index('reviews_username',inplace=True)
user_correlation_df.head()


# In[ ]:


common.head(1)


# In[ ]:


list_name = common.userID.tolist()

user_correlation_df.columns = df_subtracted.index.tolist()
user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)]


# In[ ]:


user_correlation_df_1.shape


# In[ ]:


user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]


# In[ ]:


user_correlation_df_3 = user_correlation_df_2.T


# In[ ]:


user_correlation_df_3.head()


# The products not rated by user is marked as 0 for evaluation. And make the user- item matrix representaion.

# In[ ]:


user_correlation_df_3[user_correlation_df_3<0]=0

common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))
common_user_predicted_ratings


# 
# Dummy test will be used for evaluation. To evaluate, we will only make prediction on the movies rated by the user. So, this is marked as 1. This is just opposite of dummy_train

# In[ ]:


dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='userID', columns='prod_name', values='reviews_rating').fillna(0)


# Doing prediction for the movies rated by the user

# In[ ]:


dummy_test.shape


# In[ ]:


common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test)


# In[ ]:


common_user_predicted_ratings.head()


# Calculating the RMSE for only the movies rated by user. For RMSE, normalising the rating to (1,5) range.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_user_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[ ]:


common_ = common.pivot_table(index='userID', columns='prod_name', values='reviews_rating')


# In[ ]:


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))


# In[ ]:


rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# #### **Using Item Similarity**

# In[ ]:


common =  test[test.prod_name.isin(train.prod_name)]

common.prod_name.nunique()
common.shape


# In[ ]:


common.head(1)


# In[ ]:


common_item_based_matrix = common.pivot_table(index='userID', columns='prod_name', values='reviews_rating').T


# In[ ]:


common_item_based_matrix.shape


# In[ ]:


item_correlation_df = pd.DataFrame(item_correlation)


# In[ ]:


item_correlation_df.head(1)


# In[ ]:


item_correlation_df['reviews_items'] = df_subtracted.index
item_correlation_df.set_index('reviews_items',inplace=True)
item_correlation_df.head()


# In[ ]:


common_item_based_matrix.shape


# In[ ]:


list_name = common.prod_name.tolist()


# In[ ]:


item_correlation_df.columns = df_subtracted.index.tolist()
item_correlation_df_1 =  item_correlation_df[item_correlation_df.index.isin(list_name)]


# In[ ]:


item_correlation_df_2 = item_correlation_df_1.T[item_correlation_df_1.T.index.isin(list_name)]
item_correlation_df_3 = item_correlation_df_2.T


# In[ ]:


item_correlation_df_3[item_correlation_df_3<0]=0
common_item_predicted_ratings = np.dot(item_correlation_df_3, common_item_based_matrix.fillna(0))
common_item_predicted_ratings


# In[ ]:


common_item_predicted_ratings.shape


# Dummy test will be used for evaluation. To evaluate, we will only make prediction on the movies rated by the user. So, this is marked as 1. This is just opposite of dummy_train
# 
# 

# In[ ]:


dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='userID', columns='prod_name', values='reviews_rating').T.fillna(0)

common_item_predicted_ratings = np.multiply(common_item_predicted_ratings,dummy_test)


# The products not rated is marked as 0 for evaluation. And make the item- item matrix representaion.
# 

# In[ ]:


common_ = common.pivot_table(index='userID', columns='prod_name', values='reviews_rating').T


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_item_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[ ]:


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))


# In[ ]:


rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# ### **Recommendation**

# **Generating top 20 recommendation for particular user**
# 

# In[ ]:


# save the respective files Pickle 
import pickle
pickle.dump(user_final_rating,open('user_final_rating.pkl','wb'))
user_final_rating =  pickle.load(open('user_final_rating.pkl', 'rb'))


# In[ ]:


# Using User based similarity system as its RMSE value is less than item based similarity system.
user_input = input("Enter your user name")
print(user_input)


# In[ ]:


# Recommending the Top 20 products to the user.
d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
d


# **Filtering out the Top 5 recommendation items based on Logistic Regression ML model.**

# In[ ]:


# save the respective files and models through Pickle 
import pickle
pickle.dump(logit,open('logit_model.pkl', 'wb'))
# loading pickle object
logit =  pickle.load(open('logit_model.pkl', 'rb'))

pickle.dump(word_vectorizer,open('word_vectorizer.pkl','wb'))
# loading pickle object
word_vectorizer = pickle.load(open('word_vectorizer.pkl','rb'))


# In[ ]:


# Define a function to recommend top 5 filtered products to the user.
def recommend(user_input):
    d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]

    # Based on positive sentiment percentage.
    i= 0
    a = {}
    for prod_name in d.index.tolist():
      product_name = prod_name
      product_name_review_list =df[df['prod_name']== product_name]['Review'].tolist()
      features= word_vectorizer.transform(product_name_review_list)
      logit.predict(features)
      a[product_name] = logit.predict(features).mean()*100
    b= pd.Series(a).sort_values(ascending = False).head(5).index.tolist()
    print(b)


# In[ ]:


recommend(user_input)


# In[ ]:


df.to_csv("df.csv",index=False)

