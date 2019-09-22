import os
import re
# import numpy as np
from numpy.core.umath_tests import inner1d
import pandas as pd

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import nltk
# nltk.download()
from nltk.corpus import stopwords

datafile = os.path.join('.', 'data', 'labeledTrainData.tsv')
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
df.head()

def display(text, title):
  print(title)
  print("\n----------我是分割线-------------\n")
  print(text) 

raw_example = df['review'][1]
display(raw_example, '原始数据')

example = BeautifulSoup(raw_example, 'html.parser').get_text()
display(example, '去掉HTML标签的数据')

example_letters = re.sub(r'[^a-zA-Z]', ' ', example)
display(example_letters, '去掉标点的数据')

words = example_letters.lower().split()
display(words, '纯词列表数据')

# 下载停用词和其他语料会用到
# nltk.download()

# words_nostop = [w for w in words if w not in stopwords.words('english')]
stopwords = {}.fromkeys([ line.rstrip() for line in open('./stopwords.txt')])
words_nostop = [w for w in words if w not in stopwords]
display(words_nostop, '去掉停用词数据')

# eng_stopwords = set(stopwords.words('english'))
eng_stopwords = set(stopwords)

def clean_text(text):
  text = BeautifulSoup(text, 'html.parser').get_text()
  text = re.sub(r'[^a-zA-Z]', ' ', text)
  words = text.lower().split()
  words = [w for w in words if w not in eng_stopwords]
  return ' '.join(words)

clean_text(raw_example)

df['clean_review'] = df.review.apply(clean_text)
df.head()

vectorizer = CountVectorizer(max_features = 5000) 
train_data_features = vectorizer.fit_transform(df.clean_review).toarray()
train_data_features.shape

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features, df.sentiment)

confusion_matrix(df.sentiment, forest.predict(train_data_features))

del df
del train_data_features

datafile = os.path.join('.', 'data', 'testData.tsv')
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
df['clean_review'] = df.review.apply(clean_text)
df.head()

test_data_features = vectorizer.transform(df.clean_review).toarray()
test_data_features.shape

result = forest.predict(test_data_features)
output = pd.DataFrame({'id':df.id, 'sentiment':result})

output.head()

output.to_csv(os.path.join('.', 'data', 'Bag_of_Words_model.csv'), index=False)

del df
del test_data_features

