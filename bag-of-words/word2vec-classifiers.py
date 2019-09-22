import os
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from gensim.models.word2vec import Word2Vec

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

def load_dataset(name, nrows=None):
  datasets = {
    'unlabeled_train': 'unlabeledTrainData.tsv',
    'labeled_train': 'labeledTrainData.tsv',
    'test': 'testData.tsv'
  }
  if name not in datasets:
    raise ValueError(name)
  data_file = os.path.join('.', 'data', datasets[name])
  df = pd.read_csv(data_file, sep='\t', escapechar='\\', nrows=nrows)
  print('Number of reviews: {}'.format(len(df)))
  return df

eng_stopwords = set(stopwords.words('english'))

def clean_text(text, remove_stopwords=False):
  text = BeautifulSoup(text, 'html.parser').get_text()
  text = re.sub(r'[^a-zA-Z]', ' ', text)
  words = text.lower().split()
  if remove_stopwords:
    words = [w for w in words if w not in eng_stopwords]
  return words

# model_name = '300features_40minwords_10context.model'
# model = Word2Vec.load(os.path.join('.', 'models', model_name))