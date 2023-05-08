# -*- coding: utf-8 -*-
# Example main.py
import os
from pypdf import PdfReader
import pandas as pd
from collections import OrderedDict
import nltk
nltk.download('stopwords')
import spacy
import unicodedata
#from contractions import CONTRACTION_MAP
import re
from nltk.corpus import wordnet
import collections
#from textblob import Word
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd
import joblib

from smart_fun import *



def test_model():

    file_name = 'VA Virginia Beach.pdf'

    f_name, content_list = extract_input(file_name)

    normalized_corpus = normalize_corpus(content_list)

    cluster_id = predicting_cluster(normalized_corpus)

    assert type(f_name)==str and isinstance(content_list, list) and len(content_list) > 0 and len(normalized_corpus) > 0 and isinstance(content_list, list) and len(content_list) > 0

    

    