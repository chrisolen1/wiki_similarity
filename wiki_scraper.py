#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Libraries
from bs4 import BeautifulSoup
from string import digits
import requests
import urllib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
import seaborn as sns

class wiki_scrape:
    def __init__(self, url):
        self.url = url

    def scrape_to_str(self):
        html = requests.get(self.url)
        soup = BeautifulSoup(html.content)
        wiki_str = ""
        for entry in soup.find_all(name='p'):
            paragraph = entry.get_text()
            wiki_str += paragraph
        lower_str = wiki_str.lower()
        remove_digits = str.maketrans('','',digits)
        alpha_string = lower_str.translate(remove_digits)
        cleaned_up = alpha_string.replace(".", "").replace(',',"").replace("\n"," ").replace("\\displaystyle","").replace("\\overline","").replace("\\ldots","").replace("{","").replace('}',"").replace("(","").replace(")","").replace("[","").replace("]","").replace("_","")
        wiki_split = cleaned_up.split()
        from nltk.stem.wordnet import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        wiki_lem_n = [wnl.lemmatize(x, pos = 'n') for x in wiki_split]
        wiki_lem_nv = [wnl.lemmatize(x, pos = 'v') for x in wiki_lem_n]
        wiki_lem_nva = [wnl.lemmatize(x, pos = 'a') for x in wiki_lem_nv]
        delim = " "
        wiki_join = delim.join(wiki_lem_nva)
        return(wiki_join)
  
    def scrape_to_stop_words(self):
        html = requests.get(self.url)
        soup = BeautifulSoup(html.content)
        wiki_str = ""
        for entry in soup.find_all(name='p'):
            paragraph = entry.get_text()
            wiki_str += paragraph
        lower_str = wiki_str.lower()
        remove_digits = str.maketrans('','',digits)
        alpha_string = lower_str.translate(remove_digits)
        cleaned_up = alpha_string.replace(".", "").replace(',',"").replace("\n"," ").replace("\\displaystyle","").replace("\\overline","").replace("\\ldots","").replace("{","").replace('}',"").replace("(","").replace(")","").replace("[","").replace("]","").replace("_","")
        wiki_split = cleaned_up.split()
        from nltk.stem.wordnet import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        wiki_lem_n = [wnl.lemmatize(x, pos = 'n') for x in wiki_split]
        wiki_lem_nv = [wnl.lemmatize(x, pos = 'v') for x in wiki_lem_n]
        wiki_lem_nva = [wnl.lemmatize(x, pos = 'a') for x in wiki_lem_nv]
        unique_stop_words = list(set([word for word in wiki_lem_nva if len(word) <= 3])) 
        new_stop_words = text.ENGLISH_STOP_WORDS.union(unique_stop_words)
        return(new_stop_words)

input_list = [input("Please provide a Wikipedia page link to compare: ")]

yn_input = 'Y'
while yn_input == 'Y':
    yn_input = input("Do you want to add another page link to compare? (Y/N) ")
    while yn_input != "Y" and yn_input != "N":
        print("Please answer \"Y\" or \"N\"")
        yn_input = input("Do you want to add another page link to compare? (Y/N) ")
    if yn_input == 'Y':
        next_input = [input("Please provide another Wikipedia page link to compare: ")]
        input_list += next_input
  
combined_doc = list(map(wiki_scrape.scrape_to_str, map(wiki_scrape, input_list)))
combined_stop = frozenset().union(*list(map(wiki_scrape.scrape_to_stop_words, map(wiki_scrape, input_list))))

#TFIDF Vectorizer
tfidf = TfidfVectorizer(stop_words = combined_stop, max_df = .8)
wiki_tfidf = tfidf.fit_transform(combined_doc)
total_features = len(tfidf.get_feature_names())
total_documents = len(wiki_tfidf.toarray())
print("There are %s total features and %s total documents." % (total_features, total_documents))
#Change to DataFrame
def url_to_indexname(url):
    split_url = url.split('/')[-1]
    return split_url
rows_df = list(map(url_to_indexname, input_list))

tfidf_df = pd.DataFrame(wiki_tfidf.toarray(), columns = tfidf.get_feature_names(), index = rows_df)

#Transforming Vector Via LSA
components = int(wiki_tfidf.shape[0]/2)
components = int(total_documents/4)
tfidf_matrix = wiki_tfidf.toarray()
lsa = TruncatedSVD(n_components = components, random_state = 42)
wiki_lsa = lsa.fit_transform(tfidf_matrix)
explained = lsa.explained_variance_ratio_.sum()
while explained < 0.7:
    components += 1
    lsa = TruncatedSVD(n_components = components, random_state = 42)
    wiki_lsa = lsa.fit_transform(tfidf_matrix)
    explained = lsa.explained_variance_ratio_.sum()
print("With %d principal components, %s of the variance is explained." % (components, explained))
sing_values = lsa.explained_variance_ratio_
print("Explained variance per singular value is as follows: %s." % sing_values)
#Change to DataFrame
lsa_df = pd.DataFrame(wiki_lsa, index = rows_df)

#Cosine Similarity
lsa_angle_matrix = cosine_similarity(wiki_lsa)
lsa_angle_matrix_df = pd.DataFrame(lsa_angle_matrix, columns = rows_df, index = rows_df)
lsa_angle_matrix_df.to_csv('/Users/chrisolen/documents/uchicago_courses/linear_algebra_and_matrix_analysis/final_project/lsa_angle.csv', encoding='utf-8')
sns.heatmap(lsa_angle_matrix_df,cmap='RdYlGn', linewidths=0.5, annot = True)



