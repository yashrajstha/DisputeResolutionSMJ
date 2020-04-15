import csv
import numpy as np
import io
import re
import os
import nltk
import pandas as pd

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim.parsing.porter import PorterStemmer


from nltk import data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import WordPunctTokenizer

from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

def extract_text_from_pdf(pdf_path):
	command = "pdf2txt.py {}".format(pdf_path)
	fd = os.popen(command)
	text = fd.read()
	fd.close()
	return text

def read_data(filename):
	#issue = {}
	data = []
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		next(reader)
		for row in reader:
			#issue[row[0]] = row
			data.append(row)
		return np.array(data)

def read_txt(filename):
	with open(filename, 'r') as f:
		text = f.read()
		return text
	

def lemmatize_stemming(text):
	stemmer = SnowballStemmer("english")
	return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
	
def preprocess(text):
	result = []
	for token in gensim.utils.simple_preprocess(text):
		if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
			result.append(lemmatize_stemming(token))
	return ' '.join(result)

def split_resolution(pdf_file, index_file):
	pdf_filename = pd.read_csv(pdf_file, index_col='issue_name', usecols=['issue_name', 'filename'])
	data = pd.read_csv(index_file, index_col='issue_name', usecols=['issue_name', 'Resolution'])
	#print(pdf_filename.shape, data.shape)
	resolution_index = data[data['Resolution']==1].index.values
	no_resolution_index = data[data['Resolution']!=1].index.values
	#print(resolution_index.shape, no_resolution_index.shape)
	return pdf_filename.loc[resolution_index]['filename'].values, pdf_filename.loc[no_resolution_index]['filename'].values

def documents(pdf_txt_list):
	docs = []
	for txt_i in pdf_txt_list:
		if not os.path.exists("txt/"+ txt_i + '.txt'):
			try:
				filename = "/home/huangyuanhao/share/sentiment/github/" + txt_i + ".pdf"
				#print(filename)
				text = extract_text_from_pdf(filename)
				with open("txt/"+ txt_i + '.txt', 'w') as f_txt:
					f_txt.write(text)
			except:
				print("failed to load " + filename)
		else:
			text = read_txt("txt/"+ txt_i + '.txt')
			text = text.replace(u'\xa0', u' ')
			text = text.replace(u'\x0c', u'\n')
			text = text.encode('ascii', errors = 'ignore')	
			text = text.decode('UTF-8')
			text = re.sub('\n+', '\n', text)
			preprocess_doc = preprocess(text)
			docs.append(preprocess_doc)
	return np.array(docs)
	
def display_topics(model, feature_names, no_top_words):

	for topic_idx, topic in enumerate(model.components_):
		print ("Topic %d:" % (topic_idx))
		print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


def sklearn_LDA(num_features, num_topics, num_top_words, documents):
	tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
	tf = tf_vectorizer.fit_transform(documents)
	tf_feature_names = tf_vectorizer.get_feature_names()
	lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='batch', random_state=0).fit(tf)
	display_topics(lda, tf_feature_names, num_top_words)
	return lda

def focus_measurement(distribution):
	norm_dis = distribution / distribution.sum(axis=1)[:, np.newaxis]
	N, A = norm_dis.shape
	mul_dis = norm_dis * norm_dis
	return mul_dis.sum()/10;

if __name__ == "__main__":

	data.path.append((r"/home/huangyuanhao/share/nltk_data"))
	pdf_file = "merged.csv"
	index_file = 'data_new_183.csv'
	
	resolution_1_index, resolution_0_index = split_resolution(pdf_file, index_file)
	resolution_1_doc = documents(resolution_1_index)
	resolution_0_doc = documents(resolution_0_index)
	print(len(resolution_1_doc), len(resolution_0_doc))

	num_features = 10000
	num_topics = 10
	num_top_words = 10
	lda_resolution_1 = sklearn_LDA(num_features, num_topics, num_top_words, resolution_1_doc)
	lda_resolution_0 = sklearn_LDA(num_features, num_topics, num_top_words, resolution_0_doc)
	
	#print(lda_resolution_1.components_.shape)
	#print(lda_resolution_0.components_.shape)
	focus_resolution_0 = focus_measurement(lda_resolution_0.components_)
	focus_resolution_1 = focus_measurement(lda_resolution_1.components_)
	print(focus_resolution_0, focus_resolution_1)
	# 0.04558190438762838 0.08157190257491989
	
	
	
	
	
	
	


