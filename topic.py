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

'''
def extract_text_from_pdf(pdf_path):
	resource_manager = PDFResourceManager()
	fake_file_handle = io.StringIO()
	converter = TextConverter(resource_manager, fake_file_handle)
	page_interpreter = PDFPageInterpreter(resource_manager, converter)

	with open(pdf_path, 'rb') as fh:
		for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
			page_interpreter.process_page(page)
		text = fake_file_handle.getvalue()
	# close open handles
	converter.close()
	fake_file_handle.close()
	if text:
		return text
'''

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
	
def sentiment_analysis(sen_tokenizer, sentiment_analyzer, text):

	scores = []
	sentences = sen_tokenizer.tokenize(text)
	for sentence in sentences:
		#print(sentence)
		score_sentence = []
		score = sentiment_analyzer.polarity_scores(sentence)
		#print(type(score))
		'''
		for k in score:
			print('{0}:{1},'.format(k, score[k]), end='')
			score_sentence.append(score[k])
		'''
		score_sentence.append(score['neg'])
		score_sentence.append(score['neu'])
		score_sentence.append(score['pos'])
		score_sentence.append(score['compound'])
		scores.append(score_sentence)
		#print()
	return np.mean(scores, axis=0)

def lemmatize_stemming(text):
	stemmer = SnowballStemmer("english")
	return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
	
def preprocess(text):
	result = []
	for token in gensim.utils.simple_preprocess(text):
		if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
			result.append(lemmatize_stemming(token))
	return result

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
				filename = "/home/xxxxx/share/sentiment/github/" + txt_i + ".pdf"
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
			docs.extend(preprocess_doc)
	return np.array(docs)
	
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def sklearn_LDA(num_features, num_topics, num_top_words, documents):
	tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
	tf = tf_vectorizer.fit_transform(documents)
	tf_feature_names = tf_vectorizer.get_feature_names()
	lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=10.,random_state=0).fit(tf)
	display_topics(lda, tf_feature_names, num_top_words)
	
if __name__ == "__main__":

	data.path.append((r"/home/xxxx/share/nltk_data"))
	
	#sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	#sentiment_analyzer = SentimentIntensityAnalyzer()
	pdf_file = "merged.csv"
	index_file = 'data_new_183.csv'
	
	resolution_1_index, resolution_0_index = split_resolution(pdf_file, index_file)
	#print(resolution_1_index, resolution_0_index)
	
	resolution_1_doc = documents(resolution_1_index)
	resolution_0_doc = documents(resolution_0_index)
	print(resolution_1_doc[0])
	print(resolution_0_doc[0])
	num_features = 1000
	num_topics = 10
	num_top_words = 10
	sklearn_LDA(num_features, num_topics, num_top_words, resolution_1_doc)
	sklearn_LDA(num_features, num_topics, num_top_words, resolution_0_doc)
	


	
	
	
	
	
	


