import csv
import numpy as np
import io
import re
import os
import nltk
import pandas as pd
from nltk import data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import WordPunctTokenizer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim.parsing.porter import PorterStemmer
import matplotlib.pyplot as plt

def read_txt(filename):
	with open(filename, 'r') as f:
		text = f.read()
		return text
		
def sentiment_analysis(sen_tokenizer, sentiment_analyzer, text):

	scores = []
	sentences = sen_tokenizer.tokenize(text)
	if (len(text)==0):
		return np.array([0, 0, 0, 0])
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
		#print('sentence score            ', score_sentence)
		scores.append(score_sentence)
		#print()
	return np.mean(scores, axis=0)
	
def sentiment_analysis_list(sen_tokenizer, sentiment_analyzer, comments):
	scores = []
	if (len(comments)==0):
		return np.array([0, 0, 0, 0])
	for comment in comments:
		#comment = preprocess(comment)
		score = sentiment_analysis(sen_tokenizer, sentiment_analyzer, comment)
		#print ('comment score:    ', score)
		scores.append(score)
	return np.mean(scores, axis=0)


def lemmatize_stemming(text):
	stemmer = SnowballStemmer("english")
	return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
	
def preprocess(text):
	result = []
	for token in gensim.utils.simple_preprocess(text):
		if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
			result.append(lemmatize_stemming(token))
	return ' '.join(result)
	
def figure_plot(title, x, y_list, x_label, y_label, label, color_list, filename=None, mode=None):
	plt.figure()
	for i in range(len(y_list)):
		if mode == 'scatter':
			plt.scatter(x, y_list[i], c=color_list[i], s=1, linewidths=0, label=label[i])
		else:
			plt.plot(x, y_list[i], color=color_list[i], linestyle="-", linewidth=1, label=label[i])
	plt.legend()
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	if filename != None:
		plt.savefig(filename)
	plt.show()
	
if __name__=="__main__":

	data.path.append((r"/home/xxxxx/share/nltk_data"))
	sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentiment_analyzer = SentimentIntensityAnalyzer()
	data_file = '20170418_attrib.csv'
	pdf_file = 'merged.csv'
	extra_columns = ['neg_before_attrib', 'neu_before_attrib', 'pos_before_attrib', 'compound_before_attrib', 'neg_after_attrib', 'neu_after_attrib', \
		'pos_after_attrib', 'compound_after_attrib', 'neg_diff', 'neu_diff', 'pos_diff', 'compound_diff']

	pdf_filename = pd.read_csv(pdf_file, usecols=['issue_name', 'filename'])
	data = pd.read_csv(data_file, usecols=['issue_name', 'attrib_change', 'attrib_change_comment'])
	
	data = data.merge(pdf_filename, how='left', on='issue_name')
	#data = data[data['attrib_change']==1]
	
	'''
	#gengerate target table
	sentiment_results = np.array([])
	empty_array = np.full((len(extra_columns), ), '')
	for i in range(data.shape[0]):		
		if data.loc[i]['attrib_change']==0:
			if i==0:
				sentiment_results = empty_array
			else:
				sentiment_results = np.vstack((sentiment_results, empty_array))
			continue
		text = read_txt("txt/"+ data.loc[i]['filename'] + '.txt')
		text = text.replace(u'\xa0', u' ')
		text = text.replace(u'\x0c', u'\n')
		text = text.encode('ascii', errors = 'ignore')	
		text = text.decode('UTF-8')
		text = re.sub('\n+', '\n', text)
		text = re.sub(' +', ' ', text)
		
		comments = re.split(r'commented on', text)
		num = int(data.loc[i]['attrib_change_comment'])
		print(i, len(comments), num)
		sentiment_before_attrib = sentiment_analysis_list(sen_tokenizer, sentiment_analyzer, comments[:num])
		sentiment_after_attrib = sentiment_analysis_list(sen_tokenizer, sentiment_analyzer, comments[num+1:])
		#print(i, sentiment_before_attrib, sentiment_after_attrib)
		
		
		sentiment_result=np.asarray(np.concatenate((sentiment_before_attrib, sentiment_after_attrib, sentiment_before_attrib-sentiment_after_attrib)))
		
		
		if i==0:
			sentiment_results = sentiment_result
		else:
			sentiment_results = np.vstack((sentiment_results, sentiment_result))
		
	print(sentiment_results)
	target_data = pd.read_csv(data_file, index_col='issue_name')
	
	columns_names = target_data.columns.to_list()	
	columns_names.extend(extra_columns)
	#print(columns_names)
	final_data = np.concatenate((target_data.values, sentiment_results), axis=1)
	#target_data = pd.concat([target_data, pd.DataFrame(sentiment_results, index=data.index, columns=columns_names)])
	pd.DataFrame(final_data, index = target_data.index, columns=columns_names).to_csv('20170418_attrib_sentiment.csv')
	'''
	#statistics
	columns = ['issue_name', 'attrib_change']
	columns.extend(extra_columns)
	pd_sentiment = pd.read_csv('20170418_attrib_sentiment.csv', index_col='issue_name', usecols=columns)
	print(pd_sentiment.shape)
	#pd_sentiment_resolution_0 = pd_sentiment[pd_sentiment['Resolution']!=1]
	#pd_sentiment_resolution_1 = pd_sentiment[pd_sentiment['Resolution']==1]
	
	pd_stat = pd.concat([pd_sentiment[extra_columns].mean(axis=0), pd_sentiment[extra_columns].std(axis=0)], axis=1)
	#stat_resolution_0 = pd.concat([pd_sentiment_resolution_0[extra_columns].mean(axis=0), pd_sentiment_resolution_0[extra_columns].std(axis=0)],axis=1)
	#stat_resolution_1 = pd.concat([pd_sentiment_resolution_1[extra_columns].mean(axis=0), pd_sentiment_resolution_1[extra_columns].std(axis=0)],axis=1)
	#pd_stat = pd.concat([stat_total, stat_resolution_0, stat_resolution_1], axis=1)
	pd_stat.columns=['mean', 'std']
	pd_stat.to_csv('sentiment_statistics.csv')
	print(pd_stat)
	pd_sentiment = pd_sentiment[pd_sentiment['attrib_change']==1]
	#plot part
	figure_plot(title='sentiment before attrib', x=np.arange(pd_sentiment.shape[0]), y_list=pd_sentiment[extra_columns[:4]].values.T, \
		x_label='issue index', y_label='sentiment', label=extra_columns[:4], color_list=['b', 'g', 'r', 'c'], filename='plot_attrib/sentiment_before_attrib.png', mode=None)
	figure_plot(title='sentiment after attrib', x=np.arange(pd_sentiment.shape[0]), y_list=pd_sentiment[extra_columns[4:8]].values.T, \
		x_label='issue index', y_label='sentiment', label=extra_columns[4:8], color_list=['b', 'g', 'r', 'c'], filename='plot_attrib/sentiment_after_attrib.png', mode=None)
	figure_plot(title='sentiment diff before and after attrib', x=np.arange(pd_sentiment.shape[0]), y_list=pd_sentiment[extra_columns[8:]].values.T, \
		x_label='issue index', y_label='sentiment', label=extra_columns[8:], color_list=['b', 'g', 'r', 'c'], filename='plot_attrib/diff.png', mode=None)
			
