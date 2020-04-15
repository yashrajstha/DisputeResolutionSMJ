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
	data_file = '20190716_merged_data_subsetvariables_183.csv'
	pdf_file = 'merged.csv'
	extra_columns = ['neg_25', 'neu_25', 'pos_25', 'compound_25', 'neg_75', 'neu_75', 'pos_75', 'compound_75', \
		'neg_diff', 'neu_diff', 'pos_diff', 'compound_diff']

	pdf_filename = pd.read_csv(pdf_file, index_col='issue_name', usecols=['issue_name', 'filename'])
	data = pd.read_csv(data_file, index_col='issue_name', usecols=['issue_name', 'totalcomments'])
	txt_list = pdf_filename.loc[data.index.values]['filename'].values
	total_comments = data['totalcomments'].values
	'''
	#gengerate target table
	sentiment_results = np.array([])
	for i in range(len(txt_list)):
	
		text = read_txt("txt/"+ txt_list[i] + '.txt')
		text = text.replace(u'\xa0', u' ')
		text = text.replace(u'\x0c', u'\n')
		text = text.encode('ascii', errors = 'ignore')	
		text = text.decode('UTF-8')
		text = re.sub('\n+', '\n', text)
		text = re.sub(' +', ' ', text)
		#text = text.replace(u'\n', u' ')
		
		comments = re.split(r'commented on', text)
		#print (abs(len(comments)-total_comments[i]), len(comments), txt_list[i])
		num = int(max(1, len(comments)/4))
		sentiment_first_25_percent = sentiment_analysis_list(sen_tokenizer, sentiment_analyzer, comments[:num])
		sentiment_last_25_percent = sentiment_analysis_list(sen_tokenizer, sentiment_analyzer, comments[len(comments)-num:])
		#print(i, sentiment_first_25_percent, sentiment_last_25_percent)
		
		sentiment_result = np.concatenate((sentiment_first_25_percent, sentiment_last_25_percent, sentiment_first_25_percent-sentiment_last_25_percent))
		if i==0:
			sentiment_results = sentiment_result
		else:
			sentiment_results = np.vstack((sentiment_results, sentiment_result))
		
	print(sentiment_results.shape)
	target_data = pd.read_csv(data_file, index_col='issue_name')
	
	columns_names = target_data.columns.to_list()	
	columns_names.extend(extra_columns)
	print(columns_names)
	final_data = np.concatenate((target_data.values, sentiment_results), axis=1)
	#target_data = pd.concat([target_data, pd.DataFrame(sentiment_results, index=data.index, columns=columns_names)])
	pd.DataFrame(final_data, index = target_data.index, columns=columns_names).to_csv('20190716_merged_data_subsetvariables_183_sentiment.csv')
	'''
	#statistics
	columns = ['issue_name', 'Resolution']
	columns.extend(extra_columns)
	pd_sentiment = pd.read_csv('20190716_merged_data_subsetvariables_183_sentiment.csv', index_col='issue_name', usecols=columns)
	pd_sentiment_resolution_0 = pd_sentiment[pd_sentiment['Resolution']!=1]
	pd_sentiment_resolution_1 = pd_sentiment[pd_sentiment['Resolution']==1]
	
	stat_total = pd.concat([pd_sentiment[extra_columns].mean(axis=0), pd_sentiment[extra_columns].std(axis=0)], axis=1)
	stat_resolution_0 = pd.concat([pd_sentiment_resolution_0[extra_columns].mean(axis=0), pd_sentiment_resolution_0[extra_columns].std(axis=0)],axis=1)
	stat_resolution_1 = pd.concat([pd_sentiment_resolution_1[extra_columns].mean(axis=0), pd_sentiment_resolution_1[extra_columns].std(axis=0)],axis=1)
	pd_stat = pd.concat([stat_total, stat_resolution_0, stat_resolution_1], axis=1)
	pd_stat.columns=['mean_total', 'std_total', 'mean_resolution_0', 'std_resolution_0', 'mean_resolution_1', 'std_resolution_1']
	pd_stat.to_csv('sentiment_statistics.csv')
	print(pd_stat)
	'''
	#plot part
	figure_plot(title='sentiment for top 25% comments', x=np.arange(pd_sentiment.shape[0]), y_list=pd_sentiment[extra_columns[:4]].values.T, \
		x_label='issue index', y_label='sentiment', label=extra_columns[:4], color_list=['b', 'g', 'r', 'c'], filename='plot/25.png', mode=None)
	figure_plot(title='sentiment for last 25% comments', x=np.arange(pd_sentiment.shape[0]), y_list=pd_sentiment[extra_columns[4:8]].values.T, \
		x_label='issue index', y_label='sentiment', label=extra_columns[4:8], color_list=['b', 'g', 'r', 'c'], filename='plot/75.png', mode=None)
	figure_plot(title='sentiment diff between top 25% and bot 25% comments', x=np.arange(pd_sentiment.shape[0]), y_list=pd_sentiment[extra_columns[8:]].values.T, \
		x_label='issue index', y_label='sentiment', label=extra_columns[8:], color_list=['b', 'g', 'r', 'c'], filename='plot/diff.png', mode=None)
	
	figure_plot(title='sentiment for top 25% comments resolution 0', x=np.arange(pd_sentiment_resolution_0.shape[0]), y_list=pd_sentiment_resolution_0[extra_columns[:4]].values.T, x_label='issue index', y_label='sentiment', label=extra_columns[:4], color_list=['b', 'g', 'r', 'c'], filename='plot/25_resolution_0.png', mode=None)
	figure_plot(title='sentiment for last 25% comments resolution 0', x=np.arange(pd_sentiment_resolution_0.shape[0]), y_list=pd_sentiment_resolution_0[extra_columns[4:8]].values.T, x_label='issue index', y_label='sentiment', label=extra_columns[4:8], color_list=['b', 'g', 'r', 'c'], filename='plot/75_resolution_0.png', mode=None)
	figure_plot(title='sentiment diff between top 25% and bot 25% comments resolution 0', x=np.arange(pd_sentiment_resolution_0.shape[0]), y_list=pd_sentiment_resolution_0[extra_columns[8:]].values.T, x_label='issue index', y_label='sentiment', label=extra_columns[8:], color_list=['b', 'g', 'r', 'c'], filename='plot/diff_resolution_0.png', mode=None)
	
	figure_plot(title='sentiment for top 25% comments resolution 1', x=np.arange(pd_sentiment_resolution_1.shape[0]), y_list=pd_sentiment_resolution_1[extra_columns[:4]].values.T, x_label='issue index', y_label='sentiment', label=extra_columns[:4], color_list=['b', 'g', 'r', 'c'], filename='plot/25_resolution_1.png', mode=None)
	figure_plot(title='sentiment for last 25% comments resolution 1', x=np.arange(pd_sentiment_resolution_1.shape[0]), y_list=pd_sentiment_resolution_1[extra_columns[4:8]].values.T, x_label='issue index', y_label='sentiment', label=extra_columns[4:8], color_list=['b', 'g', 'r', 'c'], filename='plot/75_resolution_1.png', mode=None)
	figure_plot(title='sentiment diff between top 25% and bot 25% comments resolution 1', x=np.arange(pd_sentiment_resolution_1.shape[0]), y_list=pd_sentiment_resolution_1[extra_columns[8:]].values.T, x_label='issue index', y_label='sentiment', label=extra_columns[8:], color_list=['b', 'g', 'r', 'c'], filename='plot/diff_resolution_1.png', mode=None)
	'''
		
