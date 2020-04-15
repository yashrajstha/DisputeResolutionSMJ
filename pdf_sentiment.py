import csv
import numpy as np
import io
import re
import os
import nltk
from nltk import data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import WordPunctTokenizer

from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from datetime import datetime

"""
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
"""

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
	
if __name__ == "__main__":

	data.path.append((r"/home/huangyuanhao/share/nltk_data"))

	#text = extract_text_from_pdf("atom_1155_VS[1].pdf")
	#index = text.index("hoolio commented on Nov 26, 2013")
	#part_one = text[:index]
	#part_two = text[index:]
	#part_one = part_one.lower()
	
	sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentiment_analyzer = SentimentIntensityAnalyzer()
	num_new_col = 8
	data_file = "/home/huangyuanhao/share/merged.csv"
	target_file = "merged_sentiment_final_183.csv"
	index_file = 'data_new_183.csv'
	
	new_columns = ['neg_p1', 'neu_p1', 'pos_p1', 'compound_p1', 'neg_p2', 'neu_p2', 'pos_p2', 'compound_p2']
	
	data = read_data(data_file)
	with open(data_file, 'r') as f:
		reader = csv.reader(f)
		for i, row in enumerate(reader):
			if i == 0:
				columns = row;
	columns.extend(new_columns)
	print(columns)
	
	#sentiment_result = np.empty((data.shape[0], num_new_col))
	sentiment_result = np.full((data.shape[0], num_new_col), '')
	data = np.concatenate((data, sentiment_result), axis=1)
	
	for i in range(data.shape[0]):
		if data[i][8] == "1":
			if data[i][13] == '':
				#data[i][-num_new_col:] = ''
				continue
			else:
				str_date = data[i][13]
				flag_year = True
				try:
					date = datetime.strptime(str_date, '%d-%b-%y')
				except ValueError:
					flag_year = False
					date = datetime.strptime(str_date, '%d-%b')
				'''
				try:
					filename = "/home/huangyuanhao/share/sentiment/github/" + data[i][1] + ".pdf"
					#print(filename)
					text = extract_text_from_pdf(filename)
					with open("txt/"+ data[i][1] + '.txt', 'w') as f_txt:
						f_txt.write(text)
				except:
					print("failed to load " + filename)
				'''
				text = read_txt("txt/"+ data[i][1] + '.txt')
				text = text.replace(u'\xa0', u' ')
				#text = text.replace(u'\xad', u'')
				text = text.replace(u'\x0c', u'\n')
				text = text.encode('ascii', errors = 'ignore')	
				text = text.decode('UTF-8')
				text = re.sub('\n+', '\n', text)
				# print(text)
				#comment = "{} commented on {}".format(data[i][10], data[i][13])
				if flag_year:
					comment = "{} commented on {}".format(data[i][10], date.strftime('%b %-d, %Y'))
				else:
					comment = "{} commented on {}".format(data[i][10], date.strftime('%b %-d'))
				try:
					index = text.index(comment)
					p1_score = sentiment_analysis(sen_tokenizer, sentiment_analyzer, text[:index])
					p2_score = sentiment_analysis(sen_tokenizer, sentiment_analyzer, text[index:])
					#print(p1_score, p2_score)
					#print(data[i].shape, data.shape)
					data[i] = np.concatenate((data[i][:-num_new_col], p1_score, p2_score))
				except ValueError:
					print(i, data[i][0], comment)
					#data[i][-num_new_col:] = ''
					continue;
	'''
	with open(target_file, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(columns)
		writer.writerows(data)
	'''
	with open(index_file, 'r') as f:
		reader = csv.reader(f)
		next(reader)
		issue_names = [row[0] for row in reader]
		issue_dict = dict.fromkeys(issue_names)
		
		idx = []
		for i in range(len(data)):
			if data[i][0] in issue_dict:
				idx.append(True)
			else:
				idx.append(False)
		with open(target_file, 'w') as tf:
			writer = csv.writer(tf)
			writer.writerow(columns)
			writer.writerows(data[idx])

	
	
	
	
	
	
	


