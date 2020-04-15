import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def box_plot(pd_data, columns, title=None, x_label=None, y_label=None, filename=None):
	plt.figure(figsize=(max(len(columns), 6), 4))
	box_attrib=pd_data.boxplot(column=columns, grid=False, showfliers=False)
	plt.xlabel(x_label)
	plt.ylabel(y_label, rotation=0)
	box_attrib.yaxis.set_label_coords(-0.08, 0.9)
	plt.title(title)
	if filename != None:
		plt.savefig(filename)
	plt.show()

def box_plot_attrib():
	extra_column_attrib = ['neg', 'neu', 'pos', 'compound']
	column_attrib = ['issue_name', 'attrib_change']
	column_attrib.extend(extra_column_attrib)
	sentiment_attrib = pd.read_csv('20170418_attrib_sentiment_all.csv', index_col='issue_name', usecols=column_attrib)
	
	box_plot(sentiment_attrib, extra_column_attrib, title='sentiment boxplot for attrib', y_label='sentiment', filename='plot_box/box_attrib.png')
	box_plot(sentiment_attrib[sentiment_attrib['attrib_change']==1], extra_column_attrib, title='sentiment boxplot for attrib_change 1', \
			y_label='sentiment', filename='plot_box/box_attrib_change_1.png')
	box_plot(sentiment_attrib[sentiment_attrib['attrib_change']==0], extra_column_attrib, title='sentiment boxplot for attrib_change 0', \
			y_label='sentiment', filename='plot_box/box_attrib_change_0.png')

def box_plot_resolution():
	extra_column_resolution = ['neg_25', 'neu_25', 'pos_25', 'compound_25', 'neg_75', 'neu_75', 'pos_75', 'compound_75', \
		'neg_diff', 'neu_diff', 'pos_diff', 'compound_diff']
	column_resolution = ['issue_name', 'Resolution']
	column_resolution.extend(extra_column_resolution)
	pd_sentiment = pd.read_csv('20190716_merged_data_subsetvariables_183_sentiment.csv', index_col='issue_name', usecols=column_resolution)
	pd_sentiment_resolution_0 = pd_sentiment[pd_sentiment['Resolution']!=1]
	pd_sentiment_resolution_1 = pd_sentiment[pd_sentiment['Resolution']==1]
	
	
	box_plot(pd_sentiment, extra_column_resolution, title='sentiment boxplot for resolution', y_label='sentiment', filename='plot_box/box_resolution.png')
	box_plot(pd_sentiment_resolution_1, extra_column_resolution, title='sentiment boxplot for resolution 1', \
			y_label='sentiment', filename='plot_box/box_resolution_1.png')
	box_plot(pd_sentiment_resolution_0, extra_column_resolution, title='sentiment boxplot for resolution 0', \
			y_label='sentiment', filename='plot_box/box_resolution_0.png')

if __name__ == '__main__':

	box_plot_attrib()
	box_plot_resolution()
