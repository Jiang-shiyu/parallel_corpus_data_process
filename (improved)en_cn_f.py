import time
import numpy as np
import jieba
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def make_pairs(en_words_freq, jp_words_freq):
	en_wmt2 = en_words_freq.where(en_words_freq>2).dropna().index.to_list()
	jp_wmt2 = jp_words_freq.where(jp_words_freq>2).dropna().index.to_list()

	pairs = [(en,jp) for en in en_wmt2 for jp in jp_wmt2]

	return pairs

def combine_en_jp(lemmatized_en_sentences, lemmatized_jp_sentences):
	combine_sentence_lists = []
	for i in range(len(lemmatized_jp_sentences)):
		combine = lemmatized_en_sentences[i]+lemmatized_jp_sentences[i]
		combine_sentence_lists.append(combine)
	return combine_sentence_lists



def count_confreq(pairs, combine_sentence_lists, outputfilename):

	start = time.time()
	confreq={}
	for en,jp in pairs:
		confreq[(jp,en)]=0
		for line in combine_sentence_lists:
			if (en in line) and (jp in line):
				confreq[(jp,en)] += 1
	end = time.time()
	print(str(end-start))
	pd.Series(confreq).to_csv(outputfilename)
	return confreq

def calculate_f(en_words_freq, cn_words_freq, inputfilename, outputfilename):
	# 构建英日词频df，准备将英日词频加入到df中
	confreq = pd.read_csv(inputfilename)
	confreq = confreq.rename(columns={'Unnamed: 0':'cn','Unnamed: 1':'en','0':'confreq'}).dropna() 

	df_en = pd.DataFrame({'en_freq':en_words_freq}, index=None)
	df_cn = pd.DataFrame({'cn_freq':cn_words_freq}, index=None)

	data = confreq.join(df_en, on='en').join(df_cn, on='cn') # 将英日词频加入df
	
	data['F_measure'] = (2*(data.confreq/data.en_freq) * (data.confreq/data.cn_freq)) / (data.confreq/data.en_freq + data.confreq/data.cn_freq)
	data['F_measure'] = data['F_measure'].fillna(0) # 将空值设置为0
	sorted_full_data_with_f = data.sort_values(by='F_measure', ascending=False) # 使df以f值降序排列
	sorted_full_data_with_f.to_csv(outputfilename)
	return sorted_full_data_with_f



def pre_handle(inputfilename, outputfilename):
	with open(inputfilename, 'r') as f:
	    f_list = [line for line in f]

	f_list_s = [i.split('\t') for i in f_list]
	df = pd.DataFrame(f_list_s, columns=['cn','en'])

	# 中文语料库处理：分词
	cn_list=[jieba.lcut(s) for s in df.cn]
	df.cn=cn_list

	# 英文语料库处理：分词、lowercase、lemmatizer
	wnl = WordNetLemmatizer()
	tokenized_sentences = [word_tokenize(sentence) for sentence in df.en]

	for i in range(len(tokenized_sentences)):
	    for j in range(len(tokenized_sentences[i])):
	        tokenized_sentences[i][j] = wnl.lemmatize(tokenized_sentences[i][j].lower())

	df.en=tokenized_sentences

	df.to_csv(outputfilename)

	# 计算词频
	cn_word_list = [w for s in df.cn for w in s]
	en_word_list = [w for s in df.en for w in s]

	cn_words_freq = pd.Series(cn_word_list).value_counts()
	en_words_freq = pd.Series(en_word_list).value_counts()

	# 准备好四个待计算共现词频的变量
	en_list = df.en.to_list() 
	cn_list = df.cn.to_list()
	return en_words_freq, cn_words_freq, en_list, cn_list

if __name__ == '__main__':

	en_words_freq, cn_words_freq, en_list, cn_list = pre_handle(inputfilename='en_cn(1000).txt', outputfilename='en_cn_prehandled.csv') # 预处理语料
	# en_list, cn_list 是已经处理好的双语语料，包括分词、还原等，是一个二维列表，[[word,word][word,word][word,word]]
	pairs=make_pairs(en_words_freq, cn_words_freq)
	combine_sentence_lists=combine_en_jp(en_list, cn_list)
	count_confreq(en_words_freq, cn_words_freq, en_list, cn_list，'cn_en_confreq.csv') # 计算共现词频，这个过程耗费时间最多
	calculate_f(en_words_freq, cn_words_freq, 'cn_en_confreq.csv', 'sorted_full_data_with_f.csv') # 计算 F-measure






















