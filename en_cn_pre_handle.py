import time
import numpy as np
import jieba
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def pre_handle(inputfilename):
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


	# 计算词频
	cn_word_list = [w for s in df.cn for w in s]
	en_word_list = [w for s in df.en for w in s]

	cn_words_freq = pd.Series(cn_word_list).value_counts()
	en_words_freq = pd.Series(en_word_list).value_counts()

	# 准备好四个待计算共现词频的变量
	en_sentences_list = df.en.to_list() 
	cn_sentences_list = df.cn.to_list()
	return en_words_freq, cn_words_freq, en_sentences_list, cn_sentences_list






















