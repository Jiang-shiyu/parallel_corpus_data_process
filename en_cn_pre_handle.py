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

	cn_words_s = pd.Series([w for s in df.cn for w in s])
	en_words_s = pd.Series([w for s in df.en for w in s])

	cn_words_s, en_words_s = clear_words_list(cn_words_s, en_words_s)
	cn_words_freq = cn_words_s.value_counts()
	en_words_freq = en_words_s.value_counts()
	# 准备好四个待计算共现词频的变量
	en_sentences_list = df.en.to_list() 
	cn_sentences_list = df.cn.to_list()
	return en_words_freq, cn_words_freq, en_sentences_list, cn_sentences_list

def clear_words_list(cn_words_s, en_words_s):
	#设置需要过滤的词
	cn_stopwords = '[0-9。，、！？ \.\(\)]'
	en_stopwords = '[0-9\.\,\?\(\)]'
	#过滤中文，得到干净的中文词表
	cn_replace_stopwords = cn_words_s.str.replace(cn_stopwords, '',regex=True)
	cn_blanc_index = cn_replace_stopwords[cn_replace_stopwords==''].index
	cn_words_s = cn_replace_stopwords.drop(cn_blanc_index)
	#cn_clear_words_list = cn_words_s.to_list()
	#过滤英文，得到干净的英文词表
	en_replace_stopwords = en_words_s.str.replace(en_stopwords, '', regex=True)
	en_blanc_index = en_replace_stopwords[en_replace_stopwords==''].index
	en_words_s = en_replace_stopwords.drop(en_blanc_index)
	#en_clear_words_list = en_words_s.to_list()
	return cn_words_s, en_words_s
