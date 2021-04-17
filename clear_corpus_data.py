
import numpy as np
import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# 抽出日语列，准备分词
def jp_extract(filename):
	with open(filename) as f:
		f_list = [line for line in f]
	f_list_s = [i.split('\t') for i in f_list]
	df = pd.DataFrame(f_list_s, columns=['en','jp'])
	with open('jp.txt', 'w') as b:
		for i in df.jp:
			b.write(i)
	print('已经把日语提取到 jp.txt, 有返回值 df.en')
	return df.en
# 将media.txt文件用jumanpp解析，将生成的文件名输入下面函数
# 将解析后的词组成句子，再将英语和日语合并成对译的csv

def jp_lemmatize(i='jp.txt',o='output.txt'):
	os.system('jumanpp {inputfile} > {outputfile}'.format(inputfile=i, outputfile=o))
	print('已经用jumanpp把日语解析到 output.txt')
    
def juamnpp2sentences(filename='output.txt'):
	with open(filename) as f:
		f_list = [line for line in f]
	f_list_s = [i.split(' ') for i in f_list]
	df = pd.DataFrame(f_list_s, columns=[i for i in range(1,19)])
	df_fillna = df[3].fillna('EOS\n') # 用第三列当做词典形
	# 将词合并为整个文本
	text = ''
	for i in df_fillna:
		text = text + '/' + i
	# 将文本分成句子
	sentences = text.split('EOS\n/')
	# 除去句子里的空格造成的双斜杠
	jp = [i.replace('//', '/') for i in sentences]
	# 处理工作完成，将已经处理过的日语和英语对译句放入csv
	df2 = pd.DataFrame()
	df2['jp'] = jp 

	print('已完成日语分词,有返回值 df.jp')
	return df2.jp


def tokenize_and_lemmatize(en=en):
# 使英文语料[df.Series]实现分词并lemmatize
# 返回一个DataFrame
	wnl = WordNetLemmatizer()
	en_sentence_list = en.to_list()
	tokenized_and_lemmatized_sentences = [word_tokenize(t) for t in en_sentence_list]

	for i in range(len(tokenized_and_lemmatized_sentences)):
		for j in range(len(tokenized_and_lemmatized_sentences[i])):
			tokenized_and_lemmatized_sentences[i][j] = wnl.lemmatize(tokenized_and_lemmatized_sentences[i][j])

	en_word_list = [word for sen in tokenized_and_lemmatized_sentences for word in sen]
	return tokenized_and_lemmatized_sentences, en_word_list


en = jp_extract(filename)
jp_lemmatize()
lemmatized_jp_sentences = juamnpp2sentences()
lemmatized_en_sentences,en_word_list = tokenize_and_lemmatize()





















