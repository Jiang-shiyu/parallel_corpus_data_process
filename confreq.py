import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def count_confreq(jp_word_list, en_word_list, combine_sentence_separated_list):

	# 将不重复的英语单词和日语单词两两组队
	jp_set = set(jp_word_list)
	en_set = set(en_word_list)
	pairs = [(x,y) for x in en_set for y in jp_set]
	confreq={}
	for en,jp in pairs:
	    confreq[(jp,en)]=0
	    for line in combine_sentence_separated_list:
	        if (en in line) and (jp in line):
	            confreq[(jp,en)] += 1
	return confreq

def isconfreq(confreq):
	confreq_is = {key:value for key,value in confreq.items() if value!=0} # 过滤共现频数为0的词对
	return confreq_is	

# 本程序用于计算平行语料库共现频数

data = pd.read_csv('en2jp.csv')

# 处理英语数据
en_sentence_list = [i for i in data.en]
en_sentence_separated_list = [word_tokenize(t) for t in en_sentence_list]
en_word_list = [word for s in en_sentence_separated_list for word in s]

jp_sentence_separated_list = [s.split('/') for s in data.jp]
jp_word_list = [word for s in jp_sentence_separated_list for word in s]

# 将英语和日语对译句合成一个列表，列表内元素为单词，以备共现查询
combine_sentence_separated_list = []
for i in range(len(jp_sentence_separated_list)):
	combine = en_sentence_separated_list[i]+jp_sentence_separated_list[i]
	combine_sentence_separated_list.append(combine)



# 共现频数查询

confreq = count_confreq(jp_word_list, en_word_list, combine_sentence_separated_list)
# confreq_pair = isconfreq(confreq)

# 将共现词和共现频数输出
data = pd.Series(confreq)
confreq_df = pd.DataFrame(data)
confreq_df.to_csv('confreq.csv')
