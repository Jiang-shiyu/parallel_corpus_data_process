import en_cn_pre_handle

if __name__ == '__main__':

	en_words_freq, cn_words_freq, en_list, cn_list = pre_handle(inputfilename='en_cn(1000).txt', outputfilename='en_cn_prehandled.csv') # 预处理语料
	# en_list, cn_list 是已经处理好的双语语料，包括分词、还原等，是一个二维列表，[[word,word][word,word][word,word]]
	pairs=make_pairs(en_words_freq, cn_words_freq)
	combine_sentence_lists=combine_en_jp(en_list, cn_list)
	count_confreq(en_words_freq, cn_words_freq, en_list, cn_list，'cn_en_confreq.csv') # 计算共现词频，这个过程耗费时间最多
	calculate_f(en_words_freq, cn_words_freq, 'cn_en_confreq.csv', 'sorted_full_data_with_f.csv') # 计算 F-measure
