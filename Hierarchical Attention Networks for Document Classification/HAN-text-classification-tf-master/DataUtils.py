#coding=utf-8
import numpy as np
import json
import pickle
import nltk
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict

#使用nltk分词分句器
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#print (type(sent_tokenizer))
word_tokenizer = WordPunctTokenizer()
#print (type(word_tokenizer))

#记录每个单词及其出现的频率
word_freq = defaultdict(int)
#print (word_freq)

# 读取数据集，并进行分词，统计每个单词出现次数，保存在word freq中
with open('yelp_academic_dataset_review.json', 'rb') as f:
    for line in f:
        review = json.loads(line.decode('utf-8'))
        words = word_tokenizer.tokenize(review['text'])
        #print (type(words))  #list
        #print (len(words))  包含标点符号
        for word in words:
            word_freq[word] += 1
    #print (review)
    #print (type(review))
    print (word_freq[','])
    print (word_freq['.'])
    print ("load finished")

# 将词频表保存下来
with open('word_freq.pickle', 'wb') as g:
    pickle.dump(word_freq, g)
    print (len(word_freq))  #159654
    print ("word_freq save finished")

num_classes = 5
# 将词频排序，并去掉出现次数最小的3个
sort_words = list(sorted(word_freq.items(), key=lambda x:-x[1]))
print (sort_words[:10], sort_words[-10:])

#构建vocablary，并将出现次数小于5的单词全部去除，视为UNKNOW
vocab = {}
i = 1
vocab['UNKNOW_TOKEN'] = 0
for word, freq in word_freq.items():
    if freq > 5:
        vocab[word] = i
        i += 1
print (i) #46960
#print (vocab)
#print (vocab['UNKNOW_TOKEN'])
UNKNOWN = 0

data_x = []
data_y = []
max_sent_in_doc = 30
max_word_in_sent = 30

#将所有的评论文件都转化为30*30的索引矩阵，也就是每篇都有30个句子，每个句子有30个单词
# 不够的补零，多余的删除，并保存到最终的数据集文件之中
with open('yelp_academic_dataset_review.json', 'rb') as f:
    for line in f:
        doc = np.zeros((30,30))
        review = json.loads(line.decode('utf-8'))
        sents = sent_tokenizer.tokenize(review['text'])    #sents是一个list，每个元素是一个句子
        #print (sents)
        #print (len(sents))
        #print (type(sents))
        for i, sent in enumerate(sents):
            if i < max_sent_in_doc:
                # word_to_index = []
                for j, word in enumerate(word_tokenizer.tokenize(sent)):
                    if j < max_word_in_sent:
                            #word_to_index.append(vocab.get(word, UNKNOWN))
                            #print (word_to_index)
                            doc[i][j] = vocab.get(word, UNKNOWN)
                #npad = (0,30 - len(word_to_index))
                #word_to_index = np.pad(word_to_index, npad, 'constant')
                # print (word_to_index)
                # #word_to_index = word_to_index.reshape(1,30)
                #doc.append(word_to_index)
                #print (len(doc))
                #print ('sxl===================')
        label = int(review['stars'])
        labels = [0] * num_classes       #[0, 0, 0, 0, 0]
        labels[label-1] = 1
        # #print (len(doc))
        # npad2 = ((0, 30-len(doc)), (0,0))
        # #print (npad2)
        #doc = np.pad(doc, npad2, 'constant')
        data_y.append(labels)
        data_x.append(doc)
    pickle.dump((data_x, data_y), open('yelp_data', 'wb'))
    print (len(data_x)) #229907
    # length = len(data_x)
    # train_x, dev_x = data_x[:int(length*0.9)], data_x[int(length*0.9)+1 :]
    # train_y, dev_y = data_y[:int(length*0.9)], data_y[int(length*0.9)+1 :]



