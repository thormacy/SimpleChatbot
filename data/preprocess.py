# coding: utf-8
"""
    Created on 2018-7-9
    @author: shengwan
    
    预处理QA对
    Input: output.txt 一个中文QA对数据集。格式为一行Q，一行A
    Output: data.pkl data是一个字典，包含三个key：word2id，id2word和trainingSamples
    （trainingSamples已经转化为向量，格式为[[Q1,A1],[Q2,A2],[Q3,A3],...]，其中Q和A均为转化为向量的list，
      eg. Q=[12, 23, 34]）
"""
import pickle
import jieba
import re

data_path = 'output.txt'
output_name = 'data.pkl'

#建立字典
def get_vocabulary(data):
    #先用结巴进行分词
    cut_list = []
    for line in data:
        cut_list.append(' '.join(jieba.cut(line, cut_all = False)))
    
    #去除特殊字符
    for i in range(len(cut_list)):
        cut_list[i] = re.sub("[\*\/\\\、\？\（\）\%\(\)]", "",cut_list[i]).strip()
        cut_list[i] = cut_list[i] + ' '
    
    #建立字典
    vocabulary = {}
    tmp = ''
    for i, line in enumerate(cut_list):
        for j, word in enumerate(line):
            if word != ' ':
                tmp = tmp + word
            else:
                if tmp in vocabulary:
                    vocabulary[tmp] += 1
                else:
                    vocabulary[tmp] = 1
                tmp = ''
    
    word2id = {}
    id2word = {}
    #特殊字符
    word2id['<pad>'] = 0
    word2id['<go>'] = 1
    word2id['<eos>'] = 2
    word2id['<unknown>'] = 3
    id2word[0] = '<pad>'
    id2word[1] = '<go>'
    id2word[2] = '<eos>'
    id2word[3] = '<unknown>'
    
    #按value值给字典排序，写入需要的两个字典word2id，id2word
    for i, vocab in enumerate(sorted(vocabulary.items(), key=lambda item:item[1], reverse=True)):
        word2id[vocab[0]] = i+4
        id2word[i+4] = vocab[0]
    
    return word2id, id2word

#将QA对转化为向量
def word2vector(cut_list, word2id):
    vector = []
    for line in cut_list:
        tmp_line = []
        line_vector = ''
        for split_word in line.strip().split(' '):
            tmp_line.append(str(word2id[split_word]))
        line_vector = ' '.join(tmp_line)
        vector.append(line_vector)
    return vector

#改变为[[Q,A]], 其中Q，A也是一个由id组成的list
def reformat_vector(vector):
    trainingSamples = []
    for i, line in enumerate(vector):
        if i%2 == 0:
            tmp = []
            tmp.append(line.split())
        else:
            tmp.append(line.split())
            trainingSamples.append(tmp)
    for i in range(len(trainingSamples)):
        for j in range(len(trainingSamples[0])):
            trainingSamples[i][j] = [int(x) for x in trainingSamples[i][j]]
    return trainingSamples

#主函数
data = []
with open(data_path,'r') as f:
    for line in f:
        data.append(line)

word2id, id2word, cut_list = get_vocabulary(data)
vector = word2vector(cut_list, word2id)
trainingSamples = reformat_vector(vector)

#save to pkl format
data = {}
data['word2id'] = word2id
data['id2word'] = id2word
data['trainingSamples'] = trainingSamples
with open(output_name, 'wb') as f:
    pickle.dump(data,f)

