import gensim
from ast import literal_eval as make_tuple
from random import shuffle
import csv
import re
import random
import sklearn.metrics
import pandas as pd
import numpy
import scipy.sparse as sps
from sklearn.metrics import average_precision_score

#By Ke Xu
#3/13/2017


#create a (paper, [authors]) list
def name_dic():
    f_data = open('PaperAuthor.csv')
    list_data = csv.reader(f_data)
    list_data.next()
    pa_list = {};
    for row in list_data:
        PID = row[0]
        AID = row[1]
        A_NAME = row[2]
        pair = (AID,A_NAME)
        if PID not in pa_list:
            pa_list[PID] = list()
        pa_list[PID].append(pair)
    return pa_list

#feature_1: check if author name in name_list
def name_match(a_id,p_id, pa_list):
    name_list = pa_list[p_id]
    for tup in name_list:
        if a_id in tup:
            return True
    return False
#feature 2: Topic model
def loss(AID,PID,a_cxt,p_cxt,model):
	count = 0;
	total = 0;
	#print AID + ' ' + PID 
	#print a_cxt 
	#print p_cxt
	#print '\n'
	for word_a in a_cxt[AID]:
		for word_p in p_cxt[PID]:
			if (word_a in model.vocab) and (word_p in model.vocab):
				count += 1
				total += model.similarity(word_a,word_p)
	if count == 0:
		return 0;
	return total/count;

def loss2(AID,PID,a_cxt,p_cxt,model):
	big = 0;
	for word_a in a_cxt[AID]:
		for word_p in p_cxt[PID]:
			if (word_a in model.vocab) and (word_p in model.vocab):
				val = model.similarity(word_a,word_p)
				if val > big:
					big = val
	return big;
'''
#feature_3: co-author similarity
def coa_match(a_id,p_id,pa_list,coa_dic):
    name_list = pa_list[p_id]
    co_list = [pair for pair in name_list if a_id not in pair]
    print co_list
    inter = 0.0;
    for person in co_list:
	coa_id = int(person[0])
	#print person[0]
	if coa_dic[int(a_id),coa_id]==1:
		print 'yeah'		
		inter += 1;
    return inter/len(co_list);
'''

def score(AID,PID,model,a_cxt,p_cxt,pa_list):
        if name_match(AID,PID,pa_list) == False:
                return 0
        elif (bool(a_cxt)==False) or (bool(p_cxt)==False):
                return 0
        else:
		#print coa_match(AID,PID,pa_list,coa_dic)
                #return coa_match(AID,PID,pa_list,coa_dic)
		return loss2(AID,PID,a_cxt,p_cxt,model)
		#+loss(AID,PID,a_cxt,p_cxt,model)
		

def set_parameters(a_cxt,p_cxt,pa_list):
        raw_paper_data = [line.strip() for line in open("paper_topic_pair.txt", 'r')]
        for item in raw_paper_data:
                temp = make_tuple(item)
                p_cxt[temp[0]] = temp[1]

        raw_author_data = [line.strip() for line in open("author_topic_pair.txt", 'r')]
        for item in raw_author_data:
                temp = make_tuple(item)
                a_cxt[temp[0]] = temp[2]
'''
def coa_stat(pa_list):
	size_row = 2293831
	size_col = 2293831

	a = sps.lil_matrix((size_row, size_col), dtype = int)
	for key,val in pa_list.iteritems():
		x = val
		co_authID = []
		for j in range(len(x)):
			co_authID.append(int(x[j][0]))
    	

	row = numpy.array(co_authID)
   	col = numpy.array(co_authID)
	for k in range(len(row)):
		m = row[k]
  		a[m, col] += numpy.ones((1, col.size))
  	
	return a
'''
# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True) 
p_cxt = {}
a_cxt = {}
pa_list = name_dic();
set_parameters(a_cxt,p_cxt,pa_list)
#coa_dic = coa_stat(pa_list)

#Chen Shao: ground truth
data = pd.read_csv('ValidSolution.csv', header=None)
data = data.T

data.colums = ['authorID', 'paperConfirmed', 'paperDeleted']

size_row = 2293831
size_col = 2259022

a = sps.lil_matrix((size_row, size_col), dtype = int)

num_col, num_samples = data.shape

for i in range(1,num_samples):
    temp = map(int, data[i][1].split(' '))
    row = numpy.array(data[i][0])
    col = numpy.array(temp)
    a[row, col] += numpy.ones((row.size, col.size))

#MAP
lines = numpy.loadtxt('Valid.csv',dtype=str,delimiter=',', skiprows=1)
#print(lines)
#print(len(lines))

avg_mean = []
AP = []
for line in lines:
	ground_data = []
	score_data = []
	author_id = int(line[0]);
	paper_id_list = line[1].split(' ')
	paper_id_list = [int(i) for i in paper_id_list]
	#print(paper_id_list[0])
	for paper_id in paper_id_list:
	    ground_data.append(a[author_id, paper_id])
	    score_data.append(score(str(author_id), str(paper_id),model,a_cxt,p_cxt,pa_list))
	
	#avg_score = average_precision_score(ground_data, score_data)
 	#avg_mean.append(avg_score)

#print numpy.mean(avg_mean)c

	sort =  [x for (y,x) in sorted(zip(score_data,paper_id_list), key=lambda pair: pair[0])]
	#print sort
	sort = list(reversed(sort))
    	#print list(reversed(sort))
	#print sort

	pos = 1;
	collection = 0.0;
	summ = 0.0;
	for PID in sort:
		truth = a[author_id,PID]
		if truth == 1:
			collection += 1;
			summ += (collection/pos)
		pos += 1;
	#print summ 
	#print '\n'
	ap = summ/collection
	AP.append(ap)

print 'score of model'
print numpy.mean(AP)

lines = numpy.loadtxt('randomBenchmark.csv',dtype=str,delimiter=',', skiprows=1)
#print(lines)
#print(len(lines))

avg_mean = []
AP = []
for line in lines:
	ground_data = []
	score_data = []
	author_id = int(line[0]);
	paper_id_list = line[1].split(' ')
	paper_id_list = [int(i) for i in paper_id_list]

	pos = 1;
	collection = 0.0;
	summ = 0.0;
	for PID in paper_id_list:
		truth = a[author_id,PID]
		if truth == 1:
			collection += 1;
			summ += (collection/pos)
		pos += 1;
	#print summ 
	#print '\n'
	ap = summ/collection
	AP.append(ap)

print 'random benchmark'
print numpy.mean(AP)

lines = numpy.loadtxt('basicCoauthorBenchmark.csv',dtype=str,delimiter=',', skiprows=1)
#print(lines)
#print(len(lines))

avg_mean = []
AP = []
for line in lines:
	ground_data = []
	score_data = []
	author_id = int(line[0]);
	paper_id_list = line[1].split(' ')
	paper_id_list = [int(i) for i in paper_id_list]

	pos = 1;
	collection = 0.0;
	summ = 0.0;
	for PID in paper_id_list:
		truth = a[author_id,PID]
		if truth == 1:
			collection += 1;
			summ += (collection/pos)
		pos += 1;
	#print summ 
	#print '\n'
	ap = summ/collection
	AP.append(ap)

print 'Coauthor benchmark'
print numpy.mean(AP)

