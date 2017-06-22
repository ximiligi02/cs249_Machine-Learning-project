import csv
import re

#Author: Ke Xu
#3/8/2017

def codeInt_paper(x):
    int_string = str(x)
    result = '';
    for num in int_string:
        #print num
        num = ord(num)-48+1
        #print chr(num+96)
        result += chr(num+96)
    return result;

def codeInt_author(x):
    int_string = str(x)
    result = '';
    for num in int_string:
        #print num
        num = ord(num)-48+1+10
        #print chr(num+96)
        result += chr(num+96)
    return result;

def readPaper():
    f_paper = open('Paper.csv')
    paper_data = csv.reader(f_paper)
    paper_data.next()
    #for row in paper_data:
    #   print row
    return paper_data;

def readAuthor():
    f_author = open('Author.csv')
    author_data = csv.reader(f_author)
    author_data.next()
    return author_data;

def getWords(text):
    return re.compile('[a-zA-Z]+').findall(text)

def processPaper(dic):
    p_data = readPaper();
    papers = [];
    i = 1;
    for row in p_data:
	#print i;
	i += 1;        
	content = list(set(map(lambda x:x.lower(),getWords(row[1]))+map(lambda x:x.lower(),getWords(row[5]))));
        record = (row[0],codeInt_paper(row[0]),content);
        papers.append(record);
        dic[row[0]] = content;
    #for item in papers:
    #   print item
    #   print "\n"
    return papers;

def processAuthor(dic,a_dic):
    #form an author list
    a_data = readAuthor();
    authors = [];
    
    for row in a_data:
        temp = row[1];
        record = (row[0],codeInt_author(row[0]),temp);
        authors.append(record);

    #Add context from training data
    t_data = open('Train.csv')
    training_data = csv.reader(t_data)
    t_data.next()
    for row in training_data:
	pos_papers = row[1].split();
	context = map(lambda x:dic[x], pos_papers)
	context = list(set([item for sublist in context for item in sublist]))	
	a_dic[row[0]] = context;
	#print a_dic[row[0]]
	#print "\n"
    
    #link author list with context
    complete_list = [];
    for item in authors:
	if item[0] in a_dic:	
		item = (item[0],item[1],item[2],a_dic[item[0]])
		complete_list.append(item)
		#print item
		#print len(item)
	else:
		item = (item[0],item[1],item[2],["NAW"])
		complete_list.append(item)
	
    #for item in complete_list:
    #   print item
    #   print "\n"
	
    return complete_list;

# Main
dic = {};
a_dic = {};
papers = processPaper(dic);
authors = processAuthor(dic,a_dic);

#Generate training data as (target,context) pairs 
'''
 papers: (num_id,code_id,context*)
 authors: (num_id,code_id,name,context*)
'''
paper_topic = [];
author_topic = [];
func_word = ['a','an','the','for','of','by','and','to','in','on'];

for record in papers:
	for word in record[2]:
		if word in func_word:		
			record[2].remove(word);
	pair = (record[0],record[2])
	paper_topic.append(pair)

for record in authors:
	if len(record) == 4:
		for word in record[3]: 
			if word in func_word:
				record[3].remove(word);
		pair = (record[0],record[2],record[3])
		author_topic.append(pair);

paper_file = open('paper_topic_pair.txt','w');
for item in paper_topic:
  print>>paper_file, item

author_file = open('author_topic_pair.txt','w');
for item in author_topic:
  print>>author_file, item

print 'done, you can now run score.py'
