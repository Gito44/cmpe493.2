import os
import json
import pickle
import math
import random


class News:
    def __init__(self, title, body, i,topic,istrain):
        self.title=title
        self.body=body
        self.id=i
        self.topic=topic
        self.istrain=istrain

base_directory= "D:/kod/cmpe493/probability data"
test_dir=os.path.join(base_directory,"test_set")
prior_dir=os.path.join(base_directory,"prior_prob.json")
vocab_dir=os.path.join(base_directory,"vocab.json")
dev_dir=os.path.join(base_directory,"dev_set")
with open(test_dir,"rb") as t:
    test_set=pickle.load(t)
with open(dev_dir,"rb") as d:
    dev_set=pickle.load(d)
with open(prior_dir, "r", encoding="iso-8859-1") as p:
    prior_prob=json.load(p)
with open(vocab_dir, "r", encoding="iso-8859-1") as v:
    vocab_list=json.load(v)

classes={}
for item in prior_prob:
    class_dir = os.path.join(base_directory, f"{item}occurrence.json")
    with open(class_dir, "r", encoding="iso-8859-1") as c:
        classes[f"{item}"] = json.load(c)


def normalize_dict(dictionary):
    values = dictionary.values()
    min_val = min(values)
    max_val = max(values)

    for key in dictionary:
        dictionary[key] = (dictionary[key] - min_val) / (max_val - min_val)

    return dictionary


def calc_feature_prob(word,dict,alpha,vocab):
    try:
        return(dict[word]+alpha)/(dict["total_words"]+vocab)

    except:
        return alpha/(dict["total_words"]+vocab)

def calc_likelihood(topic,news,dict,alpha,vocab):
    result=0
    for word in (str(news.title)+str(news.body)).split():
        result+= math.log10(calc_feature_prob(word,dict,alpha,vocab))
    result+=math.log10(prior_prob[topic])
    return result

def mnb(alpha,news,vocab):
    final_probs={}
    for item in classes:
        final_probs[item]= calc_likelihood(item,news,classes[item],alpha,vocab)
    return final_probs


def argmax(data):
    probs=normalize_dict(data)
    result=[]
    for item in probs:
        if probs[item] >0.9:
            result.append(item)
    return result



def macro_f1_values_calculator(alpha,list,set):
    count = 0
    total = 0
    totalp=0
    totalr=0
    for item in list:
        count += 1
        tp = 0
        fp = 0
        fn = 0
        for news in set:
            result = argmax(mnb(alpha, news, len(vocab_list)))
            if item in result:
                if item in news.topic:
                    tp+=1
                else:
                    fp+=1
            elif item in news.topic:
                    fn += 1
        precision = tp / (tp + fp)
        totalp+=precision
        recall = tp / (tp + fn)
        totalr+=recall
        f1 = 2 * ((precision * recall) / (precision + recall))
        total += f1
    return totalp/count, totalr/count, total / count


def micro_f1_values_calculator(alpha,list,set):
    tp = 0
    fp = 0
    fn = 0

    for news in set:
        result=argmax(mnb(alpha,news,len(vocab_list)))
        for topic in list:
            if topic in result:
                if topic in news.topic:
                        tp+=1
                else:
                    fp+=1
            elif topic in news.topic:
                fn+=1
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1= 2* ((precision*recall)/(precision+recall))
    return precision,recall,f1


alpha=2.6


print(macro_f1_values_calculator(alpha,classes,test_set))
print(micro_f1_values_calculator(alpha,classes,test_set))
