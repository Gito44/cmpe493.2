import os
import json
import pickle
import math


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
class_occ_dir=os.path.join(base_directory,"class_occurence.json")
with open(test_dir,"rb") as t:
    test_set=pickle.load(t)
with open(dev_dir,"rb") as d:
    dev_set=pickle.load(d)
with open(prior_dir, "r", encoding="iso-8859-1") as p:
    prior_prob=json.load(p)
with open(vocab_dir, "r", encoding="iso-8859-1") as v:
    vocab_list=json.load(v)
with open(class_occ_dir, "r", encoding="iso-8859-1") as co:
    class_occ=json.load(co)

classes={}
for item in prior_prob:
    class_dir = os.path.join(base_directory, f"{item}bernoulli_occurrence.json")
    with open(class_dir, "r", encoding="iso-8859-1") as c:
        classes[f"{item}"] = json.load(c)


def normalize_dict(dictionary):
    values = dictionary.values()
    min_val = min(values)
    max_val = max(values)

    for key in dictionary:
        dictionary[key] = (dictionary[key] - min_val) / (max_val - min_val)

    return dictionary


def calc_feature_prob(topic,alpha):
    prob_for_class={}
    occlist=classes[topic]
    for word in vocab_list:
        wordneg= word + "-"
        if word in occlist.keys():
            prob_for_class[word] = math.log10((occlist[word]+alpha)/(class_occ[topic]+(2*alpha)))
            prob_for_class[wordneg] = math.log10(1 - ((occlist[word]+(alpha)) / (class_occ[topic] + (2 * alpha))))
        else:
            prob_for_class[word] = math.log10((alpha)/(class_occ[topic]+(2*alpha)))
            prob_for_class[wordneg] = math.log10(1-((alpha) / (class_occ[topic] + (2 * alpha))))
    return prob_for_class



def calculate_likelihood(dictionary,new):
    result=0
    setw=set((str(new.title)+str(new.body)).split())
    for word in vocab_list:
        wordneg=word+"-"
        if word in setw:
            result+= dictionary[word]
        else:
            result+= dictionary[wordneg]
    return result


def berno(dictionary,new):
    fin_pro={}
    for item in classes:
        fin_pro[item]=calculate_likelihood(dictionary[item],new)
    return fin_pro


def idci(set,dictionary):
    id_pro={}
    for new in set:
        id_pro[new.id]=argmax(berno(dictionary,new))
    return id_pro


def argmax(data):
    probs=normalize_dict(data)
    result=[]
    for item in probs:
        if probs[item] >0.7:
            result.append(item)
    return result


def micro_f1_values_calculator(dictionary,set):
    tp=0
    fp=0
    fn=0
    for new in set:
        result=dictionary[new.id]
        for topic in classes:
            if topic in result:
                if topic in new.topic:
                    tp+=1
                else:
                    fp+=1
            elif topic in new.topic:
                fn+=1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))
    return precision,recall,f1

def macro_f1_values_calculator(dictionary,set):
    count = 0
    total = 0
    totalp = 0
    totalr = 0
    for item in classes:
        count += 1
        tp = 0
        fp = 0
        fn = 0
        for news in set:
            result = dictionary[news.id]
            if item in result:
                if item in news.topic:
                    tp+=1
                else:
                    fp+=1
            elif item in news.topic:
                    fn += 1
        precision = tp / (tp + fp)
        totalp += precision
        recall = tp / (tp + fn)
        totalr += recall
        f1 = 2 * ((precision * recall) / (precision + recall))
        total += f1
    return totalp / count, totalr / count, total / count



prob_dict={}
alpha=0.1
for i in classes:
    prob_dict[i]=calc_feature_prob(i,alpha)

dictfin = idci(test_set,prob_dict)

print(macro_f1_values_calculator(dictfin,test_set))
print(micro_f1_values_calculator(dictfin,test_set))