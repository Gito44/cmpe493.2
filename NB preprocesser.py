import sys
import string
import os
import json
import pickle
import random
path= "D:/kod/cmpe493/reutersdata"
base_directory= "D:/kod/cmpe493"
sys.setrecursionlimit(1500)
punct= string.punctuation
vowels=("a", "e", "i", "u", "o")
stop_words=("a","all","an","and","are","as","be","been","but","by","few","for","have","he","her","here","him","his","how",
            "i","in","is","it","its","many","me","my","none","of","on","or","our","she","some","the","their","them",
            "there","they","that","this","us","was","what","when","which","where","who","why","will","with","you","your")

class News:
    def __init__(self, title, body, i,topic,istrain):
        self.title=title
        self.body=body
        self.id=i
        self.topic=topic
        self.istrain=istrain

def topics_extractor(text,list):
    str1='<D>'
    str2='</D>'
    if str1 and str2 in text:
        topics_extractor(text[text.index(str2)+len(str2):],list)
        list.append(text[text.index(str1)+len(str1):text.index(str2)])
    else:
        return


def text_extractor(text,list):
    str1 = "<REUTERS"
    str2 = "</REUTERS>"
    if str1 and str2 in text:
        text_extractor(text[text.index(str2)+len(str2):],list)
        new=text[text.index(str1)+len(str1):text.index(str2)]
        new = dissector(new)
        list.append(new)
    else:
        return


def dissector(text):
    istrain=0
    if 'LEWISSPLIT="TRAIN"' in text:
        istrain=1
    if '<TEXT TYPE="UNPROC">' in text:
        id = text[text.index('NEWID="') + len('NEWID="'):text.index('">')]
        body = news_tokenizer(text[text.index('<TEXT TYPE="UNPROC">&#2;') + len('<TEXT TYPE="UNPROC">&#2;'):text.index('</TEXT>')])
        topic_text= text[text.index('<TOPICS>') + len('<TOPICS>'):text.index('</TOPICS>')]
        topics= []
        topics_extractor(topic_text,topics)

        return News(None,body,id,topics,istrain)
    elif '<TEXT TYPE="BRIEF">' in text:
        title=news_tokenizer(text[text.index("<TITLE>")+len("<TITLE>"):text.index("</TITLE>")])
        id=text[text.index('NEWID="')+len('NEWID="'):text.index('">')]
        topic_text = text[text.index('<TOPICS>') + len('<TOPICS>'):text.index('</TOPICS>')]
        topics = []
        topics_extractor(topic_text, topics)

        return News(title, None, id,topics,istrain)
    else:
        title = news_tokenizer(text[text.index("<TITLE>") + len("<TITLE>"):text.index("</TITLE>")])
        id = text[text.index('NEWID="') + len('NEWID="'):text.index('">')]
        body= news_tokenizer(text[text.index("<BODY>")+len("<BODY>"):text.index("</BODY>")])
        topic_text = text[text.index('<TOPICS>') + len('<TOPICS>'):text.index('</TOPICS>')]
        topics = []
        topics_extractor(topic_text, topics)

        return News(title,body,id,topics,istrain)


def news_tokenizer(string):
    string= " " + string + " "
    string= string.replace("\n", " ")
    string= string.lower()
    for item in punct:
        string= string.replace(item,"")
    for word in string.split():
        replace= " " + word + " "
        if word in stop_words:
            string=string.replace(replace," ")
        if word.isalpha()==False:
            string=string.replace(replace," ")
        elif word[0:2] == "lt":
            string=string.replace(replace, " ")
        else:
            if word[-4:] == "sses":
                string=string.replace(replace, " "+ word[:-2] +" ")
            elif word[-3:] == "ies":
                string=string.replace(replace," "+ word[:-2] +" ")
            elif word[-3:] == "ing":
                for vowel in vowels:
                    if vowel in word[:-3]:
                        string=string.replace(replace," "+ word[:-3] +" ")
            if word[-1:] == "s" and word[-2:] != "ss":
                string=string.replace(replace," "+ word[:-1] +" ")
    return string

liste=[]


for filename in os.listdir(path):
    with open(os.path.join(path,filename), encoding='iso-8859-1') as f:
        a=f.read()
        text_extractor(a,liste)
        f.close()


occurrence_topics={}
for item in liste:
    for topic in item.topic:
        try:
            occurrence_topics[topic]=occurrence_topics[topic]+1
        except:
            occurrence_topics[topic]=1

sorted_topics=sorted(occurrence_topics.items(), key=lambda x:x[1] , reverse=True)
top10=sorted_topics[:10]
top10topicnames=[]

for item in top10:
    top10topicnames.append(item[0])

training=[]
test=[]
development=[]

for item in liste:
    for topic in item.topic:
        if topic in top10topicnames:
            if item.istrain==1:
                training.append(item)
            else:
                test.append(item)
            break

def calculate_prior(output_folder):
    prior_prob={}
    for item in training:
        for topic in item.topic:
            if topic in top10topicnames:
                try:
                    prior_prob[topic]=prior_prob[topic]+1
                except:
                    prior_prob[topic]=1
    class_occurence_path=os.path.join(output_folder,"class_occurence.json")
    with open(class_occurence_path, 'w', encoding='iso-8859-1') as co:
        json.dump(prior_prob, co, ensure_ascii=False, indent=4)
    for item in prior_prob.keys():
        prior_prob[item]=prior_prob[item]/len(training)
    output_file_path=os.path.join(output_folder,"prior_prob.json")
    with open(output_file_path, 'w', encoding='iso-8859-1') as f:
        json.dump(prior_prob, f, ensure_ascii=False, indent=4)

def likelihood_prob_nb(dict_path,output_folder):
    with open(dict_path, 'r') as f:
        dict = json.load(f)
    vocab=set()
    for item in dict:
        class_likelihood= {}
        for document in training:
            if item in document.topic:
                for word in (str(document.body)+str(document.title)).split():
                    vocab.add(word)
                    try:
                        class_likelihood[word]=class_likelihood[word]+1
                    except:
                        class_likelihood[word]=1
        class_likelihood["total_words"] = 0
        for words in class_likelihood:
            if words != "total_words":
                class_likelihood["total_words"]+=class_likelihood[words]
        output_file_path = os.path.join(output_folder, f"{item}occurrence.json")
        with open(output_file_path, 'w', encoding='iso-8859-1') as f:
            json.dump(class_likelihood, f, ensure_ascii=False, indent=4)

    vocab=list(vocab)
    return vocab

def bernoulli_processor(output_folder):
    for item in top10topicnames:
        bernoulli_occurence={}
        for document in training:
            if item in document.topic:
                for word in set((str(document.body)+str(document.title)).split()):
                    try:
                        bernoulli_occurence[word]=bernoulli_occurence[word]+1
                    except:
                        bernoulli_occurence[word]=1
        output_file_path = os.path.join(output_folder, f"{item}bernoulli_occurrence.json")
        with open(output_file_path, 'w', encoding='iso-8859-1') as f:
            json.dump(bernoulli_occurence, f, ensure_ascii=False, indent=4)


output_folder="probability data"
os.makedirs(output_folder, exist_ok=True)
calculate_prior(output_folder)
prior_path=os.path.join(base_directory,output_folder,"prior_prob.json")
vocab=likelihood_prob_nb(prior_path,output_folder)
bernoulli_processor(output_folder)
test_path=os.path.join(base_directory,output_folder,"test_set")
development_set_path=os.path.join(base_directory,output_folder,"dev_set")
bernoulli_path=os.path.join(base_directory,output_folder,"bernoulli_occ")
vocab_path = os.path.join(base_directory,output_folder, "vocab.json")

with open(vocab_path, "w", encoding="iso-8859-1") as v:
    json.dump(vocab,v,ensure_ascii=False,indent=4)

with open(test_path, "wb") as f:
    pickle.dump(test, f)
with open(development_set_path, "wb") as d:
    pickle.dump(development, d)