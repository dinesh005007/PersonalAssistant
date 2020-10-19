import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import random
import json
import pickle
import tensorflow
import tflearn



with open('intents.json',mode='r') as file:
    data=json.load(file)


words = []
labels = []
docs_x = []
docs_y = []
check = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        check.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
    if intent["tag"] not in labels:
        labels.append(intent["tag"])


stemmer = LancasterStemmer()
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)
    

    
training = numpy.array(training)
output = numpy.array(output)

training = numpy.array(training)
output = numpy.array(output)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("./model.tfl")




def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)



def dup(inp,check):  
    value=0
    inp=nltk.word_tokenize(inp)
    inp= numpy.array(inp)
    inp= numpy.char.lower(inp)
    check=numpy.array(check)
    check=numpy.char.lower(check)
    for i in inp:
        if i in check:
                return 1
        return 0

def chat():
            inp = input("You: ")
            res=dup(inp,check)
            if res== 0:
                print("Oops!!...there must be TYPO. Say it again.")
            else:
                    results = model.predict([bag_of_words(inp, words)])
                    #print(results)
                    results_index = numpy.argmax(results)
                    #print(results_index)
            
                    tag = labels[results_index]
            
                    for tg in data["intents"]:
                        if tg['tag'] == tag:
                            responses = tg['responses']
            
                    print(random.choice(responses))


chat()





