from operator import le
import nltk 
import re
from nltk.util import pr 
import numpy as np 
import heapq
import sys
import time
from sklearn_som.som import SOM
import sklearn.metrics as sklearn
import random
from numpy import linalg as LA
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class som:
  def __init__(self) -> None:
    with open('bbc-text.csv') as f:
      content = [line.strip().split(',') for line in f] 

    content = content[1:]
    with open('stopwords.txt') as f:
      for line in f:
        stopWord = line.strip().split(' ') 
    self.s = 5
    self.preprocess(content, stopWord)
  
  def preprocess(self, content, stopWord):
    #Convert text to lower case. Remove all non-word characters. Remove all punctuations.
    for item in content:
      dataset = nltk.sent_tokenize(item[1]) 
      for i in range(len(dataset)): 
        dataset[i] = dataset[i].lower() 
        dataset[i] = re.sub(r'\W', ' ', dataset[i]) 
        dataset[i] = re.sub(r'\s+', ' ', dataset[i]) 
      item[1] = dataset
    X = []
    Y = []
    word2count = {} 
    docIndex = 0
    for item in content:
      dataset = item[1]
      vector = []
      for data in dataset: 
        words = nltk.word_tokenize(data) 
        for word in words:
          if word in stopWord or len(word) <= 2:
            continue
          vector.append(word)
          if word not in word2count.keys(): 
            word2count[word] = np.array([docIndex])
          else: 
            if docIndex not in word2count[word]:
              np.append(word2count[word],docIndex)
      docIndex += 1

      X.append(np.array(vector))
      Y.append(item[0])

    N = len(word2count)
    X = np.array(X)
    Y = np.array(Y)

    vsmList = np.zeros((len(X), N))
    print(vsmList.shape)

    for i in range(len(X)):
      doc = X[i]
      unique, counts = np.unique(doc, return_counts=True)
      diction = dict(zip(unique, counts))
      j = 0
      for key, elem in word2count.items():
        if key in diction.keys():
          tF = diction[key] / len(doc)
        else:
          tF = 0
        dF = len(elem)
        vij = np.log10(1+tF) * np.log10(N/dF)
        vsmList[i][j] = vij
        j += 1
    np.savetxt('vsm-np.txt', vsmList)
    self.vsm = vsmList
    self.Y = Y
    self.learn(vsmList, self.s)

  def learn(self, vsmList, s):
    weights = np.random.uniform(low=0, high=0.00001, size=(s, s, 28965))
    largestChange = 100
    epoc = 0
    while abs(largestChange) > 0.02 and epoc < 5:
      etha0 = 0.1
      T2 = 1000
      etha = etha0 * np.exp((-1*epoc)/T2)
      sigma0 = 100
      T1 = 1000/(np.log10(sigma0))
      sigma = sigma0 * np.exp((-1*epoc)/T1)
      epoc += 1
      largestChange = 0
      datas = list(range(0, len(vsmList)))
      while len(datas) > 0:
        r = random.choice(datas)
        datas.remove(r)
        x = vsmList[r]
        temp = x - weights[0][0]
        min1 = LA.norm(temp)
        k = (0,0)
        for j in range(0, s):
          for z in range(0, s):
            d = LA.norm(x - weights[j][z])
            if d < min1:
              min1 = d
              k = (j,z)
        #update:
        for j1 in range(s):
          for j2 in range(s):
            # djk = LA.norm([k[0], k[1]]-[j1, j2])
            djk = ((k[0]-j1)**2 + (k[1]-j2)**2)**0.5
            hKDjk = np.exp((-1*djk**2)/2*sigma**2)
            deltaW = etha * hKDjk * (x - weights[j1][j2])
            weights[j1][j2] += deltaW
            temp2 = LA.norm(deltaW)
            if temp2 > largestChange:
              largestChange = temp2
      #   if sag < 2:
      #     print("min1", min1, "k", k, "y", self.Y[r], "norm dw", temp2)
      
      # if epoc % 5 == 0:
      #   print(epoc, "largestChange", largestChange)
      # largestChange = 0 #test
    self.weights = weights
    # print("epoc", epoc)

  def predict(self, features, s):
    weights = self.weights
    min1 = LA.norm(features - weights[0][0])
    k1 = 0
    k2 = 0
    for j1 in range(s):
      for j2 in range(s):
        d = LA.norm(features - weights[j1][j2])
        if d < min1:
          min1 = d
          k1 = j1
          k2 = j2
    return (k1,k2)

  def cal_confusion(self, datas):
    clusters = [[], [], [], [], []]
    for i in range(len(datas)):
      y_pred = self.predict(datas[i])
      clusters[y_pred].append(self.Y[i])
    vote = []
    ucs = []
    for j in range(5):
      u, c = np.unique(np.array(clusters[j]), return_counts=True)
      majarotyInd = np.where(c == max(c))
      v = u[majarotyInd[0]][0]
      vote.append(v)
      ucs.append((u.tolist(),c.tolist()))
    confusion = []
    for v in vote:
      temp = []
      for (u,c) in ucs:
        if v in u:
          ind = u.index(v)
          temp.append(c[ind])
        else:
          temp.append(0)
      confusion.append(temp)
    print(confusion)

    df_cm = pd.DataFrame(confusion, index = [i for i in vote], columns = [i for i in vote])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.show()

  def hit_plot(self, datas, s):
    hit = []
    vote = []
    for j1 in range(s):
      t = []
      t2 = []
      for j2 in range(s):
        t.append([])
        t2.append([])
      hit.append(t)
      vote.append(t2)
    hit1 = []
    hit2 = []
    distance = 0
    for i in range(len(datas)):
      y_pred1, y_pred2 = self.predict(datas[i], self.s)
      distance += LA.norm(self.weights[y_pred1][y_pred2] - datas[i])
      hit[y_pred1][y_pred2].append(self.Y[i]) 
      hit1.append(y_pred1)
      hit2.append(y_pred2)

    for j1 in range(s):
      for j2 in range(s):
        u, c = np.unique(np.array(hit[j1][j2]), return_counts=True)
        majarotyInd = np.where(c == max(c))
        v = u[majarotyInd[0]][0]
        vote[j1][j2] = v

    print("matrix", vote)
    print("all distances", distance)
    fig = plt.subplots(figsize =(10, 7))
    # Creating plot
    plt.hist2d(hit1, hit2)
    plt.title("Simple 2D Histogram")
    plt.colorbar()
    # show plt
    plt.show()

start = time.time()
som1 = som()
som1.hit_plot(som1.vsm, som1.s)
end = time.time()
print(f"Runtime of the program is {end - start}")
