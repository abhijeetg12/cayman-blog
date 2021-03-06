<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  <meta http-equiv="Content-Style-Type" content="text/css" />
  <meta name="generator" content="pandoc" />
  <meta name="author" content="Abhijeet Ghawade" />
  <title>Implementing PPO in Pytorch</title>
  <style type="text/css">code{white-space: pre;}</style>
  <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
</head>
<body>
<div id="header">
<h1 class="title">Methods for document clustering, and visualization</h1>
<h2 class="author">Abhijeet Ghawade</h2>
<h3 class="date">March 2019</h3>
</div>
<h1 id="introduction">Introduction</h1>
<p> This is a blog in the continuation of the previous one, "Document classification", have a look for a better understanding.  </p>
  <hr>
<h1 id="Dataset">Dataset</h1>
<p> We will be using the <b>Reuters</b> dataset for the scope of this blog, Reuters is a benchmark dataset for document classification. <br/>
Reuters ApteMod is a collection of <b>10,788</b> documents from the Reuters financial
newswire service, partitioned into a <b>training set with 7769 documents</b>
and a <b>test set with 3019 documents</b>.  The total size of the corpus is
about 43 MB. <br/>
	<hr>
<b>Reuters is a multi-class and multi-label dataset</b>, this means there are multiple classes, and each document can belong to any of these categories making this a multi-label problem. <br/>
<hr>
In the ApteMod corpus, each document belongs to one or more
categories.  There are <b>90 categories</b> in the corpus.  The average
number of categories per document is 1.235, and the average number of
documents per category is about 148, or 1.37% of the corpus.<br/>
</p>

<h1 id="Tokenization and tf-idf">Data Preporcessing</h1>
<p>
The data in the reuters dataset consists of <b>text files</b>, which cannot be understood by the computer. There are multiple ways of representing the text data into numerical data, one of them is the <b>tf-idf vectorization</b>. Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining.<br/> 
	<hr>
This weight is a <b>statistical measure used to evaluate how important a word is to a document in a collection or corpus</b>. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus.<br/> 
Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query.<br/>
<hr>
But before we move on to using tf-idf, we need to tokenize the data. tokenization is the process of chopping up character sequences into pieces called tokens. In this particular problem statement we will use <b>nltk(natural language tool-lit) word tokenizer</b>. We will also be removing some unnecessary words called as stop words from out corpus. <br/>
After the tokenization process is complete, we will <b>TfidVectorizer from sklearn library</b> for vectorization. <br/>
</p>



<h1 id="Autoencoder for data compression">Autoencoder for data compression</h1>
<p>
I have used pytorch for building this Auto-encoder model, which consists of 4 fully connected layers. <br/>
The input to the model is the tf-idf vectors which is of the dimension 20682 per entry. <br/>
	<hr>
The first layer takes in the 20682 dimensional array, and the ouput is 1000 neurons. These 1000 neurons are connected to 30 neurons, which is the expected compression output. The 30 neurons are then connected to 1000 neurons, and these are connected to 20682 neurons in the final layer. <br/>
<hr>
I have used <b>sigmoid</b> as the activation function for all the layers. The final loss to be minimised is the mean squared error between the first and the final layer. The optimization algorithm used for training is the Adam Optimizer, with an initial learning rate of 3e-4. <br/ 
<hr>
The loss function also has a regularaization term along with the MSE loss, with a coefficient 0.001. <br/> 


</p>



<h2 id="code">Data preprocessing</h2>
<pre><code>
import numpy as np 
import pandas as pd 
from os.path import expanduser
from collections import defaultdict
from nltk.corpus import reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.optim as optim

import seaborn as sns
from string import *

from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import pycuda.driver as cuda
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
 
cachedStopWords = stopwords.words("english")
stop_words=cachedStopWords
def tokenize(text):
	min_length = 3
	words = map(lambda word: word.lower(), word_tokenize(text))
	words = [word for word in words if word not in cachedStopWords]
	tokens = (list(map(lambda token: PorterStemmer().stem(token),
	                               words)))
	p = re.compile('[a-zA-Z]+');
	filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length,tokens))
	return filtered_tokens

cuda= torch.cuda.is_available()

documents=reuters.fileids()
train_docs_id = list(filter(lambda doc: doc.startswith("train"),documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"),documents))
 
train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

vectorizer = TfidfVectorizer(stop_words=stop_words,
                             tokenizer=tokenize)
vectorised_train_documents = vectorizer.fit_transform(train_docs)
vectorised_test_documents = vectorizer.transform(test_docs)

train_docs=vectorised_train_documents.toarray()

################### Writing autoencoder model #############################
input1=torch.from_numpy(train_docs)
input1=input1.type(torch.float32)

class Autoencoder(nn.Module): 
	def __init__(self,):
		super(Autoencoder, self).__init__()

		self.fc1=nn.Linear(20682, 1000)
		self.fc2=nn.Linear(1000,30)
		self.fc3=nn.Linear(30,1000)
		self.fc4=nn.Linear(1000,20682)


	def forward(self,x):
		print (type(x))
		h1= F.sigmoid(self.fc1(x))
		print (type(h1))
		h2= F.sigmoid(self.fc2(h1))
		h3= F.sigmoid(self.fc3(h2))
		h4= F.sigmoid(self.fc4(h3))

		return h4, h2

Encoder=Autoencoder()
criterion=nn.MSELoss()
optimizer=optim.Adam(Encoder.parameters(), lr=3e-4)

print (list(Encoder.parameters()))
n_epochs=100
Losses=[]
for epoch in range(n_epochs):
	
	target=input1.clone()
	target.require_grad= False

	output1, encoded=Encoder(input1)
	
	regularization_loss = 0
	for param in Encoder.parameters():
		regularization_loss += torch.sum(torch.abs(param))

	loss=criterion(target, output1)+0.001*regularization_loss

	loss.backward()
	optimizer.step()

	print (loss.item())
	optimizer.zero_grad()
	Losses.append(loss.item())

x=range(len(Losses))
plt.plot(x, Losses)
plt.show()

y=encoded.detach().numpy()
print (np.shape(y))


#y=np.std(np.mean(y, axis=0), axis=0, ddof=1)
#print (y)
y1=np.mean(y, axis=0)
print (np.shape(y1))
y=y-y1

y2=np.std(y, axis=0, ddof=1)
y=y/y2

print (np.shape(y))


np.save('data_reuters.npy', y)
</code></pre>

<h1 id="Data Visualization">Data Visualization</h1>
<p>
We have reduced the number of dimensions of the original data from 20k to 30. But we cannot visualize dimensions more that 3 intuitively. Hence for the next part we will further reduce the 30 dimensions to 3 using PCA first. This will let us analyse the data along the dimensions with maximum variance. But PCA is a linear transformatino of data, and we only visualize the dominant axes in the data. Hence the data may not look as separated into clusters as one would expect.<br/>
<hr> 
There is an interesting solution for that in form of t-SNE, which stands for t-Stochastic Neighbour embeddings. t-SNEs is a great tool for visualizing high dimensional data in lower dimensions. It is a nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions. <br/>
We shall be visualising the data from both PCA and t-SNE in 3 dimensions for the reuters dataset.  <br/>

After this data is converted to 3-dimensions, we can visualise the clusters that are being formed on the orignial data using multiple suitable methods. In this post I am going to be dealing with two clustering algorithms, which are <br/>
<b>K-Means clustering</b> <br/>
<b>Agglomerative clustering</b></br>
<hr>
<img src="https://github.com/abhijeetg12/cayman-blog/blob/gh-pages/_posts/imgs/clusters_from_autoencoder.png?raw=true" alt="image" /><br />
<img src="https://github.com/abhijeetg12/cayman-blog/blob/gh-pages/_posts/imgs/K-Means_clustering.png?raw=true" alt="image" /><br />
<img src="https://github.com/abhijeetg12/cayman-blog/blob/gh-pages/_posts/imgs/AgglomerativeClustering.png" alt="image" /><br />
</p>

<h2 id="code">MLP Code</h2>
<pre><code>
  
N=5000
D_in=20682
H1=10000
H2=10000
D_out=90


class Neural_Network(nn.Module): 

	def __init__(self):
		super(Neural_Network, self).__init__()

		#define and give layers their names
		self.fc1=nn.Linear(D_in, H1)
		self.fc2=nn.Linear(H1, H2)
		self.fc3=nn.Linear(H2, D_out)

	def forward(self,x):

		y1=F.relu(self.fc1(x))
		y2=F.relu(self.fc2(y1))
		y3=F.relu(self.fc3(y2))


		return y3

net=Neural_Network()
print(net)

params=list(net.parameters())
print (len(params))

for i in range(len(params)):
	print (params[i].size())



input1 =torch.from_numpy(Trained)
output1= net(input1)

print (output1.size(), 'output1 size')
print (type(output1))
#this is the output of the first forward pass. 

###The train labels will have to be converted to torch tensors, for calculating the RMSE loss. 
y_train=torch.from_numpy(train_labels).float()
print (type(y_train.data))
loss=nn.BCELoss()
loss_v=loss(output1, y_train)
print (loss_v)

optimizer=torch.optim.Adam(net.parameters(), lr=1e-4)



epochs=200
loss_array=[]
for i in range(epochs):
	output1=net(input1)
	loss_v=loss(output1, y_train)
	optimizer.zero_grad()
	loss_v.backward()
	optimizer.step()
	loss_array.append(loss_v.item())
	if (i %10==0):
		torch.save(net.state_dict(), 'bce_pytorch_model/'+str(i)+'.pth')


	print ("Loss for epoch ", i, "is", loss_v.item())

np.save('loss_train_bce.npy', loss_array)

torch.save(net.state_dict(), 'bce_pytorch_model/bce_final.pth')
</code></pre>

<h1 id="Evaluation">Evaluation of the model</h1>
<p>The main metrics for text classification are <b>precision, recall, F1 </b> <br/>
<b>Precision</b> indicates the number of documents correctly assigned to a particular category out of the total number of documents predicted <br/>
<b>Recall</b> indicates the number of documents correctly assigned to a particular category, out of the total number of documents in that category.  <br/>
<b>F1 Score</b> This category combines the precision and recall, as F1 score is the harmonic mean of the precision and the recall score.  <br/>

But the evaluation in this case is being done in a multi-class and a multi-label environment, hence the evaluation metric become more complicated. 
<hr>
The below are the evaluation metrics for multi-class classification. <br/>

<b>Micro-Average</b>,
<b>Macro-Average</b> <br/>

Let us look at an example at how these measures might work for a toy classification problem, <br/>
Let us say we have a multi-class classification system with four classes, A, B, C, D.  And we are using precision as the preferred metric. <br/>
<hr>
The results for the classification are as the follows, <br/>
1) <b>Class A</b>: 1 True positive and 1 False positive <br/>
1) <b>Class B</b>: 10 True positive and 90 False positive <br/>
1) <b>Class C</b>: 1 True positive and 1 False positive <br/>
1) <b>Class D</b>: 1 True positive and 1 False positive <br/>
We can see that, Pr(A)=Pr(C)=Pr(D)=0.5 and Pr(B)=0.1<br/>
<hr>
The <b>macro-average</b> is computed as <b>Pr_macro</b>=(0.5+0.1+0.5+0.5)/4=0.4, <br/>
The <b>micro-average</b> is computed as <b>Pr_micro</b>=(1+10+1+1)/(2+100+1+1)=0.123 <br/>
 <hr>
 
 Let us look at the results achieved by our model, after training for 200 epochs, <br/>
The micro average quality numbers are as the following ,<br/>
<hr>
<b>Precision: 0.5716</b><br/>
<b>Recall: 0.9675</b><br/>
<b>F1-Score : 0.7186</b><br/>
<hr>
</p>
  
</html>
