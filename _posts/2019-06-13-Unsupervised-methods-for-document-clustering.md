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
<h1 class="title">Multi-Label Document Classification using Neural Networks</h1>
<h2 class="author">Abhijeet Ghawade</h2>
<h3 class="date">March 2019</h3>
</div>
<h1 id="introduction">Introduction</h1>
<p> Document classification is an important problem in computer science, the task is to classify a document into one or more categories. Text documents classification is a classical problem in Natural langauge processing.<br/> We will be working with content based classification in this post. There are multiple application to document classification, some of them being spam-filtering, sentiment analysis, readability assessment. </p>
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
Next up, the training and test tfidf vectors are saved in '.npy' file format for later use.
</p>




<h2 id="code">Data preprocessing</h2>
<pre><code>

import numpy as np 
from os.path import expanduser
from collections import defaultdict
from nltk.corpus import reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
 
stop_words= stopwords.words('english')

#Tokenization is the process of chopping something into pieces, called tokens 
#this includes sometimes throwing away the punctuation   #Stanford NLP group 
def tokenize(text):
	min_length =3 
	words = map(lambda word: word.lower(), word_tokenize(text))
	words = [word for word in words if word not in stop_words]

	tokens= (list(map(lambda token: PorterStemmer().stem(token), words)))
	p= re.compile('[a-zA-Z]+')
	filtered_tokens= list(filter(lambda token: p.match(token) and len(token)>=min_length, tokens))

	return filtered_tokens


#def collection_stats(): 
#List of documents
documents=reuters.fileids()
print ((str(len(documents))), 'documents')

#train_docs=list(filter(lambda doc: doc.startwith('train')))
train_docs_id = list(filter(lambda doc: doc.startswith("train"),documents))
print len(train_docs_id), 'number of train documents'
test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))
print len(test_docs_id), 'number of test docs'

train_docs= [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs= [reuters.raw(doc_id) for doc_id in test_docs_id]

#Tokenization 
vectorizer =TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize)
vectorised_train_documents = vectorizer.fit_transform(train_docs)
vectorised_test_documents = vectorizer.transform(test_docs)

dictionary=vectorizer.get_feature_names()
print (vectorised_train_documents)


nparray=vectorised_train_documents.toarray()
nparray_test=vectorised_test_documents.toarray()

np.save('vectorized_train.npy',nparray)
np.save('vectorized_test.npy', nparray_test)
np.save('train_labels.npy', train_labels)
np.save('test_labels.npy', test_docs_labels)

</code></pre>

<h1 id="Building the Neural Network">Building the Neural Network</h1>
<p>
The next part of the process is building a neural network, which will use the numpy arrays we have saved in the previous code. The first part is to load the train_vectorized, test_vectorized arrays.   <br/>
Then we define the dimensions for the network,<br/> 
<b>The input dimension is 20682</b>, which is equal to the first dimension of the tfidf vectors and also the <b>dictionary size</b>. <br/>
H1, H2 are the sizes for the hidden layers, which are both equal to 10,000. <b>The output layers is of size 90, which is equal to the number of classes</b> in the reuters dataset. <br/>
<hr>
Next, we build the Neural_Network class using <b>pytorch's nn.module function</b>. it consists of a forward function, which calculates the output after passing it though the 3 layers. The output is returned from the function. <br/>
We also need to convert all the data in numpy format to torch tensors as this is the only acceptable data type for torch functions. 
We then use a training loop where we use adam optimizer for training the model for around <b>200 epochs</b>. 
You can use any tyoe of loss function, in this code I have used MSE loss for torchs.nn <br/>
You can save the model for later evaluation. <br/>
<hr>
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
