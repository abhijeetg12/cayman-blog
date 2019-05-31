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
<h1 class="title">Implementing PPO in Pytorch</h1>
<h2 class="author">Abhijeet Ghawade</h2>
<h3 class="date">March 2019</h3>
</div>
<h1 id="introduction">Introduction</h1>
<p> Document classification is an important problem, and categorizing helps us be more organized</p>
  
<h1 id="Dataset">Dataset</h1>
<p> We will be using the Reuters dataset for the scope of this blog, Reuters is a benchmark dataset for document classification. <br />
Reuters is a multi-class multi-label dataset, this means there are multiple classes, and each document can lie in any of these categories making this a multi-label problem. The dataset has 90 classes, 7769 training documents and 3019 testing documents.<br />  
  By far most documents have either one or two labels, but some have up to 15 <br/>
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
        

<p><img src="https://github.com/abhijeetg12/PPO-PyTorch/blob/master/breakout.png?raw=true" alt="image" /><br />
</p>
</html>
