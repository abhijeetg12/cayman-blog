--- 
layout: post 
title: Trying Latex
categories: [Catagory 1, Catagory 2] 
tags: [Tag 1, Tag 2] 
comments: true 
---
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
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
<p>This is a detailed explaination for implementing PPO (Proximal Policy Optimization ) paper, we will go through the maths and the code simultaneously to understand what is required to implement upcoming papers in Deep-RL specifiaclly.<br />
I have used PyTorch for implementing the Neural Network, and Python.<br />
This is hardcoded for atari environments, but with minimal changes, you can use it for any other environment required I will follow a causal approach for the code, I hope which will be best for understanding the implementation best.<br />
We will run the code as the Main program, or else, it can be used as a Module to be imported.</p>
<pre><code>if __name__ == &quot;__main__&quot;:
    m=Main()
    m.run_training_loop()
    
    m.record_video()
    m.destroy()</code></pre>
<h1 id="class-main">class Main()</h1>
<p>We first run the class Main, beginning with it’s initializations,</p>
<pre><code>class Main(object):
    def __init__(self):

        self.gamma=0.99
        self.lamda=0.95
        self.updates=15000
        self.epochs= 4 
        self.n_workers= 8
        self.worker_steps=128
        self.n_mini_batch=4
        self.batch_size=self.n_workers*self.worker_steps  #[8*128=1024]
        self.mini_batch_size=self.batch_size // self.n_mini_batch  #[1024/4=256]
        np.random.seed(1)
        torch.manual_seed(1)


        assert(self.batch_size%self.n_mini_batch==0)
        random_worker_seed=12
        self.workers=[Worker(random_worker_seed+i) for i in range(self.n_workers)]
        self.obs= np.zeros((self.n_workers,84,84,4),dtype=np.uint8)

        for worker in self.workers:
            worker.child.send((&quot;reset&quot;, None))

        for i, worker in enumerate(self.workers):
            self.obs[i]=worker.child.recv()

        self.model=Model()
        self.trainer=Trainer(self.model)</code></pre>
<p>In this class, we have set the values of constants like <span class="math inline">\(\lambda\)</span>, <span class="math inline">\(\gamma\)</span>, the updates, epochs, number of workers, mini batches, etc.<br />
The significance of <span class="math inline">\(\lambda\)</span>, <span class="math inline">\(\gamma\)</span> will be apprent while deriving the advantage function.<br />
We are using multiprocessing to generate independent trajectories, and the updates are going to be made in mini batches, which will be optimized for certain numberof epochs.<br />
We later set random seeds for both Numpy and Torch libraries, this is essential if we want to reproduce our results, as there can be lot of variance with Deep-RL results.<br />
We go ahead and initialize 8 workers which will be used to generate 8 independent trajectories. obs is a tensor of size <span class="math inline">\([n-workers, 84, 84, 4]\)</span>.<br />
The size of frame in 84*84, and there are 4 such frames we will obtain at one time.<br />
</p>
<h1 id="class-model">class Model()</h1>
<p>And finally we initialise the Model() of out Neural network. The model consists of 3 convolutional layers, and two fully connected layer. The models predicts the Policy(<span class="math inline">\(\pi\)</span>) and the Values function of the state.</p>
<pre><code>class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv1 =nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0)
        nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))

        self.conv2 =nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))

        self.conv3 =nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))      

        self.lin = nn.Linear(in_features=7*7*64, out_features=512)

        nn.init.orthogonal_(self.lin.weight, np.sqrt(2))
        #last layer lhas 512 features in total 

        self.pi_logits = nn.Linear(in_features=512, out_features=4)

        nn.init.orthogonal_(self.pi_logits.weight, np.sqrt(0.01))

        self.value = nn.Linear(in_features=512, out_features=1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, obs):
        
        h=F.relu(self.conv1(obs))
        h=F.relu(self.conv2(h))
        h=F.relu(self.conv3(h))
        h=h.reshape((-1, 7*7*64))
        
        h=F.relu(self.lin(h))
        pi=Categorical(logits=self.pi_logits(h))
        value= self.value(h).reshape(-1)

        return pi, value

    def obs_to_torch(obs):

        obs=np.swapaxes(obs, 1,3)
        obs=np.swapaxes(obs, 3,2)

        return torch.tensor(obs, dtype=torch.float32,device=device)/255</code></pre>
<p><img src="Network" alt="image" /></p>
</body>
</html>
