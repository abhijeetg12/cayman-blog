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
The significance of <span class="math inline">\(\lambda\)</span>, <span class="math inline">\(\gamma\)</span> will be understood while deriving the advantage function and loss function.<br />
We are using multiprocessing to generate independent trajectories for multiple workers, and the updates are going to be made in mini batches, which will be optimized for certain number of epochs.<br />
We later set random seeds for both Numpy and Torch libraries, this is essential if we want to reproduce our results, as there can be lot of variance with Deep-RL results with change in initialization.<br />
We go ahead and initialize 8 workers which will be used to generate 8 independent trajectories. obs is a tensor of size <span class="math inline">\([nworkers, 84, 84, 4]\)</span>.<br />
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
<p><img src="https://github.com/abhijeetg12/cayman-blog/blob/gh-pages/_posts/Network.png?raw=true" alt="image" /><br />
The first layer is the input, which consists of the raw visual feed of the game, and 4 such frames are stacked on each other.<br />
The next one is a concolutional layer with kernel size=8 and the stride of 4, which reduces the dimension of the image to 20*20. There are 32 such filters.<br />
The next convolutional layer reduces the filter size to 9*9, there are 64 such channels, the next layer reduces the size to 7*7 with the same number of channels.<br />
The last convolutinal layer is flattened to obtain 7*7*64=3136 neurons. This layer is fullly connected to 512 neurons in the next layer.<br />
The 512-neuron layer is then connected to separate outputs corresponding the policy and the value outputs.<br />
</p>
<h1 id="training-loop">Training Loop</h1>
<p>And finally we initialise the Model() of out Neural network. The model consists of 3 convolutional layers, and two fully connected layer. The models predicts the Policy(<span class="math inline">\(\pi\)</span>) and the Values function of the state.</p>
<pre><code>    def run_training_loop(self):

        loss_vec=[]
        rewards_vec=[]

        episode_info=deque(maxlen=100)
        for update in range(self.updates):
            time_start=time.time()
            progress =update/self.updates

            learning_rate=3e-4 *(1-progress)
            clip_range= 0.1*(1-progress)

            samples, samples_episode_info=self.sample()
            
            info_loss=self.train(samples, learning_rate, clip_range)
            
            time_end=time.time()

            fps=int(self.batch_size/(time_end-time_start))

            episode_info.extend(samples_episode_info)
            
            reward_mean, length_mean =Main._get_mean_episode_info(episode_info)
            sampled_rewards= samples[&#39;rewards&#39;].data.numpy()
            rew_total=sampled_rewards.sum()

            loss_err=info_loss[5].data.numpy()
            rew_np=reward_mean#reward_mean.data.numpy()

            print loss_err, &#39;training loss \n&#39;
            loss_vec.append(loss_err)
            rewards_vec.append(rew_total)

            print(&quot;f&quot;,update,&quot;: fps=&quot;,{fps} ,&#39;reward&#39;,{rew_total},&#39; length&#39;,{length_mean}, info_loss)
            if update%10==0 and update!=0:

                np.save(&#39;loss_vec1_breakout&#39;, loss_vec)
                np.save(&#39;reward_vec1_breakout&#39;, rewards_vec)</code></pre>
<p>After having initialized the model, we now begin the training process.<br />
self.updates represents the number of times the sampling procedure, and optimization is going to take place.<br />
Learning rate, is initialized as a function of the progress percentage, this is used to anneal the learning rate. clip range is also varied similarly.<br />
samples is the vector containing the information stored from the function self.sample().</p>
<h2 id="def-sample">def sample():</h2>
<p>This function is used to sample trajectories from the current policy.</p>
<pre><code>    def sample(self):
        
        rewards=np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        actions=np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
        dones = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        obs = np.zeros((self.n_workers, self.worker_steps, 84, 84, 4), dtype=np.uint8)
        neg_log_pis = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        values = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        episode_infos = []

        for t in range(self.worker_steps):
            obs[:,t]=self.obs
            pi,v=self.model(obs_to_torch(self.obs))

            values[:, t] = v.cpu().data.numpy()
            
            a=pi.sample()
            actions[:,t]=a.cpu().data.numpy()
            
            neg_log_pis[:,t]=-pi.log_prob(a).cpu().data.numpy()

            for w,worker in enumerate(self.workers):

                worker.child.send((&quot;step&quot;,actions[w,t]))

            for w, worker in enumerate(self.workers):
                
                self.obs[w], rewards[w,t], dones[w,t], info=worker.child.recv()


                if info:
                    info[&#39;obs&#39;]=obs[w,t,:,:,3]
                    episode_infos.append(info)

        advantages = self._calc_advantages(dones, rewards, values)

        samples={&#39;obs&#39;:obs, &#39;actions&#39;:actions, &#39;values&#39;:values, &#39;neg_log_pis&#39;:neg_log_pis, &#39;advantages&#39;:advantages, &#39;rewards&#39;:rewards}

        samples_flat={}
        for k,v in samples.items():
            v=v.reshape(v.shape[0]*v.shape[1], *v.shape[2:])

            if k==&#39;obs&#39;:
                samples_flat[k]=obs_to_torch(v)
            else:
                samples_flat[k]= torch.tensor(v, device=device)
        return samples_flat, episode_infos</code></pre>
<p>The first part of the sampling process is initializing the tensors for rewards, actions, dones, observations, value functions, and negative log likelihood.<br />
np.zeros is used to initialize the tensors for each of these entities.</p>
<p>Now for a range of worker steps,all of the information will be filled by taking sequential steps, for each worker and filling in the information relevant to each step</p>
<h2 id="calculating-the-advantage">Calculating the Advantage</h2>
<pre><code>    def _calc_advantages(self, dones, rewards, values):

        advantages=np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        last_advantage=0

        _,last_value= self.model(obs_to_torch(self.obs))
        last_value= last_value.cpu().data.numpy()

        for t in reversed(range(self.worker_steps)):
            mask=1.0-dones[:,t]
            last_value =last_value*mask
            last_advantage=last_advantage*mask

            delta =rewards[:,t]+self.gamma*last_value -values[:,t]
            last_advantage=delta+self.gamma*self.lamda*last_advantage

            advantages[:,t]=last_advantage
            last_value=values[:,t]

        return advantages</code></pre>
<h3 id="maths">Maths</h3>
<p><span class="math inline">\(\hat{A_t^{1}}\)</span>=<span class="math inline">\(r_t+\gamma V(s_{t+1})-V(s)\)</span><br />
<span class="math inline">\(\hat{A_t^{2}}\)</span>=<span class="math inline">\(r_t+\gamma r_{t+1}+\gamma^2 V(s_{t+2})-V(s)\)</span><br />
...<br />
<span class="math inline">\(\hat{A_t^{\infty}}\)</span>=<span class="math inline">\(r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+...-V(s)\)</span><br />
The first advantage function <span class="math inline">\(\hat{A_t^{1}}\)</span> has a high bias, and low variance, whereas <span class="math inline">\(\hat{A_t^{\infty}}\)</span> has high variance but low bias. We want a tradeoff between these advantage functions, hence we take a weighted sum of all the advantages, called Generalized advantage estimation, <span class="math inline">\(\hat{A_t^{GAE}}=\Sigma_k w_k \hat{A_t^{k}} \)</span><br />
<span class="math inline">\(w_k\)</span> is set as <span class="math inline">\(\lambda^{k-1}\)</span><br />
<span class="math inline">\(\hat{A_t^{GAE}}=\frac{1}{1-\lambda}[\hat{A_t^{1}} + \lambda \hat{A_t^{2}}+...+\lambda^{k-1} \hat{A_t^{\infty}}] \)</span><br />
We use <span class="math inline">\(\delta_t\)</span> which is much easier to deal with that calculating advantage functions.<br />
<span class="math inline">\(\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)\)</span></p>
<p><span class="math inline">\(\hat{A_t} = \delta_t+ \gamma \lambda \delta_{t+1}+ ....+ (\gamma \lambda)^{T-t+1}\delta_{T-1} =\delta_t+ \gamma \lambda \hat{A_{t+1}}\)</span><br />
Usign these equations, we can calculate the advantage function recursively for all the states encountered. Key thing to notice is that we are reversing the order of the states, ie we start by calculating advantage for the last state fisrt and then move to the last second.<br />
Now we are done calculating all the relevant information of the samples. The next part of the process is training</p>
<h2 id="train">Train</h2>
<p>This is the part where we are training the model using the generated samples from out current policy. We plan to improve our current policy.</p>
<pre><code>    def train(self, samples, learning_rate, clip_range):

        train_info=[]

        for _ in range(self.epochs):

            indexes=torch.randperm(self.batch_size)

            for start in range(0, self.batch_size, self.mini_batch_size):

                end=start+self.mini_batch_size
                mini_batch_indexes =indexes[start:end]
                mini_batch={}

                for k,v in samples.items():
                    mini_batch[k]=v[mini_batch_indexes]

                res=self.trainer.train(learning_rate=learning_rate,             clip_range=clip_range, samples=mini_batch)

                train_info.append(res)
        return np.mean(train_info, axis=0)</code></pre>
<p>We are taking input as the samples(consisting of obs,actions,values,advantages,and rewards).<br />
Now we divide the current data into certain minibatches, and then train model, using these minibatches.<br />
Trainer.train fuction will be used to finally train using the mini-batches.</p>
<h1 id="class-trainer">Class Trainer</h1>
<p>First of all, the mode of optimization is set. We are going to use torch.optim.Adam for the optimizatin process.<br />
Trainer.train takes in the samples, which is the mini-batch in this case and also the learning rate and the clip range. We will discuss what is clip range in details.<br />
First part is creating local variables for all the attributes of the samples.<br />
The current policy is generated, for all the samples in the dataset. this is stored in pi, value.<br />
Next, we will look at the derivation for the loss function.<br />
</p>
<h2 id="maths-1">Maths</h2>
<p>The content for this part is borrowed from <strong>Berkeley Deep RL course</strong> and <strong>PPO paper</strong>. Policy Gradient methods try to optimize the function</p>
<p><span class="math inline">\(max_{\theta} J(\pi_{\theta})=E_{\tau=\pi_{\theta}}[\Sigma_{t=0}^{\infty}\gamma^{t} r_t]\)</span></p>
<p>The equation is optimized for <span class="math inline">\(\theta\)</span>, but the sample efficiency is poor. the data is thrown out after one update, and distance in parameter space does not correspond to distance in policy space.<br />
PPO used the concept of relative policy performance, to update the parameters. ie.</p>
<p><span class="math inline">\(max_{\pi^{new}}J(\pi^{new})=max_{\pi^{new}}J(\pi^{new})- J(\pi^{old})\)</span></p>
<p>Here, <span class="math inline">\(\pi^{new}\)</span> represents the new policy, and <span class="math inline">\(\pi^{old}\)</span> is the old policy.<br />
We need to find a more simplified version of the above expression.<br />
<span class="math display">\[\begin{aligned}
J(\pi^{new})- J(\pi^{old}) &amp;= J(\pi^{new})-E_{\tau \sim \pi^{new}}[V^{\pi_{old}}(s_0)] \\
&amp; = J(\pi^{new})+ E_{\tau \sim \pi^{new}}[\Sigma_{t=1}^{\infty} \gamma^t V^{\pi_{old}}(s_t)-\Sigma_{t=0}^{\infty} \gamma^t V^{\pi_{old}}(s_t)]\\
&amp; = J(\pi^{new})+ E_{\tau \sim \pi^{new}}[\Sigma_{t=0}^{\infty} \gamma^{t+1} V^{\pi_{old}}(s_{t+1})-\Sigma_{t=0}^{\infty} \gamma^t V^{\pi_{old}}(s_t)]\\
&amp; =E_{\tau \sim \pi^{new}}[\Sigma_{t=0}^{\infty}\gamma^t(R(s_t,a_t,s_{t+1})) +\gamma V^{\pi_{old}}(s_{t+1})-V^{\pi_{old}}(s_t)]\\
&amp; = E_{\tau \sim \pi^{new}}[\Sigma_{t=0}^{\infty} \gamma^t A^{\pi}(s_t,a_t)]\\
&amp; = \frac{1}{1-\gamma} E_{s \sim d^{\pi_{new}}, a \sim \pi^{new}} [A^{\pi}(s_t,a_t)]\\
&amp; = \frac{1}{1-\gamma} E_{s \sim d^{\pi_{new}}, a \sim \pi^{old}} [ \frac{\pi^{new}(a|s)}{\pi^{old}(a|s)}A^{\pi}(s_t,a_t)] (importance-sampling)\\
\end{aligned}\]</span> The only problem with the given equation is that we cannot sample from <span class="math inline">\(d^{\pi_{new}}\)</span>, so we say that <span class="math inline">\(d^{\pi_{new}}=d^{\pi_{old}}\)</span> and this assumption turns out to be pretty good, becuase there is not much difference in the discounted state distribution. We denote this as <span class="math inline">\(L^{CPI}(\theta)\)</span>. Without constraint, <span class="math inline">\(L^{CPI}(\theta)\)</span> would lead to excessively large policy updates. Therefore, large policy updates are constrained.<br />
<span class="math inline">\(r_t(\theta)=\frac{\pi^{new}(a|s)}{\pi^{old}(a|s)}A^{\pi}(s_t,a_t)\)</span><br />
</p>
<p><span class="math inline">\(L^{CLIP}(\theta)=\hat{E}_t[min(r_t(\theta), clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]\)</span><br />
</p>
<p>This is the main proposed objective function.</p>
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
