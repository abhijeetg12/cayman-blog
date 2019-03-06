

#--- 
#layout: post 
#title: Trying out LaTeX on HTML
#categories: [Catagory 1, Catago<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
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
<p>Jean Marie Linhart gives guidance for incorporating writing and communication. <span class="math inline">\(\frac{1}{2}\)</span></p>
<pre><code># Hello.java
import javax.swing.JApplet;
import java.awt.Graphics;
class Main(object):
    def __init__(self):

        self.gamma=0.99
        self.lamda=0.95
        self.updates=10000
        self.epochs= 4   #4
        self.n_workers= 8
        self.worker_steps=128
        self.n_mini_batch=4
        self.batch_size=self.n_workers*self.worker_steps  #[8*128=1024]
        self.mini_batch_size=self.batch_size // self.n_mini_batch  #[1024/4=256]
        np.random.seed(1234)
        torch.manual_seed(1234)


        assert(self.batch_size%self.n_mini_batch==0)

        self.workers=[Worker(47+i) for i in range(self.n_workers)]
        self.obs= np.zeros((self.n_workers,84,84,4),dtype=np.uint8)

        for worker in self.workers:
            worker.child.send((&quot;reset&quot;, None))

        for i, worker in enumerate(self.workers):
            self.obs[i]=worker.child.recv()

        self.model=Model()
        print(device)
        #self.model.to(device)
        #print(self.model.parameters())
        self.trainer=Trainer(self.model)
    ### Sample with current policy
    def sample(self):
        rewards=np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        actions=np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
        dones = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        obs = np.zeros((self.n_workers, self.worker_steps, 84, 84, 4), dtype=np.uint8)
        neg_log_pis = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        values = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        episode_infos = []

        for t in range(self.worker_steps):
            #print t, &#39;th time step&#39;
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
                # if rewards[w,t]!=0:
                #   print rewards[w,t],w,t
                

                if info:
                    #print info, w
                    info[&#39;obs&#39;]=obs[w,t,:,:,3]
                    episode_infos.append(info)
        #obs1=obs.cpu().data.numpy()
        #np.save(&quot;outfile.txt&quot;, obs)
        #np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        # with open(&quot;data_out.txt&quot;, &#39;w&#39;) as f:
        #   f.write(np.array2string(obs, separator=&#39;, &#39;))
        # f.close

        advantages = self._calc_advantages(dones, rewards, values)

        samples={&#39;obs&#39;:obs, &#39;actions&#39;:actions, &#39;values&#39;:values, &#39;neg_log_pis&#39;:neg_log_pis, &#39;advantages&#39;:advantages, &#39;rewards&#39;:rewards}

        samples_flat={}
        for k,v in samples.items():
            v=v.reshape(v.shape[0]*v.shape[1], *v.shape[2:])

            if k==&#39;obs&#39;:
                samples_flat[k]=obs_to_torch(v)
            else:
                samples_flat[k]= torch.tensor(v, device=device)
        return samples_flat, episode_infos

    def _calc_advantages(self, dones, rewards, values):

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

        return advantages
    def train(self, samples, learning_rate, clip_range):

        train_info=[]

        for _ in range(self.epochs):

            indexes=torch.randperm(self.batch_size)

            for start in range(0, self.batch_size, self.mini_batch_size):

                end=start+self.mini_batch_size
                mini_batch_indexes =indexes[start:end]
                mini_batch={}

                for k,v in samples.items():
                    mini_batch[k]=v[mini_batch_indexes]

                res=self.trainer.train(learning_rate=learning_rate, clip_range=clip_range, samples=mini_batch)

                train_info.append(res)
        return np.mean(train_info, axis=0)

    def run_training_loop(self):

        loss_vec=[]
        rewards_vec=[]

        episode_info=deque(maxlen=100)
        for update in range(self.updates):
            time_start=time.time()
            progress =update/self.updates

            learning_rate=3e-4 *(1-progress)
            clip_range= 0.1*(1-progress)

            samples, samples_episode_info=self.sample()
            #print samples, &quot;These are the samples in the loop no&quot;, update, &#39;\n&#39;

            info_loss=self.train(samples, learning_rate, clip_range)
            #info_loss=info_loss.numpy()
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
            #   plt.plot( range(len(loss_vec)),loss_vec)
            #   plt.show()
                #print loss_vec
                #with open(&quot;data_out.txt&quot;, &#39;w&#39;) as f:
                #   f.write((loss_vec, separator=&#39;, &#39;))
                #f.close
                np.save(&#39;loss_vec1_pong&#39;, loss_vec)
                np.save(&#39;reward_vec1_pong&#39;, rewards_vec)



    @staticmethod

    def _get_mean_episode_info(episode_info):
        #print episode_info

        if len(episode_info)&gt;0:
            return ( np.mean([info[&quot;reward&quot;] for info in episode_info]), np.mean([info[&quot;length&quot;] for info in episode_info]))

        else:
            return np.nan, np.nan

    def destroy(self):
        for worker in self.workers:
            worker.child.send((&quot;close&quot;, None))  


}</code></pre>
</body>
</html>
ry 2] 
#tags: [Tag 1, Tag 2] 
#comments: true 
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
  ```html
<html>
  <head>
  </head>
  <body>
    <p>Hello, World!</p>
  </body>
</html>
```
<p>Jean Marie Linhart gives guidance for incorporating writing and communication. <span class="math inline">\(\frac{1}{2}\)</span></p>
</body>
</html>
