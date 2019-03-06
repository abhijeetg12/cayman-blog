
---
title: " Proximal Policy Optimization"
tagline: "Implementing PPO paper on Atari-gym environments, using PyTorch"
categories: junk
image: /thumbnail-mobile.png
author: "Abhijeet Ghawade"
meta: "Springfield"
---
This blog will help you in undersatding the maths behind PPO, and also, the coding which is required to efficiently implement PPO in pyton using PyTorch. 
The best way to learn something is to implement it, and you learn a lot more things on the way doing it, I have learned quite a lot of things implementing this paper. 
\mathcal{W}(A,f) = (T,\bar{f})

## Synthetic data creation

Machine learning is all about data, lots of data.
Therefore it is really helpful if you can create data on your own, and learn how to visualise parts of it.
i have generated 2 classes with 20 features each. Each class is given by a multivariate Gaussian distribution, with both
classes sharing the same covariance matrix. Ensuring that the covariance matrix is not spherical, i.e., that it is not a diagonal matrix, with all the diagonal entries being the same. there are 2000 examples for each class. the centroids for these classes has been chosen as [0,0,0,...0], [1,1,1,...,1] respectively. 
enough so that there is some overlap in the classes.

Code for the dataset creation
```python
meanA=np.zeros(20)
#the data set has a mean at [0,0,0,...,0]
meanB=2*np.ones(20)#the data set has a mean at [2,2,2...,2]
cov=np.random.rand(20,20)
A=np.random.multivariate_normal(meanA,cov,length).T
B=np.random.multivariate_normal(meanB,cov,length).T
```

## k-NN Classifier
k-NN is short for K nearest neighbour. k-Nearest Neighbor (KNN) is one of the most popular algorithms for pattern recognition. Many researchers have found that the KNN algorithm accomplishes very good performance in their
experiments on dierent data sets. In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-
parametric method used for classication and regression.[1] In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression: In
k-NN classication, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer,
typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. In k-NN regression, the output is the property value for the object. This value is the average of the values of its k nearest
neighbors. On our dataset, I implemented the k-NN for various values of k ranging
from 1 to 100 in unit steps. The best accuracy was recieved for k=4, which is 95.57% acurate. al-
though this value is prone to change with change in datasets. We can observe that the kNN accuracy is increasing initially with increases in k, but after a while it starts to fall off. There is one peculiar thing when we plot kNN wrt n, we see there are spikes in every alternate value of k, this is a really interesting problem, and the reason in that when the value of k is even, there might be equal  number of points from each class in the nearest area, in that case the classifier has to give a random class as the output, which reduces the efficiency for that particular value of k, hence its best to keep the values of k as odd.  

### Some great subheading (h3)

Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum.

Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc.
<p>Jean Marie Linhart gives guidance for the equations here <span class="math inline">\(\epsilon=\gamma \frac{1}{2}\)</span></p>

### Some great subheading (h3)

Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

> This quote will change your life. It will reveal the secrets of the universe, and all the wonders of humanity. Don't misuse it.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt.

### Some great subheading (h3)

Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum.

```html
<html>
  <head>
  </head>
  <body>
    <p>Hello, World!</p>
  </body>
</html>
```


In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris.

#### You might want a sub-subheading (h4)

In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris.

In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris.

#### But it's probably overkill (h4)

In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris.

### Oh hai, an unordered list!!

In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris.

- First item, yo
- Second item, dawg
- Third item, what what?!
- Fourth item, fo sheezy my neezy

### Oh hai, an ordered list!!

In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris.

1. First item, yo
2. Second item, dawg
3. Third item, what what?!
4. Fourth item, fo sheezy my neezy



## Headings are cool! (h2)

Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc.
