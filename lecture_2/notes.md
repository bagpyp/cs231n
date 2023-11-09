
# Nearest Neighbor

NN classifier just remembers all the training data at train time
Instant at train time, laborious at test time (backwards)
Scales linearly with training data size

Choice of L1 (Manhattan) metric vs L2 (Euclidean) is
arbitrary and discrete.  This is a hyperparameter

# k-Nearest Neighbor

Better performance at test time
k is another hyperparameter
Finds the k most similar (not just argmin)
Do a majority vote on the k labels (4 trucks one horse)

# Linear Classifier

Where the model is f(x,W) (x is image, W is parameters)
Simplest approach is 
```
f(x,W) = Wx + b
```
f is 10x1
x is (32*32*3)x1
W is 10x(32*32*3)
b is 10x1