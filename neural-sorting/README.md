## Neural Sorting

This project explores the concept of sorting using neural networks.


### Dataset
The dataset consists of a list of list of numbers, generated randomly.
In a first version the neural network will capable to work with fixed length arrays, it must be interpreted as a proof of concept, in a future version the neural network will be able to work with variable length arrays.


### Model Architecture
Nisba


### Loss Function
The loss function compute the total entropy of a list of numbers. The entropy is calculated using the formula:

`loss(x) = sum(x_i - x_(i-1))`

There are two types of `minimum` value that the function can reach: when the `minimum > 0` the list of numbers will be sorted in ascending order, and when the `minimum < 0` the list of numbers will be sorted in descending order.


### Optimizer
Classic SGD is used.


### Training
The whole training process consist in minimizing the `loss function` (entropy) of the given list of numbers, the model will learn the optimal configuration of that list where disorder is minimized, so the numbers will be sorted in ascending or descending order depending on the `minimum` value of the loss function.


### Learning
Un-supervised learning is used, the model will learn to reduce entropy of the given list of numbers.