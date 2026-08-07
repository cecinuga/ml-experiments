## Neural Sorting

This project explores the concept of sorting using neural networks.


### Loss Function

The loss function compute the total entropy of a list of numbers. The entropy is calculated using the formula:

`loss(x) = sum(x_i - x_(i-1))`

There are two types of `minimum` value that the function can reach: when the `minimum > 0` the list of numbers will be sorted in ascending order, and when the `minimum < 0` the list of numbers will be sorted in descending order.

### Training

The whole training process consist in minimizing the `loss function` (entropy) of the given list of numbers, the model will learn the optimal configuration of that list where disorder is minimized, so the numbers will be sorted in ascending or descending order depending on the `minimum` value of the loss function.