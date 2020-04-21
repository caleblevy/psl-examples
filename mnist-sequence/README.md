### MNIST Sequence

## Problem

An example that augments MNIST with structured data.
This is essentially the same experiments as the MNIST experiment in the DASL paper (Sikka 2020).
We choose a small number of training samples per label (2, 5, 10, 20),
and then augment the data with sequences like ((X + Y) % 10 = Z).
These sequences provide the structure for our model and are constructed only with unlabeled images.

## Dataset

MNIST.

## Keywords

 - `cli`
 - `neural`
