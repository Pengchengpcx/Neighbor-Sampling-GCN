Graph Convolutional Networks in PyTorch
====
Toy pytorch implementation for neighbor sampling for SGD traning based on GCN for semi-supervised classification [1].

Note: original adjacent matrix is implemented by Sparse CUDA. For convenience, it is converted into dense version which significantly lowers down the training speed due to the lack of backend optimization.

For a high-level introduction to GCNs, see:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

![Graph Convolutional Networks](figure.png)

Note: There are subtle differences between the TensorFlow implementation in https://github.com/tkipf/gcn and this PyTorch re-implementation. This re-implementation serves as a proof of concept and is not intended for reproduction of the results reported in [1].

This implementation makes use of the Cora dataset from [2].

## Installation

```python setup.py install```

## Requirements

  * PyTorch 0.4 or 0.5
  * Python 2.7 or 3.6

## Usage

```python train.py```

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)

[3] [Hamilton, William L, Ying, Rex, and Leskovec, Jure. Inductive representation learning on large graphs.](https://arxiv.org/abs/1706.02216)

