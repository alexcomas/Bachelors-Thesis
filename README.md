# Bachelors Thesis

Code for "Building a robust Network Intrusion Detection system using Graph Neural Networks" project
This is a model which builds upon the model presented in the paper [Unveiling the potential of Graph Neural Networks for robust Intrusion Detection](https://arxiv.org/pdf/2107.14756.pdf).

By Alexandre Comas Gispert (alex.comas.g@gmail.com or s233148@dtu.dk)

## Abstract

In recent years there has been a major increase in malicious activities on the Internet, which caused major disruptions and sever economic and privacy implications. For this reason, the research group with which this Thesis will be developed published the paper [Unveiling the potential of Graph Neural Networks for robust Intrusion Detection](https://arxiv.org/pdf/2107.14756.pdf). In this paper, the authors propose "a novel Graph Neural Network (GNN) model that uses a non-standard message passing architecture specially designed to process and learn from host-connection graphs".

In [the paper](https://arxiv.org/pdf/2107.14756.pdf), the training and evaluation of the model has been done in its entirety with the well known [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), thus further evaluation and testing is needed. To this end, the model
has to be modified and each dataset has to be adequately preprocessed for the evaluation of the
model, and if necessary include it in the training.

The concept of a GNN-based NIDS has the theoretical advantage of learning to detect the attacks
by their structure, so they should be more resistant to adversarial attacks. In [the paper](https://arxiv.org/pdf/2107.14756.pdf), the robustness against some adversarial attacks is shown, such as the variation of packet size and the variation of the inter-arrival times. In this work we will aim to show the benefits of the GNN structure, as it makes the model much more robust to data with different scales than the data the model was trained on, but with similar essential structures, which will be the ones the model will distinguish.

Furthermore, as the world of Deep Learning evolves, we will aim to replicate and improve the implementation in [the paper](https://arxiv.org/pdf/2107.14756.pdf) on a fast growing framework, which allows for more transparency and customizability: Pytorch.

## Description

### Bachelor's Thesis code

#### Pytorch code

Code developed by me in its entirety.

#### Tensorflow code

Majority of the code taken from [here](https://github.com/BNN-UPC/GNN-NIDS). Some of it has been updated and adapted.


### Library `comaslib` (provisional name)

The idea is to adapt the files from the Bachelor's Thesis so it can be used as a library.

#### Structure

```
comaslib
    ├───data
    │   ├───processing
    │   ├───tf
    │   └───torch
    │       ├───dgl
    │       ├───dict
    │       └───torch_geometric
    ├───model
    │   ├───tf
    │   └───torch
    │       ├───dgl
    │       ├───dict
    │       └───torch_geometric
    └───utils
        ├───tf
        ├───torch
```