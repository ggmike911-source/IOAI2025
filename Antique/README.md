# Antique

## Overview

The objective of this task is to train a model that is capable of identifying fake antiques and real antiques. However, most of the sample labels are unknown, and we only have 4 labelled samples.

---

## Problem Setup

- Input dimension: [500, 5]
- Output dimension: [500, 1]
- Goal: Train a model to classify the authenticity of antique paintings.

---

## Baseline Information

| Model | Handle for unknown samples | Score (Average) |
|:-----:|:--------------------------:|:---------------:|
| SVC | Randomly | 0.475 |

```python
y[y == 0] = np.random.choice([-1, 1], size=(y == 0).sum())
```

The baseline scored 0.47 and 0.48 on test set A and test set B, respectively. The poor result (50% accuracy on 2 classes is essentially blind guessing) was mainly due to the fact that the unknown samples were labelled randomly.

## Thought Process

During the competition, I did not actually change the random assigning labels part. Instead, I changed the SVC to a simple neural network using PyTorch, and the accuracy of the model was around 0.75 at the competition.

Looking back now, the main problem we have to tackle is actually the unknown samples rather than building a more robust model. Because if the data contain a lot of noise, no matter how robust the model is, our predictions will be inaccurate. We have to think of ways to classify the labels for the unknown samples.

There are many ways to tackle unknown samples, but we have to identify which method is the most appropriate here. Looking at the data, only 4 of the samples are labelled, while 496 are unlabelled. The most effective way is to plot the data points against the features and look for patterns visually. So I will first plot the data points in a 2D graph using Matplotlib.

### 2D Features Graph

<img src="figs\2D_features_graph.png" width="80%" alt="2D Features Graph">

We can take a look at the first graph. We can see a typhoon-like pattern in it, so we can assume that the left side of the pattern corresponds to replicas and the right side corresponds to authentic pieces.

Now, when we look at the other graphs, there is not any obvious pattern we can capture just by looking at them. Next, maybe we can plot them against 3 features at a time and construct a 3D graph.

### 3D Features Graph

<img src="figs\3D_features_graph.png" width="80%" alt="3D Features Graph">

There is not an obvious pattern in these 3D graphs. However, we can kind of see that replicas and authentic pieces occupy different sides of the space.

Just based on what we saw in the graph, we can start to pick which method we should use to classify the unknown samples.

### K-Nearest Neighbors

<img src="figs\KNN-graph.png" width="500" alt="KNN graph on feature 1 and feature 0">

It will perform poorly because there are only 4 labelled neighbors in the entire dataset. The graph represents the area that will classify the sample as replica or authentic. We can see that even one of the authentic labels was in the replica area. Therefore, a lot of the samples would be labelled wrongly if we opt for KNN.

### K-Means Clustering

This will also perform poorly because, from what we saw in the 2D graphs, none of these have 2 visible clusters that can be separated. Therefore, the labels will also contain a lot of noise.

### Spectral Clustering

<img src="figs\Spectral_Clustering.png" width="80%" alt="Spectral Clustering graph">

Spectral clustering is the method used in the official solution. Looking at the graphs, apart from graph 1, the other graphs contain a lot of noise. Therefore, we can actually ignore features 2, 3, 4, and 5 and only train our model solely on features 0 and 1, as they have the strongest signal among all the other features.

The reason why spectral clustering is good at capturing complex patterns like spirals is that it groups data based on graph structure and local connectivity instead of simple geometric distances.

#### Official Solution

| Model | Handle for unknown samples | Score (Average) |
|:-----:|:--------------------------:|:---------------:|
| SVC | Spectral Clustering | 0.98 |

### Label Spreading

<img src="figs\Label_Spreading.png" width="500" alt="Label Spreading on feature 1 and feature 0">

The alternative method I can think of is label spreading. Looking at the graph, it does a fairly good job of separating the 2 clusters because label spreading is pretty similar to spectral clustering.

Therefore, I can experiment with label spreading and see how it performs on the task.

| Model | Handle for unknown samples | Score (Average) |
|:-----:|:--------------------------:|:---------------:|
| SVC | Label Spreading | 0.90 |

```python
label_prop_model = LabelSpreading(kernel='rbf', gamma=10, max_iter=50)
```

---

## Summary

The main problem we have to tackle here is the unknown samples, and by plotting the graphs against each feature, we realize that features 0 and 1 give the strongest signal while the other features are mostly noise.