# Chest x-ray Landmark Segmentation Dataset.

This git repo contains 911 landmark annotations for chest x-ray images from JSRT, Shenzhen, Montgomery and Padchest datasets.\
We are grateful to Padchest's team for providing people an easy way to download our subset of images, under "Sample 2".

The landmark annotations were obtained with an ensemble of HybridGNet 2-IGSC models, using dense segmentation masks as input.

![Example](figs/landmarks.png)

Pre-processing scripts are provided for every dataset, please download them from their official websites.\
Links provided in the corresponding .ipynb file.

Lung annotations are available for images from all datasets. \
Heart annotations are only available for JSRT and Padchest images.

Annotations:\
RL : Right Lung (44 landmarks) \
LL : Left Lung (50 landmarks) \
H : Heart (26 landmarks)

![workflow](figs/info.png)

## Baseline results

Results were taken under a 20% balanced test holdout set. For more details check on https://github.com/ngaggion/Chest-x-ray-Baselines

Results for both lungs and heart segmentation, using the JSRT + Padchest subset:

|      **Model** |        **MSE** | **Dice Lungs** | **Dice Heart** |  **HD Lungs** |  **HD Heart** |
|---------------:|---------------:|---------------:|---------------:|--------------:|--------------:|
|            PCA |     359.696809 |       0.938072 |       0.909292 |     50.469137 |     43.823793 |
|             FC |     359.423800 |       0.938108 |       0.907268 |     50.506582 |     44.751405 |
| **HybridGNet** | **220.685126** |   **0.965563** |   **0.933040** | **37.550593** | **35.982695** |

Results for lung segmentation only, using the full dataset:

|      **Model** |        **MSE** | **Dice Lungs** |  **HD Lungs** |
|---------------:|---------------:|---------------:|--------------:|
|            PCA |     216.482534 |       0.948304 |     40.945048 |
|             FC |     232.796428 |       0.945217 |     42.013886 |
| **HybridGNet** | **164.904743** |   **0.966357** | **34.492442** |

## Citation

If you are using the annotations, please cite our work:

Gaggion, N., Mansilla, L., Milone, D. H., & Ferrante, E. (2021, September). Hybrid graph convolutional neural networks for landmark-based anatomical segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 600-610). Springer, Cham.

https://link.springer.com/chapter/10.1007%2F978-3-030-87193-2_57

```
@InProceedings{10.1007/978-3-030-87193-2_57,
author="Gaggion, Nicol{\'a}s
and Mansilla, Lucas
and Milone, Diego H.
and Ferrante, Enzo",
title="Hybrid Graph Convolutional Neural Networks for Landmark-Based Anatomical Segmentation",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="600--610",
isbn="978-3-030-87193-2"
}
```
