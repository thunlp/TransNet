# TransNet
Source code and datasets of IJCAI2017 paper "TransNet: Translation-Based Network Representation Learning for Social Relation Extraction".

This work is selected as an example of the [“MLTrain”](http://auai.org/uai2017/training.php) training event in UAI 2017 (The Conference on Uncertainty in Artificial Intelligence). We release an ipython notebook that demonstrates the algorithm of TransNet. Details please refer to the "ipynb" directory.

## Datasets
This folder "data" contains three different scales of datasets extracted from [Aminer](https://cn.aminer.org/). Please unzip the "data.zip" file before using it.

* **aminer_s**: 187,939 vertices, 1,619,278 edges and 100 labels.
* **aminer_m**: 268,037 vertices, 2,747,386 edges and 500 labels.
* **aminer_l**: 945,589 vertices, 5,056,050 edges and 500 labels.

## Run

Run the following command for training TransNet:

    python train.py name_of_dataset alpha beta warm_up_to_reload transnet_to_reload

Here is an example:

    python train.py aminer_s/ 0.5 20 -1 -1
    
Explanations of the parameters:

* name_of_dataset: name of dataset ("aminer_s/", "aminer_m/" or "aminer_l/")
* alpha: the weight of autoencoder loss
* beta: the weight of non-zero element in autoencoder
* warm_up_to_reload: if >=0, reload saved autoencoder parameters and skip warm-up process
* transnet_to_reload: if >=0, reload saved TransNet parameters

## Dependencies
* Tensorflow == 0.12
* Scipy == 0.18.1
* Numpy == 1.11.2

## Cite
If you use the code, please cite this paper:

_Cunchao Tu, Zhengyan Zhang, Zhiyuan Liu, Maosong Sun. TransNet: Translation-Based Network Representation Learning for Social Relation Extraction.  The 26th International Joint Conference on Artificial Intelligence (IJCAI 2017)._

For more related works on network representation learning, please refer to my [homepage](http://thunlp.org/~tcc/).
