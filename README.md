## Pytorch Implementation of CLIC.
### The official code of our paper:
### CLIC: Contrastive Learning Framework for Unsupervised Image Complexity Representation

- [x] arXiv: [https://arxiv.org/abs/2411.12792]()

### CLIC Framework and Results
<p align="center">
    <img width="50%" src="figure/framework.png">
</p>

<p align="center">
    <img width="50%" src="figure/result.png">
</p>


### t-SNE Visualization and Activation Map
<p align="center">
    <img width="50%" src="figure/t-SNE.png">
</p>
<p align="center">
    <img width="50%" src="figure/act_map.png">
</p>

## Usage of CLIC

### 1. Requirements

Note that the version is not required, just in our experiment.
```
python==3.7.6
torch==1.12.0+cu116
torchaudio==0.12.0+cu116
torchvision==0.13.0+cu116
```

### 2. Data Preparation

Download [ImageNet](https://image-net.org/) and [Flickr](https://huggingface.co/datasets/bigdata-pw/Flickr).

Fine-tuning dataset is IC9600. You can see their [github page](https://github.com/tinglyfeng/IC9600).

Flickr parser scripts is `./dada/get_flickr.py`. ImageNet parser please see [ImageNet](https://image-net.org/).

Images collection scripts is `./data/uniform_sample.py`. Then you can get the clic dataset and **folder architecture** is below:

```
clic_dataset
    |—— images
        |- 001.jpg
        |- 002.jpg
```

### 3. Unsupervised Training

To do unsupervised pre-training run:
```
python train.py 
# you can modify args in this scripts.
```

### 4. Fine-tuning

To do fine-tuning on [IC9600](https://github.com/tinglyfeng/IC9600) run:
```
python fine_tuning.py
# you can modify args in this scripts.
```

### Acknowledgment
* [MoCo](https://github.com/facebookresearch/moco): Official PyTorch implementation of the MoCo.
* [ICNet&IC9600](https://github.com/tinglyfeng/IC9600):IC9600: A Benchmark Dataset for Automatic Image Complexity Assessment.
* [ImageNet](https://image-net.org/): ImageNet: An large-scale image dataset.
* [Flickr-5B](https://huggingface.co/datasets/bigdata-pw/Flickr): Flickr: Approximately 5 billion images from Flickr.
