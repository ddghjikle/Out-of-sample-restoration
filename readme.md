# Generalizing to Out-of-Sample Degradations via Model Reprogramming (IEEE TIP 2024)
Official implementation of ["**[Generalizing to Out-of-Sample Degradations via Model Reprogramming](https://scholar.google.com.hk/citations?hl=zh-CN&user=mD3lO60AAAAJ&view_op=list_works&sortby=pubdate)**"], *[Runhua Jiang](https://scholar.google.com.hk/citations?hl=zh-CN&view_op=list_works&gmla=AOV7GLOtVSSA5YAZIfaFwu9aInPr2OI4l-brSoherrpowMD_NndZ-hEbqmPezX6qy8zAq-KrcP1em6MjegzbUfzLK7U&user=mD3lO60AAAAJ)*, *[YaHong Han](http://cic.tju.edu.cn/faculty/hanyahong/index.html)*.
DOI: ***

We will update links once arXiv or journal versions are available.

> Existing image restoration models are typically designed for specific tasks and struggle to generalize to out-of-sample degradations not encountered during training. While zero-shot methods can address this limitation by fine-tuning model parameters on testing samples, their effectiveness relies on predefined natural priors and physical models of specific degradations. Nevertheless, determining out-of-sample degradations faced in real-world scenarios is always impractical. As a result, it is more desirable to train restoration models with inherent generalization ability. To this end, this work introduces the Out-of-Sample Restoration (OSR) task, which aims to develop restoration models capable of handling out-of-sample degradations. An intuitive solution involves pre-translating out-of-sample degradations to known degradations of restoration models. However, directly translating them in the image space could lead to complex image translation issues. To address this issue, we propose a model reprogramming framework, which translates out-of-sample degradations by quantum mechanic and wave functions. Specifically, input images are decoupled as wave functions of amplitude and phase terms. The translation of out-of-sample degradation is performed by adapting the phase term. Meanwhile, the image content is maintained and enhanced in the amplitude term. By taking these two terms as inputs, restoration models are able to handle out-of-sample degradations without fine-tuning. Through extensive experiments across multiple evaluation cases, we demonstrate the effectiveness and flexibility of our proposed framework.

### We will release training codes of Mindspore and codes of pytorch as soon as possible.

## 1. Introduction

<p align="center">
    <img src='/fig/intro.png' width=700/>
</p>

The introduced out-of-sample restoration task is to develop models with the capability of handling unknown degradations. It extends previous restoration researches as paying more attention to cross-degradation generalization.  Specifically, task-specific methods focus on establishing image-to-image translation network for a certain kind of degradation, while task-agnostic methods combine various types of degraded images to train a single restoration network.

<p align="center">
    <img src='/fig/self_comparison_4_00.png' width=700/>
</p>

As above results show, while the task-specific network Dehaze effectively eliminates haze degradation, it struggles to address out-of-sample degradations such as blur and rain. Furthermore, the task-agnostic network SR+dehaze, trained on datasets encompassing super-resolution and dehazing networks, fails to exhibit improved performance on rainy examples. Although zero-shot researches can address this issue by fine-tuning restoration models on testing samples, they require prior knowledge of degradation categories. In contrast, the OSR task aims to learn generalizable models from a limited set of training samples, making it different and complementary to previous researches.

## 2. Method

<p align="center">
    <img src='/fig/framework_new_2_00.png' width=800/>
</p>


First of all, the input transform module is designed following quantum mechanism, where entities are represented by wave functions comprising both amplitude and phase components. The amplitude corresponds to a real-valued feature that represents the maximum intensity of the wave, while the phase term modulates intensity patterns by indicating locations of each point. This design allows for the decoupling of input images into continuous vectors of content and style, with the style representation aligned to recognizable degradations of the reprogrammed model. Second, the reprogrammed model aims to map the style representation and enhance the content details by these components. Since no existing methods study the problem of reprogramming restoration models, two kinds of restoration models, \emph{i.e.}, randomly initialized and specifically trained, are explored in subsequent experiments. Finally, after processing these two components, the output transform function formulates wave functions and remaps them into the original image space to yield clear outputs.

## 3. Usage
### 3.1 Data
The datasets used in the paper are available at the following links. Please download and place them according to instructions of each task.
* Super-resolution: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/),[Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), [BSD100](https://drive.google.com/file/d/1n-7pmwjP0isZBK7w3tx2y8CTastlABx1/view?usp=sharing)
* Rain: [Rain1200](https://github.com/hezhangsprinter/DID-MDN?tab=readme-ov-file)
* Noise: [Five5K](https://data.csail.mit.edu/graphics/fivek/), [CBSD68](https://drive.google.com/file/d/1baLpOjNlTCNbREUDAZf9Lso6YCeUOQER/view?usp=sharing)
* Blur: [GoPro](https://seungjunnah.github.io/Datasets/gopro.html), [HIDE](https://github.com/joanshen0508/HA_deblur), [RealBlur](https://github.com/rimchang/RealBlur)
* Haze: [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)


### 3.2 Dependencies

* Python 3.8.15
* PyTorch 1.9.0
* Mindspore

### 3.3 Train

**To Do.**

### 3.4 Test

**We first release testing codes based on the Mindspore framework.**

```
python test.py
```



### Citation
If you find our code or paper useful, please consider citing:
```

```

### Acknowledgments

The code is build on both [GridDehazeNet](https://github.com/proteus1991/GridDehazeNet) and [Mindspore](https://www.mindspore.cn/). Of course, many wonderful studies are referred in our experiments. Thanks them very much for their sharing.
