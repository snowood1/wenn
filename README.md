# WGAN-ENN

This repository contains the essential code for the paper [Multidimensional Uncertainty-Aware Evidential Neural Networks (AAAI 2021)](https://arxiv.org/abs/2012.13676).

The code is written by Python 3.8 and pytorch 1.5 in GPU version. It has also been tested under Python 3.6 and pytorch 1.7.

## How to run

1. Create folders 'datasets' and 'results' to save download datasets and output results.
2. The datasets used included MNIST, notMNIST, FashionMNIST,CIFAR10, CIFAR100, SVHN, LSUN.
   You can download [notMNIST_small](http://yaroslavvb.com/upload/notMNIST/)
   You can evaluate on either LSUN_classroom or LSUN_resize. The results are close.
   You can download [LSUN_resize](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz?file_subpath=%2FLSUN_resize)
   Other datasets will be download automatically when you run the code at the first time. 
  
3. Run test_demo.py
4. We usually first pretrain the classfier to reach a good accuracy and then feed it into our joint alogrithm to calibrate its unceratiny.
   I have provided all the pretrained classifiers insider the folder 'pretrain'. Note that this pretrained models are different from the normal
   pretrained model. Because the output of the pretrained ENN is alpha, not softmax probability.
  
5. Our model use the rest code as baselines.
 [Outlier Explosure](https://github.com/hendrycks/outlier-exposure).  Our baselines, MSP and OE come from this repo.
 [CCU](https://github.com/AlexMeinke/certified-certain-uncertainty) also contain many baselines considered in our paper. 
 [BayesByHpernet](https://github.com/pawni/BayesByHypernet) contains all the baselines of bayes methods considered in our paper.



## Citation

If you find this repo useful in your research, please consider citing:

    @article{hu2020multidimensional,
      title={Multidimensional Uncertainty-Aware Evidential Neural Networks},
      author={Hu, Yibo and Ou, Yuzhe and Zhao, Xujiang and Cho, Jin-Hee and Chen, Feng},
      journal={arXiv preprint arXiv:2012.13676},
      year={2020}
    }
