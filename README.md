# WGAN-ENN

This repository contains the essential code for the paper [Multidimensional Uncertainty-Aware Evidential Neural Networks (AAAI 2021)](https://arxiv.org/abs/2012.13676).

The code is written by Python 3.8 and pytorch 1.5 in GPU version. It has also been tested under Python 3.6 and pytorch 1.7.

## How to run

1. Create folders 'datasets' and 'results' to save download datasets and output results.
2. Most of the datasets will be download automatically when you run the code at the first time. 
   You can download [notMNIST_small](http://yaroslavvb.com/upload/notMNIST/)
   You can evaluate on either LSUN_classroom or [LSUN_resize](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz?file_subpath=%2FLSUN_resize). The results are close.
  
3. Run test_demo.py
4. We first pretrain the ENN classfier to reach a good accuracy and then feed it into the alogrithm to calibrate its unceratiny.
   All the pretrained classifiers is under the folder 'pretrain'. 
   Note that this pretrained ENN models are different from the normal softmax pretrained model.
  
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
