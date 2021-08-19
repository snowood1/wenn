# WGAN-ENN

This repository contains the essential code for the paper [Multidimensional Uncertainty-Aware Evidential Neural Networks (AAAI 2021)](https://arxiv.org/abs/2012.13676).

The code is written by Python 3.8 and pytorch 1.5 in GPU version. It has also been tested under Python 3.6 and pytorch 1.7.


## Quick Start

1. Create folders 'datasets' and 'results' to save downloaded datasets and output results.
2. Most of the datasets will be downloaded automatically. 
   You can download [notMNIST_small](http://yaroslavvb.com/upload/notMNIST/)
   You can evaluate on either LSUN_classroom or [LSUN_resize](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz?file_subpath=%2FLSUN_resize). The results are close.
3. Run test_demo.py


## More details
4. We first pretrain the ENN classfier to reach a good accuracy and then feed it into the alogrithm to calibrate its uncertainty.
   All the pretrained classifiers are under the folder '[pretrain](https://github.com/snowood1/wenn/tree/main/pretrain)'. 
   Note that the pretrained ENN models are **different** from the normal softmax pretrained model.
   It is trained using Eq.9, not cross-entropy.  Please also refer to the pretrain function in main.py.
   
  
5. The repo below contain a lot of baselines considered in our paper. But we change all the base classfier as ResNet-20 in our experiments.

   * [Outlier Explosure](https://github.com/hendrycks/outlier-exposure)
 
   * [CCU](https://github.com/AlexMeinke/certified-certain-uncertainty) 
 
   * [BayesByHpernet](https://github.com/pawni/BayesByHypernet)
 
6. We also provide an [example](https://github.com/snowood1/wenn/blob/main/demo_saved_results.ipynb) of plotting Figure 4, Figure 5 and Figure 6.



## Citation

If you find this repo useful in your research, please consider citing:

    @article{hu2020multidimensional,
      title={Multidimensional Uncertainty-Aware Evidential Neural Networks},
      volume={35},
      url={https://ojs.aaai.org/index.php/AAAI/article/view/16954}, 
      number={9}, 
      journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
      author={Hu, Yibo and Ou, Yuzhe and Zhao, Xujiang and Cho, Jin-Hee and Chen, Feng}, 
      year={2021}, 
      month={May},
      pages={7815-7822}
    }
