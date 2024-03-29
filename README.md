# CycleGAN-PyTorch  

CycleGAN的Pytorch实现 原论文：[https://arxiv.org/abs/1703.10593]  

## 原论文的浅要解读（渣翻）  
### 什么是GAN?  
>We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G
>that captures the data distribution, and a discriminative model D that estimates
>the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This
>framework corresponds to a minimax two-player game.
>GAN：Generative Adversarial Network，由一个辨别器D和一个生成器G构成。我们同时训练这两个模型。  
训练过程中，生成器的目标就是尽量生成真实的图片去欺骗辨别器D。而网络D的目标就是尽量把网络G生成的图片和真实的图片分别开来。
这样，G和D构成了一个动态的“博弈过程”。
一个专门制作赝品的团队，为了


