# CycleGAN-PyTorch from scratch
原论文：[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

## 什么是[GAN](https://arxiv.org/pdf/1406.2661.pdf)?
>
>We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models:   
>a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample   
> came from the training data rather than G.
> The training procedure for G is to maximize the probability of D making a mistake.  
>This framework corresponds to a minimax [two-player game](https://zh.wikipedia.org/zh-cn/%E9%9B%B6%E5%92%8C%E5%8D%9A%E5%BC%88)  
>
打个比方，一个专门制作赝品的团队，想要让他们伪造的梵高的画作达到以假乱真的水平。他们找来了一位曾经在梵高博物馆工作的鉴别师进行合作。他们将自己伪造的作品递给专家看，让专家
提意见，再进行改正，并重新给专家验证。在不断的重复提意见-改进-验证的博弈过程中，赝品团队和鉴别师的水平都不断得到了提高，最终终于得到了一幅完全能够骗过世界上顶级鉴别师的画作。
## [CycleGAN](https://arxiv.org/pdf/1406.2661.pdf)的过人之处：
cycle
## 原论文的代码架构（见[原文](https://arxiv.org/pdf/1406.2661.pdf)附录7.2）
### Generator architectures
架构来源于Johnson et al，对于128*128的图像采用6个**残差块**，对于256*256的图像采用9个**残差块**。
### Discriminator architectures
 
## 网络中不同Block机制的详细说明
### [residual blocks](https://arxiv.org/pdf/1512.03385.pdf)(残差块)：
#### ResNet的出现:
我们在堆叠**layers**的过程中，肯定希望随着复杂度的提高，网路的表现更好。但实际实验中发现very deep的[**CNN model**](https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)，在图像识别过程中的表现不佳，训练效果并不会简单的随着网络的不断堆积而不断提升。相反，在实验中，随着模型的不断堆叠，存在着**退化**（degradation：with the network depth increasing, accuracy gets saturated）问题。且此问题的原因并不来自于[**Over Fitting**](https://en.wikipedia.org/wiki/Overfitting)。  
#### 解决方案:
试想当较浅的model，已经表现的足够好。我们在进一步堆叠**layers**的过程中，
自然期望能够提高其表现，前文已提及，这无法行得通, 
那我们便想退而求其次，至少，随着**layers**的堆叠，模型不会变的更差，即什么都不做 **恒等映射（identity mapping)**
实现$H(x)=x$。但事实上，因为[**Rlelu**](https://zh.wikipedia.org/wiki/%E7%BA%BF%E6%80%A7%E6%95%B4%E6%B5%81%E5%87%BD%E6%95%B0)**即非线性mapping**的存在，原始映射很大程度上是$H(x)=F(x)+x$。为实现**恒等映射**，我们要实现$F(x)=H(x)-x$，这便是**偏差方程**。
原映射便变为$H(x)=F(x)+x$。而实现残差映射问题要比原映射更为容易，**通过一系列的非线性变换，实现残差为0，显然要比实现恒等映射要容易的多**。  
#### 实现：（constructing！！！）
借助于带有short cut的feedforward neural network。此结构与VGG-19相比，在更高的层数堆叠下，反而有更少的复杂度和准确性。  
![picture](https://production-media.paperswithcode.com/methods/resnet-e1548261477164_2_mD02h5A.png "Residual learning: a building block")
