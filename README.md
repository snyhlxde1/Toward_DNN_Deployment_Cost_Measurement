# Toward_DNN_Deployment_Cost_Measurement

## Reference
[1] Ma, Ningning, et al. **Shufflenet v2: Practical guidelines for efficient cnn architecture design**. Proceedings of the European conference on computer vision (ECCV). 2018. [[paper]](https://arxiv.org/abs/1807.11164v1).

[2] **THOP: PyTorch-OpCounter**. [[code]](https://github.com/Lyken17/pytorch-OpCounter).

[3] **Flops counter for convolutional networks in pytorch framework**. [[code]](https://github.com/sovrasov/flops-counter.pytorch).

[4] Chang, Jiho, et al. **Reducing MAC operation in convolutional neural network with sign prediction.** 2018 International Conference on Information and Communication Technology Convergence (ICTC). IEEE, 2018. [[paper]](https://junheecho.com/assets/papers/ictc18.pdf).

[5] Model optimization: **model FLOPs**. [[slides]](https://indico.cern.ch/event/917049/contributions/3856417/attachments/2034165/3405345/Quantized_CNN_LLP.pdf).

[6] Wang, Xin, et al. **Skipnet: Learning dynamic routing in convolutional networks.** Proceedings of the European Conference on Computer Vision (ECCV). 2018. [[paper]](https://arxiv.org/abs/1711.09485)[[code]](https://github.com/ucbdrive/skipnet).

[7] ICLR‘20 Once-for-All tutorial: **Train One Network and Specialize it for Efficient Deployment**. [[paper]](https://arxiv.org/pdf/1908.09791.pdf), [[code]](https://github.com/mit-han-lab/once-for-all/tree/master/tutorial), [[talk]](https://youtu.be/a_OeT8MXzWI).


## Cost Evaluator
Evaluator class that can be leveraged to measure the deployement cost of a DNN. The costs are usually quantified with the following metrics:
1. Memory cost (in MB).
2. FLOPs.
3. MAC/MACCs (memory access cost/multiply-and-accumulate cost).

For memory cost, notice that the unit used here is MB with $1024^2$B = 1MB. 

According to [2, 3], FLOPs can be quantified by multiplying the size of the feature map on the basis of the parameters. 

$$\text{FLOPs} = \left[\left(K_h \times K_w\right) \times c_1 +1\right]  \left( h \times w \right) c_2 \\
= h w\left[  K_h \times K_w\times c_1c_2 + c_2\right] \\
= \text{feature map size} \times  \text{number of parameters}$$

Notice that The input and output to convolutional layers are three-dimensional feature maps or namely tensors of size $h \times w \times c$ where $h$ and $w$ are spatial sizes of the feature map.

For memory access cost (MAC), It can be quantified with number of multiply-and-accumulate operations (MACCs). [1] proposes a metric that sets a lower bound for MAC as 


$$\text{MAC} \geq 2 \sqrt{h w B} + \dfrac{B}{hw}$$

It reaches the lower bound when input channels and output channels are equal.

For $1\times 1$ group convolution, MAC can be precisely calculated as

$$\text{MAC} = hw (c_1+c_2) + \dfrac{c_1 c_2}{g} \\
= hwc_1 + \dfrac{Bg}{c_1} + \dfrac{B}{hw}$$

According to [2, 3], for a convolutional layer with kernel size $K$, MAC can be quantified with the number of MACCs as

$$\text{MAC} = \left(K\times K\right) \times \left(h \times w \right)\times c_1 \times c_2 \\
= \text{kernel size} \times \text{feature map size} \times \text{input channel} \times \text{output channel}$$
