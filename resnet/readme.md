#***ResNet学习笔记***  
在深度学习当中神经网络越深，往往有着越高的性能，但同时越难以训练，会面临梯度消失、梯度爆炸（反向传播的时候，由于梯度是一个小于1或者大于的1的数不断累乘导致梯度趋近于0或无穷）等问题 。  
而ResNet这样一种残差学习网络，它通过引入残差学习中短路连接（Shortcut Connection）的结构，和批量归一化（Batch Normalization）的方法有效解决了梯度消失或者爆炸的问题。  

#**残差学习**：残差块（residual block）、 短路连接（Shortcut Connection）   
ResNet的基本单元是残差块（residual block），其输入为x，通过两个带有激活函数的卷积层得到残差映射F(x)，然后将原始输入x与残差映射相加，得到输出H(x) = F(x) + x。这种设计使得网络可以学习残差，而不是直接学习整个映射。

例如输入为x，经过卷积层的处理得到F(x)，那么ResNet的残差块的输出为H(x) = F(x) + x。这样有助于梯度更直接地传播，因为梯度不需要通过多个层逐渐传递，而是通过短路连接（Shortcut Connection）直接反向传播，缓解了梯度消失或爆炸问题。  
  
#**批量归一化（Batch Normalization）**  
批量归一化（Batch Normalization）,通常是在残差块（residual block）内的激活函数之前应用的。

1. 位置： BN 被放置在每个残差块的卷积操作之后、激活函数之前。具体而言，对于每个残差块，BN 的应用顺序为卷积-BN-激活函数。

2. 操作：在每个残差块内，对卷积的输出进行批量归一化，然后再应用激活函数。这有助于规范化每层的输入，加速收敛，提高网络的训练稳定性和泛化性能。  
  
在每个残差块内进行批量归一化。这样的设计使得 ResNet 在训练时更加稳定，加速收敛。这些都有助于ResNet训练非常深的网络。

#***代码复现***  
在阅读论文后，在已有代码基础上，定义了残差块，更换了神经网络实现了代码复现。并在查阅资料的过程中发现有BasicBlock，ResidualBlock，Bottleneck三种残差块。  
1. BasicBlock是基础版本，主要用来构建ResNet18和ResNet34网络，里面只包含两个卷积层，使用了两个3&times;3的卷积，通道数都是64，卷积后接着 BN 和 ReLU。 
2. ResidualBlock与BasicBlock主要不同是`out += residual`使用了残差连接将输入输出相加，而BasicBlock没有。
3. Bottleneck主要用在ResNet50及以上的网络结构，与BasicBlock不同的是这里有 3 个卷积，分别为1&times;1和3&times;3和1&times;1大小的卷积核，分别用于压缩维度、卷积处理、恢复维度。
这里的通道数是变化的，1&times;1卷积层的作用就是用于改变特征图的通数，使得可以和恒等映射x相叠加，另外这里的1&times;1卷积层改变维度的很重要的一点是可以降低网络参数量，这也是为什么更深层的网络采用BottleNeck而不是BasicBlock的原因。  

于是想要在res18中对比BasicBlock，ResidualBlock；在res50中对比ResidualBlock，Bottleneck。它们在batch_size,learning_rate等参数一样的情况下，其在CIFAR-10数据上的准确率。  
以下是实验结果的可视化图表：  
1. 100epochs后BasicBlock的准确率收敛于80%左右  
<img height="200" src="D:\github\resnet\18  BasicBlock\basicblock accuracy.png" width="200"/>  
用时83min  
<img height="200" src="D:\github\resnet\18  BasicBlock\start time.png" width="200"/><img height="200" src="D:\github\resnet\18  BasicBlock\end time.png" width="200"/>  
2. 100epochs后ResidualBlock的准确率收敛于85%左右  
<img height="200" src="D:\github\resnet\18 ResidualBlock\ResidualBlock.accuracy.png" width="200"/>  
用时84min  
<img height="200" src="D:\github\resnet\18 ResidualBlock\start_time.png" width="200"/><img height="200" src="D:\github\resnet\18 ResidualBlock\end_time.png" width="200"/>  
**总结：** ResidualBlock在略微增加了计算量的同时，准确率上升了5%。可见残差链接在略微增加了计算开销的情况下，对模型准确率有提升。  
（由于个人能力有限，在更换res50的网络的情况下，我的计算机在训练数次后出现了显存爆炸的问题，在尝试梯度累计（gradient accumulation）将bitch_size=64分为4个连续求4个梯度后平均再送入优化器后，仍然无法解决，遂放弃。）
