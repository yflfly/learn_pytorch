# coding:utf-8
import torchvision.models as models

model = models.resnet18()
for param in model.named_parameters():
    print(param[0])

'''
nn.Module里面关于参数有两个很重要的属性named_parameters()和parameters()，前者给出网络层的名字和参数的迭代器，而后者仅仅是参数的迭代器。
'''

'''
运行的结果如下所示：
conv1.weight
bn1.weight
bn1.bias
layer1.0.conv1.weight
layer1.0.bn1.weight
layer1.0.bn1.bias
layer1.0.conv2.weight
layer1.0.bn2.weight
layer1.0.bn2.bias
layer1.1.conv1.weight
layer1.1.bn1.weight
layer1.1.bn1.bias
layer1.1.conv2.weight
layer1.1.bn2.weight
layer1.1.bn2.bias
layer2.0.conv1.weight
layer2.0.bn1.weight
layer2.0.bn1.bias
layer2.0.conv2.weight
layer2.0.bn2.weight
layer2.0.bn2.bias
layer2.0.downsample.0.weight
layer2.0.downsample.1.weight
layer2.0.downsample.1.bias
layer2.1.conv1.weight
layer2.1.bn1.weight
layer2.1.bn1.bias
layer2.1.conv2.weight
layer2.1.bn2.weight
layer2.1.bn2.bias
layer3.0.conv1.weight
layer3.0.bn1.weight
layer3.0.bn1.bias
layer3.0.conv2.weight
layer3.0.bn2.weight
layer3.0.bn2.bias
layer3.0.downsample.0.weight
layer3.0.downsample.1.weight
layer3.0.downsample.1.bias
layer3.1.conv1.weight
layer3.1.bn1.weight
layer3.1.bn1.bias
layer3.1.conv2.weight
layer3.1.bn2.weight
layer3.1.bn2.bias
layer4.0.conv1.weight
layer4.0.bn1.weight
layer4.0.bn1.bias
layer4.0.conv2.weight
layer4.0.bn2.weight
layer4.0.bn2.bias
layer4.0.downsample.0.weight
layer4.0.downsample.1.weight
layer4.0.downsample.1.bias
layer4.1.conv1.weight
layer4.1.bn1.weight
layer4.1.bn1.bias
layer4.1.conv2.weight
layer4.1.bn2.weight
layer4.1.bn2.bias
fc.weight
fc.bias
'''

model = models.resnet18()
for param in model.parameters():
    print(param)

'''
运行的结果如下所示：

Parameter containing:
tensor([[[[-6.1063e-04, -1.0914e-02,  2.3233e-02,  ...,  1.7720e-02,
            3.4688e-02, -4.9885e-02],
          [-1.9044e-02, -1.7548e-02,  5.4321e-04,  ..., -9.3911e-03,
            3.4166e-02, -3.6741e-02],
          [ 2.1910e-02, -2.1339e-02,  2.2120e-02,  ...,  6.7339e-03,
           -1.4824e-03,  2.3262e-02],
          ...,
Parameter containing:
tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], requires_grad=True)
Parameter containing:
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       requires_grad=True)
Parameter containing:
tensor([[[[-3.2374e-02,  2.4081e-02, -1.9869e-02],
          [-2.6259e-02,  1.6268e-03,  1.7452e-02],
          [-2.4676e-02, -4.7733e-03, -1.0286e-01]],

         [[-3.1331e-02,  5.5093e-02, -7.7530e-02],
          [-3.4580e-02, -1.2533e-01,  3.4215e-02],
          [ 3.1873e-02,  1.2433e-02, -2.3854e-02]],

         [[-5.1912e-02,  5.2431e-02, -4.6777e-02],
          [ 5.5206e-02, -1.3511e-02,  6.8645e-02],
          [-4.1090e-02, -4.5478e-02, -8.6213e-02]],

         ...,
等等
'''