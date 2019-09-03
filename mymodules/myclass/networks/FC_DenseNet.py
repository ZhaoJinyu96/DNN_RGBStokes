# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:10:26 2019

@author: 0000145046
"""

import chainer
import chainer.links as L
import chainer.functions as F


#------------------------------------
# This network is originally used for a semantic segmentation task in
# Jegou st al. [2017].
# please refer to:
#   https://arxiv.org/pdf/1611.09326.pdf
#------------------------------------
class DenseBlock(chainer.ChainList):
    
    def __init__(self, in_size, layer_num, growth_rate):
        
        W = chainer.initializers.HeNormal()
        super().__init__()
        
        with self.init_scope():
            for layer in range(layer_num):
                
                bn1 = L.BatchNormalization(in_size + growth_rate*layer)
                self.add_link(bn1)
                bn1.name = "bn1{}".format(layer)
                
                bottleneck = L.Convolution2D(
                        in_size + growth_rate*layer, 128, 1, stride=1, pad=0,
                        initialW=W, nobias=True)
                self.add_link(bottleneck)
                bottleneck.name = "bottleneck{}".format(layer)
                
                bn2 = L.BatchNormalization(128)
                self.add_link(bn2)
                bn2.name = "bn2{}".format(layer)
                
                conv = L.Convolution2D(
                        128, growth_rate, 3, stride=1, pad=1,
                        initialW=W, nobias=True)
                self.add_link(conv)
                conv.name = "conv{}".format(layer)
                
    
    def __call__(self, x):
        
        outputs = []
        
        for link in self.children():
            if "bn1" in link.name:
                x_ = F.relu(link(x))
            elif "bottleneck" in link.name:
                x_ = link(x_)
            elif "bn2" in link.name:
                x_ = F.relu(x_)
            elif "conv" in link.name:
                x_ = link(x_)
                outputs.append(x_)
                x = F.concat((x, x_), axis=1)
                
        for i, x_ in enumerate(outputs):
            if i == 0:
                x = x_
            else:
                x = F.concat((x, x_), axis=1)
                
        return x
    
    
class Transition_Down(chainer.Chain):
    
    def __init__(self, size):
        
        W = chainer.initializers.HeNormal()
        super().__init__()
        
        with self.init_scope():
            self.bn = L.BatchNormalization(size)
            self.conv = L.Convolution2D(
                    size, size, 1, stride=1, pad=0,
                    initialW=W, nobias=True)
            
    def __call__(self, x):
        
        x = F.relu(self.bn(x))
        x = self.conv(x)
        
        return F.max_pooling_2d(x, 2, stride=2, pad=0)
    
    
class Transition_Up(chainer.Chain):
    
    def __init__(self, size):
        
        W = chainer.initializers.HeNormal()
        super().__init__()
        
        with self.init_scope():
            self.conv = L.Deconvolution2D(
                    size, size, 4, stride=2, pad=1,
                    initialW=W, nobias=True)
            
    def __call__(self, x):
        
        return self.conv(x)
    
    
    
class FC_DenseNet67(chainer.Chain):
    
    def __init__(self, out_size):
        
        W = chainer.initializers.HeNormal()
        super().__init__()
        
        with self.init_scope():
            # first conv
            self.conv1 = L.Convolution2D(
                    None, 48, 3, stride=1, pad=1,
                    initialW=W, nobias=True)
            
            # down
            self.dense1 = DenseBlock(48, 5, 16)
            self.td1 = Transition_Down(128)
            self.dense2 = DenseBlock(128, 5, 16)
            self.td2 = Transition_Down(208)
            self.dense3 = DenseBlock(208, 5, 16)
            self.td3 = Transition_Down(288)
            self.dense4 = DenseBlock(288, 5, 16)
            self.td4 = Transition_Down(368)
            self.dense5 = DenseBlock(368, 5, 16)
            self.td5 = Transition_Down(448)
            
            # bottleneck
            self.dense6 = DenseBlock(448, 5, 16)
            
            # up
            self.tu1 = Transition_Up(80)
            self.dense7 = DenseBlock(528, 5, 16)
            self.tu2 = Transition_Up(80)
            self.dense8 = DenseBlock(448, 5, 16)
            self.tu3 = Transition_Up(80)
            self.dense9 = DenseBlock(368, 5, 16)
            self.tu4 = Transition_Up(80)
            self.dense10 = DenseBlock(288, 5, 16)
            self.tu5 = Transition_Up(80)
            self.dense11 = DenseBlock(208, 5, 16)
            
            # last conv
            self.conv2 = L.Convolution2D(
                    80, out_size, 1, stride=1, pad=0,
                    initialW=W, nobias=True)
            
    def __call__(self, x):
        # first conv
        x = self.conv1(x)
        
        # down
        h = self.dense1(x)
        h1 = F.concat((x, h), axis=1)
        x = self.td1(h1)
        
        h = self.dense2(x)
        h2 = F.concat((x, h), axis=1)
        x = self.td2(h2)
        
        h = self.dense3(x)
        h3 = F.concat((x, h), axis=1)
        x = self.td3(h3)
        
        h = self.dense4(x)
        h4 = F.concat((x, h), axis=1)
        x = self.td4(h4)
        
        h = self.dense5(x)
        h5 = F.concat((x, h), axis=1)
        x = self.td5(h5)
        
        # bottleneck
        h = self.dense6(x)
        
        # last conv
        h = self.tu1(h)
        h = F.concat((h, h5), axis=1)
        h = self.dense7(h)
        
        h = self.tu2(h)
        h = F.concat((h, h4), axis=1)
        h = self.dense8(h)
        
        h = self.tu3(h)
        h = F.concat((h, h3), axis=1)
        h = self.dense9(h)
        
        h = self.tu4(h)
        h = F.concat((h, h2), axis=1)
        h = self.dense10(h)
        
        h = self.tu5(h)
        h = F.concat((h, h1), axis=1)
        h = self.dense11(h)
        
        # last conv
        h = self.conv2(h)
        
        return h
        
            