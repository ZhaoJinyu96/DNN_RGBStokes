# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:20:47 2019

@author: 0000145046
"""

import chainer
from chainer import initializers
import chainer.links as L
import chainer.functions as F


#-----------------------------------
# ResNet
# please refer to https://arxiv.org/pdf/1606.00373.pdf
#-----------------------------------
class ResBlockA(chainer.Chain):
    
    def __init__(self, in_size, ch, out_size):
        
        W = chainer.initializers.HeNormal()
        super(ResBlockA, self).__init__()
        
        with self.init_scope():           
            self.conv1 = L.Convolution2D(
                    in_size, ch, 3, stride=1, pad=1, 
                    initialW=W, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            
            self.conv2 = L.Convolution2D(
                    ch, out_size, 3, stride=1, pad=1, 
                    initialW=W, nobias=True)
            self.bn2 = L.BatchNormalization(out_size)
            
            
    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
            
        return F.relu(h + x)


class ResBlockB(chainer.Chain):
    
    def __init__(self, in_size, ch, out_size):
        
        W = chainer.initializers.HeNormal()
        super(ResBlockB, self).__init__()        
        with self.init_scope():
            
            self.conv1 = L.Convolution2D(
                    in_size, ch, 3, stride=2, pad=1, 
                    initialW=W, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            
            self.conv2 = L.Convolution2D(
                    ch, out_size, 3, stride=1, pad=1, 
                    initialW=W, nobias=True)
            self.bn2 = L.BatchNormalization(out_size)
            
            self.conv3 = L.Convolution2D(
                    in_size, out_size, 1, stride=2, pad=0,
                    initialW=W, nobias=True)
            self.bn3 = L.BatchNormalization(out_size)
            
            
    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = self.bn2(self.conv2(h1))
        
        h2 = self.bn3(self.conv3(x))
        
        return F.relu(h1 + h2)
    

class Upconv(chainer.Chain):
    
    def __init__(self, in_size, out_size):
        
        W = chainer.initializers.HeNormal()
        super(Upconv, self).__init__()        
        with self.init_scope():
            self.conv = L.Convolution2D(
                    in_size, out_size, 5, stride=1, pad=2,
                    initialW=W, nobias=True)
            
    def __call__(self, x):
        h = F.unpooling_2d(x, 2, cover_all=False)
        h = F.relu(self.conv(h))
        
        return h

class UpProj(chainer.Chain):
    
    def __init__(self, in_size, out_size):
        
        W = chainer.initializers.HeNormal()
        super(UpProj, self).__init__()
        with self.init_scope():
            
            self.conv1 = L.Convolution2D(
                    in_size, out_size, 5, stride=1, pad=2,
                    initialW=W, nobias=True)
            
            self.conv2 = L.Convolution2D(
                    out_size, out_size, 3, stride=1, pad=1,
                    initialW=W, nobias=True)
            
            self.conv_ = L.Convolution2D(
                    in_size, out_size, 5, stride=1, pad=2,
                    initialW=W, nobias=True)
            
    def __call__(self, x):
        h = F.unpooling_2d(x, 2, cover_all=False)
        
        h1 = F.relu(self.conv1(h))
        h1 = self.conv2(h1)
        
        h2 = self.conv_(h)
        
        return F.relu(h1 + h2)
        


class ResNet18(chainer.Chain):
        
    def __init__(self, out_size):
        super(ResNet18, self).__init__()       
        with self.init_scope():
            #initial block
            self.conv_init = L.Convolution2D(
                    None, 64, 7, stride=4, pad=3, 
                    initialW=initializers.HeNormal(), nobias=True)
            self.bn_init = L.BatchNormalization(64)
            
            #down sampling phase
            self.a01 = ResBlockA(64, 64, 64)
            self.a02 = ResBlockA(64, 64, 64)
            self.b03 = ResBlockB(64, 128, 128)
            self.a04 = ResBlockA(128, 128, 128)
            self.b05 = ResBlockB(128, 256, 256)
            self.a06 = ResBlockA(256, 256, 256)
            self.b07 = ResBlockB(256, 512, 512)
            self.a08 = ResBlockA(512, 512, 512)
            
            #middle block
            self.conv_mid = L.Convolution2D(
                    512, 512, 1, stride=1, pad=0,
                    initialW=initializers.HeNormal(), nobias=True)
            self.bn_mid = L.BatchNormalization(512)
            
            #up sampling phase
            self.up01 = Upconv(512, 256)
            self.up02 = Upconv(256, 128)
            self.up03 = Upconv(128, 64)
            self.up04 = Upconv(64, 32)
            self.up05 = Upconv(32, 16)
            self.up06 = Upconv(16, 8)
            
            #last output
            self.conv_last = L.Convolution2D(
                    8, out_size, 3, stride=1, pad=1,
                    initialW=initializers.HeNormal(), nobias=True)
            
            
    def __call__(self, x):
        #initial block
        h = self.bn_init(self.conv_init(x))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2, pad=0)
        
        #down sampling phase
        h = self.a01(h)
        h = self.a02(h)
        h = self.b03(h)
        h = self.a04(h)
        h = self.b05(h)
        h = self.a06(h)
        h = self.b07(h)
        h = self.a08(h)
        
        #mid
        h = self.bn_mid(self.conv_mid(h))
        
        #upsampling
        h = self.up01(h)
        h = self.up02(h)
        h = self.up03(h)
        h = self.up04(h)
        h = self.up05(h)
        h = self.up06(h)
        
        #last output
        h = F.relu(self.conv_last(h))
      
        return h


class ResNet34(chainer.Chain):
    
    def __init__(self, out_size):
        super(ResNet34, self).__init__()
        with self.init_scope():
            #initial block
            self.conv_init = L.Convolution2D(
                    None, 64, 7, stride=4, pad=3, 
                    initialW=initializers.HeNormal(), nobias=True)
            self.bn_init = L.BatchNormalization(64)
            
            #down sampling phase
            self.a1_01 = ResBlockA(64, 64, 64)
            self.a1_02 = ResBlockA(64, 64, 64)
            self.a1_03 = ResBlockA(64, 64, 64)
            
            self.b2_01 = ResBlockB(64, 128, 128)
            self.a2_02 = ResBlockA(128, 128, 128)
            self.a2_03 = ResBlockA(128, 128, 128)
            self.a2_04 = ResBlockA(128, 128, 128)
            
            self.b3_01 = ResBlockB(128, 256, 256)
            self.a3_02 = ResBlockA(256, 256, 256)
            self.a3_03 = ResBlockA(256, 256, 256)
            self.a3_04 = ResBlockA(256, 256, 256)
            self.a3_05 = ResBlockA(256, 256, 256)
            self.a3_06 = ResBlockA(256, 256, 256)
            
            self.b4_01 = ResBlockB(256, 512, 512)
            self.a4_02 = ResBlockA(512, 512, 512)
            self.a4_03 = ResBlockA(512, 512, 512)
            
            #middle block
            self.conv_mid = L.Convolution2D(
                    512, 512, 1, stride=1, pad=0,
                    initialW=initializers.HeNormal(), nobias=True)
            self.bn_mid = L.BatchNormalization(512)
            
            #up sampling phase
            self.up01 = Upconv(512, 256)
            self.up02 = Upconv(256, 128)
            self.up03 = Upconv(128, 64)
            self.up04 = Upconv(64, 32)
            self.up05 = Upconv(32, 16)
            self.up06 = Upconv(16, 8)
            
            #last output
            self.conv_last = L.Convolution2D(
                    8, out_size, 3, stride=1, pad=1,
                    initialW=initializers.HeNormal(), nobias=True)
            
    def __call__(self, x):
        #initial block
        h = self.bn_init(self.conv_init(x))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2, pad=0)
        
        #down sampling phase
        h = self.a1_01(h)
        h = self.a1_02(h)
        h = self.a1_03(h)
        
        h = self.b2_01(h)
        h = self.a2_02(h)
        h = self.a2_03(h)
        h = self.a2_04(h)
        
        h = self.b3_01(h)
        h = self.a3_02(h)
        h = self.a3_03(h)
        h = self.a3_04(h)
        h = self.a3_05(h)
        h = self.a3_06(h)
        
        h = self.b4_01(h)
        h = self.a4_02(h)
        h = self.a4_03(h)
        
        #mid
        h = self.bn_mid(self.conv_mid(h))
        
        #upsampling
        h = self.up01(h)
        h = self.up02(h)
        h = self.up03(h)
        h = self.up04(h)
        h = self.up05(h)
        h = self.up06(h)
        
        #last output
        h = F.relu(self.conv_last(h))
      
        return h
 
       
class ResNet34_UpProj(chainer.Chain):
    
    def __init__(self, out_size):
        super(ResNet34_UpProj, self).__init__()
        with self.init_scope():
            #initial block
            self.conv_init = L.Convolution2D(
                    None, 64, 7, stride=4, pad=3, 
                    initialW=initializers.HeNormal(), nobias=True)
            self.bn_init = L.BatchNormalization(64)
            
            #down sampling phase
            self.a1_01 = ResBlockA(64, 64, 64)
            self.a1_02 = ResBlockA(64, 64, 64)
            self.a1_03 = ResBlockA(64, 64, 64)
            
            self.b2_01 = ResBlockB(64, 128, 128)
            self.a2_02 = ResBlockA(128, 128, 128)
            self.a2_03 = ResBlockA(128, 128, 128)
            self.a2_04 = ResBlockA(128, 128, 128)
            
            self.b3_01 = ResBlockB(128, 256, 256)
            self.a3_02 = ResBlockA(256, 256, 256)
            self.a3_03 = ResBlockA(256, 256, 256)
            self.a3_04 = ResBlockA(256, 256, 256)
            self.a3_05 = ResBlockA(256, 256, 256)
            self.a3_06 = ResBlockA(256, 256, 256)
            
            self.b4_01 = ResBlockB(256, 512, 512)
            self.a4_02 = ResBlockA(512, 512, 512)
            self.a4_03 = ResBlockA(512, 512, 512)
            
            #middle block
            self.conv_mid = L.Convolution2D(
                    512, 512, 1, stride=1, pad=0,
                    initialW=initializers.HeNormal(), nobias=True)
            self.bn_mid = L.BatchNormalization(512)
            
            #up sampling phase
            self.up01 = UpProj(512, 256)
            self.up02 = UpProj(256, 128)
            self.up03 = UpProj(128, 64)
            self.up04 = UpProj(64, 32)
            self.up05 = UpProj(32, 16)
            self.up06 = UpProj(16, 8)
            
            #last output
            self.conv_last = L.Convolution2D(
                    8, out_size, 3, stride=1, pad=1,
                    initialW=initializers.HeNormal(), nobias=True)
            
    def __call__(self, x):
        #initial block
        h = self.bn_init(self.conv_init(x))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2, pad=0)
        
        #down sampling phase
        h = self.a1_01(h)
        h = self.a1_02(h)
        h = self.a1_03(h)
        
        h = self.b2_01(h)
        h = self.a2_02(h)
        h = self.a2_03(h)
        h = self.a2_04(h)
        
        h = self.b3_01(h)
        h = self.a3_02(h)
        h = self.a3_03(h)
        h = self.a3_04(h)
        h = self.a3_05(h)
        h = self.a3_06(h)
        
        h = self.b4_01(h)
        h = self.a4_02(h)
        h = self.a4_03(h)
        
        #mid
        h = self.bn_mid(self.conv_mid(h))
        
        #upsampling
        h = self.up01(h)
        h = self.up02(h)
        h = self.up03(h)
        h = self.up04(h)
        h = self.up05(h)
        h = self.up06(h)
        
        #last output
        h = F.relu(self.conv_last(h))
      
        return h
    
    

class ResNet_latefusion(chainer.Chain):
    
    def __init__(self, out_size):
        super(ResNet_latefusion, self).__init__()
        with self.init_scope():
            #two pararell ResNet
            self.res1 = ResNet18(out_size)
            self.res2 = ResNet18(out_size)
            
            #conv layer after conacat
            self.conv = L.Convolution2D(
                    6, 3, 1, stride=1, pad=0,
                    initialW=initializers.HeNormal(), nobias=True)
            
            
    def __call__(self, x):
        #separate dataset into s0xRGB, s1s2xRGB.
        x = F.split_axis(x, indices_or_sections=9, axis=1)
        x1 = F.concat([x[0], x[1], x[2]])
        x2 = F.concat([x[3], x[4], x[5], x[6], x[7], x[8]])
        
        #two pararell ResNet
        h1 = self.res1(x1)
        h2 = self.res2(x2)
        
        #concat
        h = F.concat([h1, h2])
        h = F.relu(self.conv(h))
        
        return h

