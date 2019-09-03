# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:22:19 2019

@author: 0000145046
"""

import chainer
import chainer.links as L
import chainer.functions as F


#-----------------------------------
# VGG-16 and UNet like network introduced in Zhang et al. [2017]
# please refer to:
#     https://joonyoung-cv.github.io/assets/paper/17_cvpr_physically_based.pdf
#     https://arxiv.org/pdf/1409.1556.pdf
#-----------------------------------
class EncoderBlock(chainer.ChainList):
    def __init__(self, in_size, out_size, layer_num):
        
        W = chainer.initializers.HeNormal()
        super(EncoderBlock, self).__init__()
        
        with self.init_scope():
            for layer in range(layer_num):
                if layer == 0: # Initial layer
                    conv = L.Convolution2D(
                        in_size, out_size, 3, stride=1, pad=1,
                        initialW=W, nobias=True)
                else:
                    conv = L.Convolution2D(
                        out_size, out_size, 3, stride=1, pad=1,
                        initialW=W, nobias=True)
                
                # Add link
                self.add_link(conv)
                conv.name = "conv{}".format(layer)
                    
                bn = L.BatchNormalization(out_size)
                self.add_link(bn)
                bn.name = "bn{}".format(layer)
        

    def __call__(self, x):
        for link in self.children():
            x_ = link(x)
            if "conv" in link.name:
                x = x_
            elif "bn" in link.name:
                x = F.relu(x_)
        
        return F.max_pooling_2d(x, 2, stride=2, pad=0)
    

class EncoderBlock_wopool(chainer.ChainList):
    def __init__(self, in_size, out_size, layer_num):
        
        W = chainer.initializers.HeNormal()
        super(EncoderBlock_wopool, self).__init__()
        
        with self.init_scope():       
            for layer in range(layer_num):
                if layer == 0: # Initial layer
                    conv = L.Convolution2D(
                        in_size, out_size, 3, stride=1, pad=1,
                        initialW=W, nobias=True)
                    
                elif layer == layer_num-1: # Last layer
                    conv = L.Convolution2D(
                    out_size, out_size, 4, stride=2, pad=1,
                    initialW=W, nobias=True)

                else: # Middle layer
                    conv = L.Convolution2D(
                        out_size, out_size, 3, stride=1, pad=1,
                        initialW=W, nobias=True)
                
                # Add link
                self.add_link(conv)
                conv.name = "conv{}".format(layer)
                    
                bn = L.BatchNormalization(out_size)
                self.add_link(bn)
                bn.name = "bn{}".format(layer)

                  
    def __call__(self, x):
        for link in self.children():
            x_ = link(x)
            if "conv" in link.name:
                x = x_
            elif "bn" in link.name:
                x = F.relu(x_)
            
        return x


class DecoderBlock(chainer.ChainList):
    def __init__(self, in_size, out_size, layer_num):
        
        W = chainer.initializers.HeNormal()
        super(DecoderBlock, self).__init__()
        
        with self.init_scope():
            for layer in range(layer_num):
                if layer == 0:
                    conv = L.Convolution2D(
                        in_size, out_size, 3, stride=1, pad=1,
                        initialW=W, nobias=True)
                else:
                    conv = L.Convolution2D(
                        out_size, out_size, 3, stride=1, pad=1,
                        initialW=W, nobias=True)
                    
                # Add link
                self.add_link(conv)
                conv.name = "conv{}".format(layer)
                    
                bn = L.BatchNormalization(out_size)
                self.add_link(bn)
                bn.name = "bn{}".format(layer)

    def __call__(self, x):
        for link in self.children():
            x_ = link(x)
            if "conv" in link.name:
                x = x_
            elif "bn" in link.name:
                x = F.relu(x_)

        return F.unpooling_2d(x, 2, cover_all=False)


class DecoderBlock_wopool(chainer.ChainList):
    def __init__(self, in_size, out_size, layer_num):
        
        W = chainer.initializers.HeNormal()
        super(DecoderBlock_wopool, self).__init__()
        
        with self.init_scope():
            for layer in range(layer_num):
                if layer == 0:
                    conv = L.Convolution2D(
                        in_size, out_size, 3, stride=1, pad=1,
                        initialW=W, nobias=True)
                
                elif layer == layer_num-1:
                    conv = L.Deconvolution2D(
                    out_size, out_size, 4, stride=2, pad=1,
                    initialW=W, nobias=True)
                
                else:
                    conv = L.Convolution2D(
                        out_size, out_size, 3, stride=1, pad=1,
                        initialW=W, nobias=True)
                    
                # Add link
                self.add_link(conv)
                conv.name = "conv{}".format(layer)
                    
                bn = L.BatchNormalization(out_size)
                self.add_link(bn)
                bn.name = "bn{}".format(layer)
                
                    
    def __call__(self, x):
        for link in self.children():
            x_ = link(x)
            if "conv" in link.name:
                x = x_
            elif "bn" in link.name:
                x = F.relu(x_)
            
        return x


class PBR_CVPR2017_mod(chainer.Chain):
    
    def __init__(self, out_size):
        W = chainer.initializers.HeNormal()
        super(PBR_CVPR2017_mod, self).__init__()
        
        with self.init_scope():
            #encoder
            self.enc1 = EncoderBlock_wopool(None, 64, 3)
            self.enc2 = EncoderBlock_wopool(64, 128, 3)
            self.enc3 = EncoderBlock_wopool(128, 256, 4)
            self.enc4 = EncoderBlock_wopool(256, 512, 4)
            
            #middle
            self.conv1 = L.Convolution2D(
                    512, 512, 3, stride=1, pad=1,
                    initialW=W, nobias=True)
            self.bn1 = L.BatchNormalization(512)
            
            #decoder
            self.dec4 = DecoderBlock_wopool(512, 256, 4)
            self.dec3 = DecoderBlock_wopool(256, 128, 4)
            self.dec2 = DecoderBlock_wopool(128, 64, 3)
            self.dec1 = DecoderBlock_wopool(64, 64, 3)
            
            self.conv2 = L.Convolution2D(
                    64, out_size, 3, stride=1, pad=1,
                    initialW=W, nobias=True)
            
    def __call__(self, x):
        #encode
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h = self.enc4(h3)
        
        #middle layer
        h = F.relu(self.bn1(self.conv1(h)))
        
        #decode
        h = self.dec4(h)
        h += h3
        h = self.dec3(h)
        h += h2
        h = self.dec2(h)
        h += h1
        h = self.dec1(h)
        
        #last output
        h = self.conv2(h)
        
        return h
    

class PBR_CVPR2017_mod_deep(chainer.Chain):
    
    def __init__(self, out_size):
        W = chainer.initializers.HeNormal()
        super().__init__()
        
        with self.init_scope():
            #encoder
            self.enc1 = EncoderBlock_wopool(None, 64, 3)
            self.enc2 = EncoderBlock_wopool(64, 128, 3)
            self.enc3 = EncoderBlock_wopool(128, 256, 4)
            self.enc4 = EncoderBlock_wopool(256, 512, 4)
            self.enc5 = EncoderBlock_wopool(512, 1024, 4)
            
            #middle
            self.conv1 = L.Convolution2D(
                    1024, 1024, 3, stride=1, pad=1,
                    initialW=W, nobias=True)
            self.bn1 = L.BatchNormalization(1024)
            
            #decoder
            self.dec5 = DecoderBlock_wopool(1024, 512, 4)
            self.dec4 = DecoderBlock_wopool(512, 256, 4)
            self.dec3 = DecoderBlock_wopool(256, 128, 4)
            self.dec2 = DecoderBlock_wopool(128, 64, 3)
            self.dec1 = DecoderBlock_wopool(64, 64, 3)
            
            self.conv2 = L.Convolution2D(
                    64, out_size, 3, stride=1, pad=1,
                    initialW=W, nobias=True)
            
    def __call__(self, x):
        #encode
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h4 = self.enc4(h3)
        h  = self.enc5(h4) 
        
        #middle layer
        h = F.relu(self.bn1(self.conv1(h)))
        
        #decode
        h = self.dec5(h)
        h += h4
        h = self.dec4(h)
        h += h3
        h = self.dec3(h)
        h += h2
        h = self.dec2(h)
        h += h1
        h = self.dec1(h)
        
        #last output
        h = self.conv2(h)
        
        return h
    
    
    
class PBR_CVPR2017_mod_concat(chainer.Chain):
    
    def __init__(self, out_size):
        W = chainer.initializers.HeNormal()
        super().__init__()
        
        with self.init_scope():
            #encoder
            self.enc1 = EncoderBlock_wopool(None, 64, 3)
            self.enc2 = EncoderBlock_wopool(64, 128, 3)
            self.enc3 = EncoderBlock_wopool(128, 256, 4)
            self.enc4 = EncoderBlock_wopool(256, 512, 4)
            
            #middle
            self.conv1 = L.Convolution2D(
                    512, 512, 3, stride=1, pad=1,
                    initialW=W, nobias=True)
            self.bn1 = L.BatchNormalization(512)
            
            #decoder
            self.dec4 = DecoderBlock_wopool(512, 256, 4)
            self.dec3 = DecoderBlock_wopool(512, 128, 4)
            self.dec2 = DecoderBlock_wopool(256, 64, 3)
            self.dec1 = DecoderBlock_wopool(128, 64, 3)
            
            self.conv2 = L.Convolution2D(
                    64, out_size, 3, stride=1, pad=1,
                    initialW=W, nobias=True)
            
    def __call__(self, x):
        #encode
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h = self.enc4(h3)
        
        #middle layer
        h = F.relu(self.bn1(self.conv1(h)))
        
        #decode
        h = self.dec4(h)
        h = F.concat((h, h3), axis=1)
        h = self.dec3(h)
        h = F.concat((h, h2), axis=1)
        h = self.dec2(h)
        h = F.concat((h, h1), axis=1)
        h = self.dec1(h)
        
        #last output
        h = self.conv2(h)
        
        return h


class PBR_CVPR2017(chainer.Chain):
    def __init__(self, out_size):
        W = chainer.initializers.HeNormal()
        super(PBR_CVPR2017, self).__init__()

        with self.init_scope():
            #encoder
            self.enc1 = EncoderBlock(None, 64, 2)
            self.enc2 = EncoderBlock(64, 128, 2)
            self.enc3 = EncoderBlock(128, 256, 3)
            self.enc4 = EncoderBlock(256, 512, 3)
            
            #middle
            self.conv1 = L.Convolution2D(
                    512, 512, 3, stride=1, pad=1,
                    initialW=W, nobias=True)
            self.bn1 = L.BatchNormalization(512)
            
            #decoder
            self.dec4 = DecoderBlock(512, 256, 3)
            self.dec3 = DecoderBlock(256, 128, 3)
            self.dec2 = DecoderBlock(128, 64, 2)
            self.dec1 = DecoderBlock(64, 64, 2)
            
            self.conv2 = L.Convolution2D(
                    64, out_size, 3, stride=1, pad=1,
                    initialW=W, nobias=True)

    def __call__(self, x):
        #encode
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h = self.enc4(h3)
        
        #middle layer
        h = F.relu(self.bn1(self.conv1(h)))
        
        #decode
        h = self.dec4(h)
        h += h3
        h = self.dec3(h)
        h += h2
        h = self.dec2(h)
        h += h1
        h = self.dec1(h)
        
        #last output
        h = self.conv2(h)
        
        return h