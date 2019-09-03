# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 11:23:26 2018

@author: 0000145046
"""


import numpy as np

from chainer import function_node
from chainer.backends import cuda
import chainer.functions
from chainer.utils import type_check


class MaskedMeanSquaredError(function_node.FunctionNode):
    
    """Mean squared error without mask area"""
    
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[2].dtype == np.float32,
            in_types[0].shape == in_types[1].shape == in_types[2].shape
            )
        
    def forward(self, inputs):
        self.retain_inputs((0, 1, 2))
        
        # x: output, t: label, mask: mask
        x, t, mask = inputs[0], inputs[1], inputs[2] 
        
        diff = x - t
        
        # remove mask area
        diff = diff[mask.astype(bool)]
        self.size = diff.dtype.type(diff.size)
        
        return diff.dot(diff) / self.size,
    
    def backward(self, indexes, gy):
        # If you forget what is indexes, please refer below.
        # https://docs.chainer.org/en/stable/reference/generated/chainer.FunctionNode.html
        x, t, mask = self.get_retained_inputs()
        
        #xp = cuda.get_array_module(*mask)
        #print(xp.max(mask.data))
        
        ret = []
        diff = x - t
        
        gy0 = chainer.functions.broadcast_to(gy[0], diff.shape)
        gx0 = gy0 * diff * (2. / self.size) * mask
        
        if 0 in indexes:
            ret.append(gx0)
        if 1 in indexes:
            ret.append(-gx0)
            
        return ret

    
def masked_mean_squared_error(x, t, mask):
    """Mean squared error function.
    This function computes mean squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.
    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        t (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        mask (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
    Returns:
        ~chainer.Variable:
            A variable holding an array representing the mean squared
            error of two inputs without masked area.
    """
    return MaskedMeanSquaredError().apply((x, t, mask))[0]


class MaskedMeanSquaredError_difspeWeight(function_node.FunctionNode):
    
    def __init__(self, specular_weight):
        self.speweight = specular_weight
    
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[2].dtype == np.float32,
            in_types[0].shape == in_types[1].shape == in_types[2].shape
            )
        
    def forward(self, inputs):
        self.retain_inputs((0, 1, 2))
        
        # x: output, t: label, mask: mask
        x, t, mask = inputs[0], inputs[1], inputs[2]
        
        diff = x - t
        
        diff_spe = diff[:,:3,:,:]
        diff_dif = diff[:,3:,:,:]
        
        # remove mask area
        mask = mask[:,:3,:,:]
        diff_spe = diff_spe[mask.astype(bool)]
        diff_dif = diff_dif[mask.astype(bool)]
        self.size = diff_spe.dtype.type(diff_spe.size)
        
        return (diff_spe.dot(diff_spe)*self.speweight + diff_dif.dot(diff_dif)) / self.size,
    
    def backward(self, indexes, gy):
        # If you forget what is indexes, please refer below.
        # https://docs.chainer.org/en/stable/reference/generated/chainer.FunctionNode.html
        x, t, mask = self.get_retained_inputs()
        x, t = x.data, t.data
        
        xp = cuda.get_array_module(*mask)
        
        diff = xp.zeros_like(x)
        
        diff[:,:3,:,:] = self.speweight * (x[:,:3,:,:] - t[:,:3,:,:])
        diff[:,3:,:,:] = x[:,3:,:,:] - t[:,3:,:,:]
        
        ret = []
        gy0 = chainer.functions.broadcast_to(gy[0], diff.shape)
        gx0 = gy0 * diff * (2. / self.size) * mask
        
        if 0 in indexes:
            ret.append(gx0)
        if 1 in indexes:
            ret.append(-gx0)
            
        return ret
    

def masked_mean_squared_error_difspeWeight(x, t, mask, specular_weight=1.2):
    return MaskedMeanSquaredError_difspeWeight(specular_weight=specular_weight).apply((x, t, mask))[0]


class Masked_MSEnGradError(function_node.FunctionNode):
    
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[2].dtype == np.float32,
            in_types[0].shape == in_types[1].shape == in_types[2].shape
            )
        
    def my_strided(self, x, shape=None, strides=None):
        xp = cuda.get_array_module(*x)
        
        shape = x.shape if shape is None else tuple(shape)
        strides = x.strides if strides is None else tuple(strides)
    
        return xp.ndarray(shape=shape, dtype=x.dtype,
                            memptr=x.data, strides=strides)
        
        
    def my_convolution2d(self, img, kernel):
        # https://qiita.com/secang0/items/f3a3ff629988dc660d87
        xp = cuda.get_array_module(*img)
        
        sub_shape = tuple(np.subtract(img.shape, kernel.shape) + 1)
        submatrices = self.my_strided(img,kernel.shape + sub_shape,img.strides * 2)
        
        convolved_matrix = xp.einsum('ij,ijkl->kl', kernel, submatrices)
        return convolved_matrix
        
        
    def laplacian(self, img):
        """
        In :
            ndarray or cdarray which dimensions are in (bsize, ch, h, w).
        Out:
            gradient of input image which dimensions are
            also in (bsize, ch, h, w).
        """
        xp = cuda.get_array_module(*img)

        dst = xp.zeros_like(img)
        
        kernel = xp.array([[0,1,0],
                          [1,-4,1],
                          [0,1,0]], dtype=xp.float32)
        
        for bc in range(img.shape[0]): # for each batch
            for ch in range(img.shape[1]):
                img_bc = img[bc,ch,:,:]  #(h, w)
                dst[bc,ch,1:-1,1:-1] = self.my_convolution2d(img_bc, kernel)
            
        return dst

        
    def forward(self, inputs):
        self.retain_inputs((0, 1, 2))
        
        # x: output, t: label, mask: mask
        x, t, mask = inputs[0], inputs[1], inputs[2] 
        
        self.x_grad = self.laplacian(x)
        self.t_grad = self.laplacian(t)
        
        diff = x - t
        diff_grad = self.x_grad - self.t_grad
        
        diff = diff[mask.astype(bool)]
        diff_grad = diff_grad[mask.astype(bool)]
        
        self.size = diff.dtype.type(diff.size)
        
        return (diff.dot(diff) + diff_grad.dot(diff_grad)) / self.size,
    
    def backward(self, indexes, gy):
        # If you forget what is indexes, please refer below.
        # https://docs.chainer.org/en/stable/reference/generated/chainer.FunctionNode.html
        x, t, mask = self.get_retained_inputs()
        
        ret = []
        
        diff = x - t
        diff_grad = self.x_grad - self.t_grad
        
        gy0 = chainer.functions.broadcast_to(gy[0], diff.shape)
        
        gx0 = gy0 * (diff -4*diff_grad) * (2. / self.size) * mask
        
        if 0 in indexes:
            ret.append(gx0)
        if 1 in indexes:
            ret.append(-gx0)
            
        return ret

def masked_MSE_gradient_error(x, t, mask):
    """Mean squared error function.
    This function computes mean squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.
    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        t (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        mask (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
    Returns:
        ~chainer.Variable:
            A variable holding an array representing the mean squared
            error of two inputs without masked area.
    """
    return Masked_MSEnGradError().apply((x, t, mask))[0]
        


class MaskedDotProductError(function_node.FunctionNode):
    
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[2].dtype == np.float32,
            in_types[0].shape == in_types[1].shape == in_types[2].shape
            )
        
    def forward(self, inputs):
        self.retain_inputs((0,1,2))
        xp = cuda.get_array_module(*inputs)
        
        # x: output, t: label, mask: mask
        x, t, mask = inputs[0], inputs[1], inputs[2] # (b_size, ch, w, h)
        
        coserr = 1 - xp.sum(x*t, axis=1) # (b_size, w, h)
        mask = mask[:,0,:,:] # (b_size, w, h)
        
        # remove mask area
        coserr = coserr[mask.astype(bool)]
        self.size = coserr.dtype.type(coserr.size)
        
        return xp.sum(coserr) / self.size,
        
    
    def backward(self, indexes, gy):
        x, t, mask = self.get_retained_inputs()
        
        gy = chainer.functions.broadcast_to(gy[0], x.shape)
        
        ret = []
        if 0 in indexes:
            ret.append(-(gy*t*mask) / self.size)
        if 1 in indexes:
            ret.append(-(gy*x*mask) / self.size)
        
        return ret
    
def masked_dot_product_error(x, t, mask):
    """Mean squared error function.
    This function computes mean squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.
    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        t (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        mask (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
    Returns:
        ~chainer.Variable:
            A variable holding an array representing the dot product
            error of two inputs without masked area.
    """
    return MaskedDotProductError().apply((x, t, mask))[0]
