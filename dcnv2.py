import math
from torch import nn
from torch.autograd import Function
import torch
import numpy as np
import ModulatedDeformableConvolution

torch.manual_seed(42)


class DCNFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, Offset, Mask,
                num_deformable_group,kernel, pad, stride, dilation,
                use_bias , bias):
        """

        :param ctx:
        :param input: N,C,H,W
        :param weights: C`,C*kernel*kernel
        :param bias: C` or None
        :param Offset: N,num_deformable_group*kernel*kernel*2,H,W
        :param Mask: N,num_deformable_group*kernel*kernel,H,W
        :param num_deformable_group: int , default 1
        :param kernel: 3
        :param pad: 1 
        :param stride: 1 
        :param dilation: 1 
        :return: N, C', H, W
        """
        #outputs = ModulatedDeformableConvolution.test(input)
        outputs = ModulatedDeformableConvolution.forward(input, weights ,Offset, Mask,
                                                         num_deformable_group,
                                                         kernel, pad, stride, dilation,
                                                         use_bias ,bias)
        #  outputs : std::vector<at::Tensor> [out,col]
        #    out : N, C', H, W
        #    col : C*kernel*kernel,NHW
        out, col = outputs[:2]

        ctx.kernel = kernel
        ctx.pad = pad
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.num_deformable_group = num_deformable_group
        ctx.use_bias = use_bias
        ctx.save_for_backward(input,weights, bias, Offset, Mask, col)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.contiguous() 
        input, weights, bias, Offset, Mask, col = ctx.saved_tensors
        outputs = ModulatedDeformableConvolution.backward(grad_out,
                                                          input, weights, Offset, Mask, col,
                                                          ctx.num_deformable_group, ctx.kernel, ctx.pad, ctx.stride,
                                                          ctx.dilation,
                                                          ctx.use_bias, bias)
        if ctx.use_bias:
            grad_im, grad_weights, grad_offset, grad_mask , grad_bias = outputs
            return grad_im, grad_weights, grad_offset, grad_mask, \
                   None, None, None, None, None , \
                   None, grad_bias

        else:
            grad_im, grad_weights, grad_offset, grad_mask            = outputs
            return grad_im, grad_weights, grad_offset, grad_mask, \
                   None, None, None, None, None , \
                   None, None


class DCNConv(nn.Module):
    def __init__(self, inplanes, planes, use_bias,
                    num_deformable_group=1,
                    kernel=3, pad=1,stride=1, dilation=1,
                    phase='TRAIN'):
        super(DCNConv, self).__init__()
        self.use_bias = use_bias
        self.num_deformable_group = num_deformable_group
        self.kernel = kernel
        self.pad = pad
        self.stride = stride
        self.dilation = dilation

        assert inplanes%self.num_deformable_group==0
        # Warning : make sure H`/W` output after offset/maskconv  same with H_OUT/W_OUT of deformable conv
        #         SO stride/padding/dilation of offset/maskconv same with deformable conv
        #         kernel_size/groups I think use not smaller/bigger than deformable conv , need experiment
        self.conv2_offset = nn.Conv2d(inplanes,out_channels=self.num_deformable_group*self.kernel*self.kernel*2,
                                      kernel_size=self.kernel,stride=self.stride, padding=self.pad, dilation=self.dilation, bias=True,
                                      groups=self.num_deformable_group)
        self.conv2_mask   = nn.Conv2d(inplanes,out_channels=self.num_deformable_group*self.kernel*self.kernel,
                                      kernel_size=self.kernel,stride=self.stride, padding=self.pad, dilation=self.dilation, bias=True,
                                      groups=self.num_deformable_group)

        # Conv weight / bias
        self.weights = torch.empty(planes,inplanes*kernel*kernel,requires_grad= phase=='TRAIN') # planes,inplanes*3*3
        if self.use_bias: # Resnet Conv Deprecated Bias when Subsequently is BN
            self.bias = torch.empty(planes,requires_grad= phase=='TRAIN') # planes

        self.init_weights()

    def init_weights(self):
        # Important Must Init weight to zero => offset is zero , mask is 1
        nn.init.zeros_(self.conv2_offset.weight)
        if self.conv2_offset.bias is not None:
            nn.init.zeros_(self.conv2_offset.bias)
        nn.init.zeros_(self.conv2_mask.weight)
        if self.conv2_mask.bias is not None:
            nn.init.zeros_(self.conv2_mask.bias)

        # Init Conv Weight and bias
        nn.init.kaiming_normal_(self.weights,mode='fan_in',nonlinearity='relu')
        if self.use_bias:
            nn.init.zeros_(self.bias)


    def forward(self, x):
        offset = self.conv2_offset(x) # N,num_deformable_group*kernel*kernel*2,H,W
        mask = self.conv2_mask(x)     # N,num_deformable_group*kernel*kernel  ,H,W
        mask = 2 * torch.sigmoid(mask)
        out = DCNFunction.apply(x , self.weights, offset, mask,
                                self.num_deformable_group,
                                self.kernel,self.pad,self.stride,self.dilation,
                                self.use_bias , self.bias if self.use_bias else torch.empty(0))

        return out


class DCNPooling(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

if __name__ == '__main__':
    N = 2
    channels = 2
    filters = 3
    H = W = 4
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    myconv=DCNConv(channels,filters,use_bias=False,num_deformable_group=1,
                    kernel=3, pad=2,stride=1, dilation=1)
    myconv.init_weights()
    x=np.array([i for i in range(N*channels*H*W)])
    x=x.reshape(N,channels,H,W)
    x= torch.Tensor(x)
    x.requires_grad=True
    # check forward/backward 
    out = myconv(x)
    _sum = torch.sum(out)
    _sum.backward()
    
    # check numerical and analytical grad
    status = torch.autograd.gradcheck(myconv,[x])
    print("Finish backward")
