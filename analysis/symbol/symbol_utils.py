import sys
import os
import mxnet as mx
sys.path.append(os.path.join(os.path.abspath('..'), 'recognition'))
from config import config

def Conv(**kwargs):
    #name = kwargs.get('name')
    #_weight = mx.symbol.Variable(name+'_weight')
    #_bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
    #body = mx.sym.Convolution(weight = _weight, bias = _bias, **kwargs)
    body = mx.sym.Convolution(**kwargs)
    return body

def Act(data, act_type, name):
    #ignore param act_type, set it in this function 
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    else:
      body = mx.sym.Activation(data=data, act_type=act_type, name=name)
    return body

bn_mom = config.bn_mom

def residual_unit_v3(data, num_filter, stride, dim_match, name, **kwargs):
    
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    #print('in unit3')
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    conv1 = Conv(data=bn1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                  no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act1 = Act(data=bn2, act_type=config.net_act, name=name + '_relu1')
    conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                  no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

    if dim_match:
        shortcut = data
    else:
        conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                        workspace=workspace, name=name+'_conv1sc')
        shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return bn3 + shortcut

def residual_unit_v1l(data, num_filter, stride, dim_match, name, bottle_neck):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    workspace = config.workspace
    bn_mom = config.bn_mom
    memonger = False
    use_se = config.net_se
    act_type = config.net_act
    #print('in unit1')
    if bottle_neck:
        conv1 = Conv(data=data, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
        conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn3 = mx.symbol.broadcast_mul(bn3, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return Act(data=bn3 + shortcut, act_type=act_type, name=name + '_relu3')
    else:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn2, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn2 = mx.symbol.broadcast_mul(bn2, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return Act(data=bn2 + shortcut, act_type=act_type, name=name + '_relu3')

def get_head(data, version_input, num_filter):
    bn_mom = config.bn_mom
    workspace = config.workspace
    kwargs = {'bn_mom': bn_mom, 'workspace' : workspace}
    data = data-127.5
    data = data*0.0078125
    #data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if version_input==0:
      body = Conv(data=data, num_filter=num_filter, kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                no_bias=True, name="conv0", workspace=workspace)
      body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
      body = Act(data=body, act_type=config.net_act, name='relu0')
      body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    else:
      body = data
      _num_filter = min(num_filter, 64)
      body = Conv(data=body, num_filter=_num_filter, kernel=(3,3), stride=(1,1), pad=(1, 1),
                                no_bias=True, name="conv0", workspace=workspace)
      body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
      body = Act(data=body, act_type=config.net_act, name='relu0')
      #body = residual_unit_v3(body, _num_filter, (2, 2), False, name='head', **kwargs)
      body = residual_unit_v1l(body, _num_filter, (2, 2), False, name='head', bottle_neck=False)
    return body


