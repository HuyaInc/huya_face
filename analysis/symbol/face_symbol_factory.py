import sys
import os
import numpy as np
import mxnet as mx
import importlib

import memonger

sys.path.append(os.path.join(os.path.abspath('..'), 'recognition'))
from config import config

bn_mom = config.bn_mom

def import_net(net_name):
    """Helper function to import module"""
    return importlib.import_module(net_name)

def bn_relu_conv(bottom, ks, nout, stride, pad, name):
    bn = mx.symbol.BatchNorm(data=bottom, eps=2e-5, momentum=bn_mom, name="{}_bn".format(name))
    relu = mx.symbol.Activation(data=bn, act_type="relu", name="{}_relu".format(name))
    conv = mx.symbol.Convolution(data=relu, kernel=(ks,ks), pad=(pad,pad), \
              stride=(stride,stride), num_filter=nout, name="{}_conv".format(name))
    return conv

def bn_relu_pool_conv(bottom, nout, name):
    bn = mx.symbol.BatchNorm(data=bottom, eps=2e-5, momentum=bn_mom, name="{}_bn".format(name))
    relu = mx.symbol.Activation(data=bn, act_type="relu", name="{}_relu".format(name))
    pool = mx.symbol.Pooling(data=relu, global_pool=True, kernel=(7, 7), pool_type='avg', name='{}_pool'.format(name))
    conv = mx.symbol.Convolution(data=pool, kernel=(1,1), pad=(0,0), stride=(1,1),
            num_filter=nout, name="{}_k_1_conv".format(name))
    return conv

def pool_kpt(bottom, nout, name):
    pool = mx.symbol.Pooling(data=bottom, global_pool=True, kernel=(7, 7), pool_type='avg', name='{}_pool'.format(name))
    conv = mx.symbol.Convolution(data=pool, kernel=(1,1), pad=(0,0), stride=(1,1),
            num_filter=nout, name="{}_k_1_conv".format(name))
    return conv




def get_kpt_feat(body):
    kpt_feat = pool_kpt(body, 68*2, 'kpt_block')
    return kpt_feat


def get_fc1(last_conv, feat_dim, fc_type, name_prefix, input_channel=512):
    body = last_conv
    if fc_type=='Z':
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=True)
        body = mx.symbol.Dropout(data=body, p=0.4)
        fc1 = body
    elif fc_type=='E_Nodp':
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='{}bn1'.format(name_prefix), cudnn_off=True)
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=feat_dim, name='{}pre_fc1'.format(name_prefix))
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='{}fc1'.format(name_prefix), cudnn_off=True)
    elif fc_type=='E':
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='{}bn1'.format(name_prefix), cudnn_off=True)
        body = mx.symbol.Dropout(data=body, p=0.4)
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=feat_dim, name='{}pre_fc1'.format(name_prefix))
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='{}fc1'.format(name_prefix), cudnn_off=True)
    elif fc_type=='EPoolingDropout':
        body = mx.sym.Pooling(body, pool_type='max', global_pool=True)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='{}bn1'.format(name_prefix), cudnn_off=True)
        body = mx.symbol.Dropout(data=body, p=0.4)
        body = mx.symbol.Flatten(body)
        fc1 = mx.sym.BatchNorm(data=body, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='{}fc1'.format(name_prefix), cudnn_off=True)

    elif fc_type=='E_Pooling':
        body_bn = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='{}bn1'.format(name_prefix), cudnn_off=True)
        body_out = mx.symbol.Dropout(data=body_bn, p=0.4)
        e_fc1 = mx.sym.FullyConnected(data=body_out, num_hidden=feat_dim, name='{}pre_fc1'.format(name_prefix))

        pool = mx.sym.Pooling(body, pool_type='avg', global_pool=True)
        pool_fc1 = mx.sym.Flatten(data=pool, name='pre_flatten')

        fc1 = pool_fc1 + e_fc1
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='{}fc1'.format(name_prefix), cudnn_off=True)

    elif fc_type == 'E_kpt_bn':
        bn_body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=True)

        ## branch for cls feat
        drop_fc = mx.symbol.Dropout(data=bn_body, p=0.4)
        fc1_feat = mx.sym.FullyConnected(data=drop_fc, num_hidden=feat_dim, name='pre_fc1')
        fc1_feat = mx.sym.BatchNorm(data=fc1_feat, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=True)

        ## branch for loc landmarks
        kpt_pred = mx.sym.FullyConnected(data=fc1_feat, num_hidden=68*2, name='kpt_fc')
        kpt_pred = mx.sym.BatchNorm(data=kpt_pred, eps=2e-5, name='kpt_pred')
        
        fc1 = [fc1_feat, kpt_pred]
    elif fc_type == 'E_kpt_norm':
        bn_body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=True)

        ## branch for cls feat
        drop_fc = mx.sym.Dropout(data=bn_body, p=0.4)
        fc1_feat = mx.sym.FullyConnected(data=drop_fc, num_hidden=feat_dim, name='pre_fc1')
        fc1_feat = mx.sym.BatchNorm(data=fc1_feat, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=True)

        ## branch for loc landmarks
        kpt_weight = mx.sym.Variable("kpt_weight" , shape=(68*2, feat_dim))
        kpt_weight = mx.sym.L2Normalization(kpt_weight, mode='instance')

        nfc1_kpt_feat = mx.sym.L2Normalization(fc1_feat, mode='instance', name='kpt_norm_fc')
        kpt_pred = mx.sym.FullyConnected(data=nfc1_kpt_feat, weight=kpt_weight, no_bias=True, num_hidden=68*2, name='kpt_pred')
        kpt_pred = 0.5 * kpt_pred + 0.5
        
        fc1 = [fc1_feat, kpt_pred]
    elif fc_type=='FC':
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=True)
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=feat_dim, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=True)
    elif fc_type=='GAP':
        bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=True)
        relu1 = Act(data=bn1, act_type=config.net_act, name='relu1')
        # Although kernel is not used here when global_pool=True, we should put one
        pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
        flat = mx.sym.Flatten(data=pool1)
        fc1 = mx.sym.FullyConnected(data=flat, num_hidden=feat_dim, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=True)
    elif fc_type=='GNAP': #mobilefacenet++
        filters_in = 512 # param in mobilefacenet
        if feat_dim > filters_in:
            body = mx.sym.Convolution(data=last_conv, num_filter=feat_dim, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True, name='convx')
            body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=0.9, name='convx_bn', cudnn_off=True)
            body = Act(data=body, act_type=config.net_act, name='convx_relu')
            filters_in = feat_dim
        else:
            body = last_conv
        body = mx.sym.BatchNorm(data=body, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn6f', cudnn_off=True)  
  
        spatial_norm=body*body
        spatial_norm=mx.sym.sum(data=spatial_norm, axis=1, keepdims=True)
        spatial_sqrt=mx.sym.sqrt(spatial_norm)
        #spatial_mean=mx.sym.mean(spatial_sqrt, axis=(1,2,3), keepdims=True)
        spatial_mean=mx.sym.mean(spatial_sqrt)
        spatial_div_inverse=mx.sym.broadcast_div(spatial_mean, spatial_sqrt)
  
        spatial_attention_inverse=mx.symbol.tile(spatial_div_inverse, reps=(1,filters_in,1,1))   
        body=body*spatial_attention_inverse
        #body = mx.sym.broadcast_mul(body, spatial_div_inverse)
  
        fc1 = mx.sym.Pooling(body, kernel=(7, 7), global_pool=True, pool_type='avg')
        if feat_dim<filters_in:
          fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn6w', cudnn_off=True)
          fc1 = mx.sym.FullyConnected(data=fc1, num_hidden=feat_dim, name='pre_fc1')
        else:
          fc1 = mx.sym.Flatten(data=fc1)
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=0.9, name='fc1', cudnn_off=True)
    elif fc_type=="GDC": #mobilefacenet_v1
        def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
            conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
            bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=bn_mom)    
            return bn

        conv_6_dw = Linear(last_conv, num_filter=input_channel, num_group=input_channel, kernel=(7,7), pad=(0, 0), stride=(1, 1), name="conv_6dw7_7")  
        conv_6_f = mx.sym.FullyConnected(data=conv_6_dw, num_hidden=feat_dim, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=conv_6_f, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=True)
    elif fc_type=='F':
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=True)
        body = mx.symbol.Dropout(data=body, p=0.4)
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=feat_dim, name='fc1')
    elif fc_type=='G':
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=True)
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=feat_dim, name='fc1')
    elif fc_type=='H':
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=feat_dim, name='fc1')
    elif fc_type=='I':
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=True)
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=feat_dim, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=True)
    elif fc_type=='J':
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=feat_dim, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=True)
    elif fc_type == 'Backbone_Flatten':
        body = mx.sym.Pooling(body, pool_type='avg', global_pool=True)
        fc1 = mx.sym.Flatten(data=body, name='fc1')
    elif fc_type == 'GPBN':
        body = mx.sym.Pooling(body, pool_type='avg', global_pool=True)
        flat = mx.sym.Flatten(data=body, name='pre_flatten')
        fc1 = mx.sym.BatchNorm(data=flat, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=True)
    elif fc_type == 'GPBN_kpt':
        body = mx.sym.Pooling(body, pool_type='avg', global_pool=True)
        flat = mx.sym.Flatten(data=body, name='pre_flatten')
        fc1_feat = mx.sym.BatchNorm(data=flat, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=True)

        kpt_pred = mx.sym.FullyConnected(data=fc1_feat, num_hidden=68*2, name='kpt_pred')
        fc1 = [fc1_feat, kpt_pred]
    else:
        assert False
    return fc1

def infer_net_shapes(net,input_size=300):
    '''
    infert net shapes
    '''
    for inter in net.get_internals():
        #print('%s is %s' % (inter.name,inter.list_arguments()))
        if 'data' in inter.list_arguments():
            _,out_shape,_ = inter.infer_shape(data=(1,3,input_size,input_size))
            print('%s shape is %s'  % (inter.name, out_shape))

def infer_decode_shape(net, feat_dim=512):
    for inter in net.get_internals():
        #print('%s is %s' % (inter.name,inter.list_arguments()))
        if 'data' in inter.list_arguments():
            _,out_shape,_ = inter.infer_shape(data=(1,feat_dim))
            print('%s shape is %s'  % (inter.name, out_shape))

def infer_decode_em_shape(net, **kwargs):
    for inter in net.get_internals():
        #print('%s is %s' % (inter.name,inter.list_arguments()))
        if 'data' in inter.list_arguments():
            img_shape = (1, 3, 112, 112)
            _,out_shape,_ = inter.infer_shape(data=img_shape, **kwargs)
            print('%s shape is %s'  % (inter.name, out_shape))


def get_decode_img(emb_size, decode_num_layer=18, for_face=True, decode_channel=3):
    from config import config
    net_name = 'fgan_resnet'
    config.num_layers = decode_num_layer
    decode_name_prefix = 'decode'
    decode_net = import_net(net_name)

    feat = mx.sym.Variable(name='data')

    feat = mx.sym.reshape(feat, (-1, emb_size, 1, 1))
    feat = mx.sym.L2Normalization(feat, mode='instance', name='%s_feat_in' % decode_name_prefix)

    decode_img = decode_net.get_symbol(feat, for_face, decode_channel, name_prefix=decode_name_prefix)
    decode_img = mx.sym.clip(decode_img, 0, 1, name='%s_clip_img' % decode_name_prefix)

    if config.memonger:
        dshape = (config.per_batch_size, config.emb_size)
        net_mem_planned = memonger.search_plan(decode_img, data=dshape)
        old_cost = memonger.get_cost(decode_img, data=dshape)
        new_cost = memonger.get_cost(net_mem_planned, data=dshape)

        print('Gan restore dec Old feature map cost=%d MB' % old_cost)
        print('Gan restore dec New feature map cost=%d MB' % new_cost)
        decode_img = net_mem_planned

    return decode_img


def get_descriminator_embedding_sym(config, name_prefix, epsilon=1e-20):
    decode_img = mx.sym.Variable(name='data')

    dec_cls_net_mul = 2

    decode_cls_net_input = decode_img

    image_size = config.image_shape
    image_size = image_size[0] * image_size[1] * image_size[2]

    decode_mean = mx.sym.mean(decode_cls_net_input, axis=(1,2,3), keepdims=True)
    decode_mean_diff = mx.sym.broadcast_sub(decode_cls_net_input, decode_mean)
    decode_var = mx.sym.sum(mx.sym.square(decode_mean_diff), axis=(1,2,3), keepdims=True)/image_size
    decode_norm_img = mx.sym.broadcast_div(decode_mean_diff, decode_var + epsilon)
    
    decode_cls_net = import_net(config.net_name)
    print("==dec cls net ", decode_cls_net)
    backbone_conv = decode_cls_net.get_symbol(decode_norm_img)
    body = mx.sym.BatchNorm(data=backbone_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='%sbn1' % name_prefix, cudnn_off=True)
    body = mx.symbol.Dropout(data=body, p=0.4)
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=512, name='%spre_fc1' % name_prefix)
    embedding = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='%sfc1' % name_prefix, cudnn_off=True)

    
    if config.memonger:
        dshape = (dec_cls_net_mul*config.per_batch_size, config.image_shape[2], config.image_shape[0], config.image_shape[1])
        net_mem_planned = memonger.search_plan(embedding, data=dshape)
        old_cost = memonger.get_cost(embedding, data=dshape)
        new_cost = memonger.get_cost(net_mem_planned, data=dshape)

        print('Dec cls Old feature map cost=%d MB' % old_cost)
        print('Dec cls New feature map cost=%d MB' % new_cost)
        embedding = net_mem_planned

    return embedding

def get_symbol_embedding(config, with_label_sym=True):
    data = mx.sym.Variable(name='data')

    if config.net_input == 0:
        assert False
        #data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data', cudnn_off=True)
        data = mx.sym.identity(data=data, name='id')
        data = data-127.5
        data = data*0.0078125
    elif config.net_input == 2:
        print("========= net input 2 instance norm input data =======")
        epsilon = 1e-20
        data = mx.sym.identity(data=data, name='id')
        instance_mean = mx.sym.mean(data, axis=(1,2,3), keepdims=True) # gray image input
        mean_diff = mx.sym.broadcast_sub(data, instance_mean)
        data_var = mx.sym.sum(mx.sym.square(mean_diff), axis=(1,2,3), keepdims=True) / (112*112*3)
        data = mx.sym.broadcast_div(mean_diff, data_var+epsilon)
    elif config.net_input == 10:
        data = mx.sym.identity(data=data, name='id')
        data = mx.sym.mean(data, axis=1, keepdims=True) # gray image input
        data = data-127.5
        data = data*0.0078125
    else:
        data = mx.sym.identity(data=data, name='id')
        data = data-127.5
        data = data*0.0078125

    net = import_net(config.net_name)
    backbone_conv = net.get_symbol(data)
    embedding = get_fc1(backbone_conv, config.emb_size, config.net_output, config.name_prefix)

    if config.memonger:
        dshape = (config.per_batch_size, config.image_shape[2], config.image_shape[0], config.image_shape[1])
        net_mem_planned = memonger.search_plan(embedding, data=dshape)
        old_cost = memonger.get_cost(embedding, data=dshape)
        new_cost = memonger.get_cost(net_mem_planned, data=dshape)

        print('Old feature map cost=%d MB' % old_cost)
        print('New feature map cost=%d MB' % new_cost)
        embedding = net_mem_planned
    #infer_net_shapes(embedding, input_size=32)

    if isinstance(embedding, list):
        out_list = embedding
    else:
        out_list = [embedding]

    if with_label_sym:
        all_label = mx.symbol.Variable('softmax_label')
        all_label = mx.symbol.BlockGrad(all_label)
        out_list.append(all_label)
    out = mx.symbol.Group(out_list)
    return out

def phi_theta_fun(theta):
    phi_theta = 3 * mx.sym.cos(0.5*theta - 1.5*np.pi) + 1
    return phi_theta

def phi_linear(theta):
    phi_theta = -(1+2 * np.cos(0.5))/np.pi * theta + np.cos(0.5) 
    return phi_theta

def phi_large_linear(theta):
    phi_theta = -(1+2 * np.cos(0.5))/np.pi * theta + np.cos(np.pi/4) 
    return phi_theta

def phi_linear_smooth(theta):
    phi_theta = -0.7* theta + 0.6
    return phi_theta

def phi_cos(theta):
    cos_theta = mx.sym.cos(theta)
    return cos_theta


def get_symbol_arcface(args, config, name_prefix=config.name_prefix, cls_num=None):
    if cls_num is None:
        cls_num = args.ctx_num_classes

    embedding = mx.symbol.Variable('data')
    gt_label = mx.symbol.Variable('softmax_label')
    #print('call get_sym_arcface with', args, config)
    #print(name_prefix, args._ctxid)
    _weight = mx.symbol.Variable("%sfc7_%d_weight" % (name_prefix, args._ctxid), \
            shape=(cls_num, config.emb_size),  \
            lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult)	

    out_list = []

    if config.loss_name == 'feat_mom_phi':
        ## compute margin loss
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, \
                                               mode='instance', name='%sfc1n_%d' % (name_prefix, args._ctxid))
        fc7 = mx.sym.FullyConnected(data=nembedding, \
                                    weight=_weight, no_bias=True, \
                                    num_hidden=cls_num, \
                                    name='%sfc7_%d' % (name_prefix, args._ctxid))

        out_list.append(fc7)
    elif config.loss_name == 'theta_phi':
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, \
                                               mode='instance', name='%sfc1n_%d' % (name_prefix, args._ctxid))
        fc7 = mx.sym.FullyConnected(data=nembedding, \
                                    weight=_weight, no_bias=True, \
                                    num_hidden=cls_num, \
                                    name='%sfc7_%d' % (name_prefix, args._ctxid))
        cos_theta = mx.sym.clip(fc7, -1.0, 1.0)
        theta = mx.sym.arccos(cos_theta)
        out_list.append(theta)
    else:
        assert False

    out = mx.symbol.Group(out_list)
    return out

def test_fc():
    from config import config
    config.net_name = 'fresnet'
    config.num_layers = 100

    net = import_net(config.net_name)
    data = mx.sym.Variable(name='data')

    backbone_conv = net.get_symbol(data, name_prefix='encode_')
    fc1 = get_fc1(backbone_conv, 512, 'E', input_channel=512)
    #fc1 = mx.sym.Group(fc1)

    infer_net_shapes(fc1, 112)

def test_gan():
    feat_dim = 512
    decode_img = get_decode_img(feat_dim, decode_num_layer=18, for_face=True, decode_channel=3)
    infer_decode_shape(decode_img, feat_dim=feat_dim)

def test_dec_cls():
    from config import config
    config.net_name = 'fresnet'
    config.num_layers = 18
    config.per_batch_size = 64
    config.image_shape = (3, 112, 112)
    decode_embedding = get_descriminator_embedding_sym(config, name_prefix='dec_cls_')
    infer_decode_em_shape(decode_embedding)#, input_data=(1,3,112,112))


if __name__ == '__main__':
    #test_fc()
    #test_gan()
    test_dec_cls()
    
