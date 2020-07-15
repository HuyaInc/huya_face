import logging
import copy
import time
import os
import sys
import numpy as np
import math
import functools

import mxnet as mx
from mxnet import context as ctx
from mxnet.initializer import Uniform
from mxnet.module.base_module import BaseModule
from mxnet.module.module import Module
from mxnet import metric
from mxnet.model import BatchEndParam
from mxnet import io
import mxnet.ndarray as nd

def parall_log_softmax_ce_loss(datas, device_onehot_labels, ctx):
    ctx_max_list = list(map(lambda fc7_out : mx.nd.max(fc7_out, axis=1, keepdims=True).as_in_context(ctx), datas))
    local_fc7_max = mx.nd.max(nd.concat(*ctx_max_list, dim=1), axis=1, keepdims=True)
    z_list = list(map(lambda fc7_out : fc7_out - local_fc7_max.as_in_context(fc7_out.context), datas))
    
    ctx_exp_sum_list = list(map(lambda z: mx.nd.sum(mx.nd.exp(z), axis=1, keepdims=True).as_in_context(ctx), z_list))
    log_exp_sum = mx.nd.log(mx.nd.add_n(*ctx_exp_sum_list))

    ce_loss_list = [mx.nd.sum((log_exp_sum.as_in_context(z.context) - z) * device_onehot_label)
                            for z, device_onehot_label in zip(z_list, device_onehot_labels)]
    ce_loss = mx.nd.add_n(*[ce.as_in_context(ctx) for ce in ce_loss_list])
    
    return ce_loss

def cls_argmax(datas, ctx):
    assert len(datas) == 1
    return mx.nd.argmax(datas[0], axis=-1)

def parall_argmax(datas, ctx):
    sub_max = mx.nd.concat(*[mx.nd.max(data, axis=-1, keepdims=True).as_in_context(ctx)
                                for data in datas], dim=1)
    sub_arg_max = mx.nd.concat(*[data.shape[1] * i + mx.nd.argmax(data, axis=-1, keepdims=True).as_in_context(ctx)
                                for i, data in enumerate(datas)], dim=1)
    part_arg_max = mx.nd.argmax(sub_max, axis=1)
    return mx.nd.pick(sub_arg_max, part_arg_max, axis=1)

def parall_argmin(datas, ctx):
    sub_min = mx.nd.concat(*[mx.nd.min(data, axis=-1, keepdims=True).as_in_context(ctx)
                                for data in datas], dim=1)
    sub_arg_min = mx.nd.concat(*[data.shape[1] * i + mx.nd.argmin(data, axis=-1, keepdims=True).as_in_context(ctx)
                                for i, data in enumerate(datas)], dim=1)
    part_arg_min = mx.nd.argmin(sub_min, axis=1)
    return mx.nd.pick(sub_arg_min, part_arg_min, axis=1)

def parall_topk_value(datas, ctx, k=2):
    top2_values = mx.nd.concat(*[mx.nd.topk(data, axis=-1, k=k, ret_typ='value').as_in_context(ctx) 
                                for data in datas], dim=1)
    top2_prob = mx.nd.topk(top2_values, axis=-1, k=k, ret_typ='value')
    return top2_prob

def parall_pick_teacher_cos_label(teacher_preds, device_labels, ctx_num_classes, ctx):
    onehot_device_labels = [nd.one_hot(label, depth=ctx_num_classes, 
                                    on_value = 1.0, off_value = 0.0)
                                for label in device_labels]

    teacher_cos_sim_scores = [mx.nd.sum(teacher_pred * device_onehot_label, axis=1, keepdims=True)
                            for teacher_pred, device_onehot_label in zip(teacher_preds, onehot_device_labels)]
    teacher_cos_sim_scores = mx.nd.concat(*[teacher_sim_score.as_in_context(ctx) for teacher_sim_score in teacher_cos_sim_scores], dim=1)
    teacher_cos_sim_scores = mx.nd.sum(teacher_cos_sim_scores, axis=1, keepdims=True)
    return teacher_cos_sim_scores

def parall_topk_index(datas, ctx, k=5):
    topk_sub_values = mx.nd.concat(*[mx.nd.topk(data, k=k, ret_typ='value').as_in_context(ctx)
                                 for data in datas], dim=1)
    topk_sub_index = mx.nd.concat(*[data.shape[1]*i+mx.nd.topk(data, k=k).as_in_context(ctx)
                                 for i, data in enumerate(datas)], dim=1)
    topk_all_index = mx.nd.topk(topk_sub_values, k=k)
    topk_index = mx.nd.concat(*[mx.nd.pick(topk_sub_index, topk_all_index.T[i], axis=1, keepdims=True) for i in range(k)], dim=1)
    return topk_index

def nd_phi_linear(theta):
    phi_theta = -(1+2 * np.cos(0.5))/np.pi * theta + np.cos(0.5) 
    return phi_theta

def nd_phi_linear_plus_n(theta, n):
    phi_theta = -(1+2 * np.cos(0.5))/np.pi * theta + n
    return phi_theta

def nd_phi_linear_slope_n(theta, n):
    phi_theta = -n * theta + 1
    return phi_theta

def nd_phi_cos(theta):
    cos_theta = mx.nd.cos(theta)
    return cos_theta

def nd_arcface_phi(theta):
    phi_theta = mx.nd.cos(theta+0.5)
    return phi_theta

def nd_linear_smooth(theta):
    phi_linear_smooth = -0.7* theta + 0.6
    return phi_linear_smooth

def nd_linear_large(theta):
    phi_theta = -0.876996 * theta + 0.5
    return phi_theta

def penalize_with_cos_psi(fc_pred_datas, onehot_device_labels, phi_fn, loss_s):
    phi_out_list = []
    for y_hat, onehot_device_label in zip(fc_pred_datas, onehot_device_labels):
        onehot_cos_theta = onehot_device_label * y_hat
        cos_theta = mx.nd.clip(onehot_cos_theta, -1.0, 1.0)
        theta = mx.nd.arccos(cos_theta)

        phi_theta  = phi_fn(theta)
        onehot_phi_theta = onehot_device_label * phi_theta

        y_out = loss_s * (y_hat - onehot_cos_theta + onehot_phi_theta)
        phi_out_list.append(y_out)
    return phi_out_list

def penalize_linear_psi(fc_pred_datas, onehot_device_labels, phi_fn, loss_s, slope, margin):
    phi_out_list = []
    for y_hat, onehot_device_label in zip(fc_pred_datas, onehot_device_labels):
        linear_theta =  -slope * y_hat + 1 + margin
        onehot_theta = onehot_device_label * linear_theta

        phi_theta = -slope * y_hat + 1
        onehot_phi_theta = onehot_device_label * phi_theta

        y_out = loss_s * (linear_theta - onehot_theta + onehot_phi_theta)
        phi_out_list.append(y_out)
    return phi_out_list

def cls_log_softmax_ce_loss_fn(datas, device_onehot_labels, ctx):
    assert len(datas) == 1
    assert len(device_onehot_labels) == 1

    fc7_out = datas[0].as_in_context(ctx)
    device_onehot_label = device_onehot_labels[0].as_in_context(ctx)
    fc_max = mx.nd.max(fc7_out, axis=1, keepdims=True)

    z = fc7_out - fc_max
    exp_sum = mx.nd.sum(mx.nd.exp(z), axis=1, keepdims=True)
    log_exp_sum = mx.nd.log(exp_sum)

    ce_loss = (log_exp_sum - z) * device_onehot_label
    ce_loss = mx.nd.sum(ce_loss) 
    
    return ce_loss

def cls_loss_fun(cls_pred_datas, labels, cls_num, ctx, phi_fn, psi_norm_fn, target_fn, loss_s):
    assert len(labels) == 1
    onehot_device_labels = [nd.one_hot(label, depth=cls_num, 
                                    on_value = 1.0, off_value = 0.0)
                                for label in labels]

    phi_datas = psi_norm_fn(cls_pred_datas, onehot_device_labels, phi_fn, loss_s)

    ## check phi pred correct
    phi_pred = target_fn(phi_datas, ctx)
    pred_correct = nd.equal(phi_pred, labels[0])
    
    label_loss = cls_log_softmax_ce_loss_fn(phi_datas, onehot_device_labels, ctx)
    cls_loss = label_loss
    
    return cls_loss, pred_correct

def parall_cls_loss(cls_pred_datas, labels, y_label, ctx, ctx_num_classes, phi_fn, psi_norm_fn, parral_target_fn, loss_s):
    onehot_device_labels = [nd.one_hot(label, depth=ctx_num_classes, 
                                    on_value = 1.0, off_value = 0.0)
                                for label in labels]

    phi_datas = psi_norm_fn(cls_pred_datas, onehot_device_labels, phi_fn, loss_s)

    ## check phi pred correct
    phi_pred = parral_target_fn(phi_datas, ctx)
    pred_correct = nd.equal(phi_pred, y_label)
    
    label_loss = parall_log_softmax_ce_loss(phi_datas, onehot_device_labels, ctx)
    cls_loss = label_loss
    
    return cls_loss, pred_correct

def constant_diff(restore_img, constant_img_label, restore_scale, batch_size):
    diff = restore_img - constant_img_label
    diff_loss = 1 -  mx.nd.smooth_l1(scalar=3.0, data=diff)
    constant_loss = mx.nd.mean(diff_loss)
    constant_loss = batch_size * constant_loss 
    return constant_loss

def l1_gan_loss(restore_img, gan_img_label, restore_scale, batch_size):
    restore_error = restore_img - gan_img_label
    restore_loss = restore_scale * mx.nd.smooth_l1(scalar=3.0, data=restore_error)
    restore_loss = mx.nd.mean(restore_loss)
    restore_loss = batch_size * restore_loss
    return restore_loss

def dssim_loss(restore_img, gan_image_label, restore_scale, batch_size):
    restore_mean = mx.nd.mean(restore_img, axis=(1,2,3), keepdims=True)
    label_mean = mx.nd.mean(gan_image_label, axis=(1,2,3), keepdims=True)

    restore_var = mx.nd.mean((restore_img - restore_mean)**2, axis=(1,2,3), keepdims=True)
    label_var = mx.nd.mean((gan_image_label - label_mean)**2, axis=(1,2,3), keepdims=True)

    covariance = mx.nd.mean(restore_img * gan_image_label, axis=(1,2,3), keepdims=True) - (restore_mean * label_mean)

    c1 = 0.01**2
    c2 = 0.03**2
    ssim = (2 * restore_mean * label_mean + c1) * (2 * covariance + c2) / ((restore_mean**2 + label_mean**2 + c1) * (restore_var + label_var + c2)) 
    dssim = (1-ssim)/2
    dssim = batch_size * mx.nd.mean(dssim)
    return dssim

def both_dssim_l1_loss(restore_img, gan_image_label, restore_scale, batch_size):
    dssim = dssim_loss(restore_img, gan_image_label[0], restore_scale, batch_size)
    restore_loss = l1_gan_loss(restore_img, gan_image_label[1], restore_scale, batch_size)
    gan_loss = dssim + restore_loss 
    return gan_loss

def both_ones_constant_l1_loss(restore_img, gan_image_label, restore_scale, batch_size):
    constant_loss = constant_diff(restore_img, gan_image_label[0], restore_scale, batch_size)
    restore_loss = l1_gan_loss(restore_img, gan_image_label[1], restore_scale, batch_size)
    gan_loss = constant_loss + restore_loss
    return gan_loss

def parall_total_loss(cls_pred_datas, labels, y_label, ctx, ctx_num_classes, 
        phi_fn, psi_norm_fn, parral_target_fn, loss_s, restore_img, restore_scale, gan_img_label, gan_loss_fun, 
        descriminator_cls_pred_list, descriminator_cls_labels, descriminator_cls_num, batch_size):
    with mx.autograd.record():
        cls_loss = mx.nd.array([0], ctx=ctx)
        pred_correct = mx.nd.array([0], ctx=ctx)

        ## get true label loss
        cls_loss, pred_correct = parall_cls_loss(cls_pred_datas, labels, y_label, ctx, ctx_num_classes, phi_fn, psi_norm_fn, parral_target_fn, loss_s)

        ## get dec label loss
        descriminator_cls_loss = mx.nd.array([0], ctx=ctx)
        descriminator_correct = mx.nd.array([0], ctx=ctx)
        if len(descriminator_cls_pred_list) > 0:
            descriminator_cls_loss, descriminator_correct = cls_loss_fun(descriminator_cls_pred_list, descriminator_cls_labels, 
                    descriminator_cls_num, ctx, phi_fn, psi_norm_fn, cls_argmax, loss_s)

        ## get restore gan loss
        restore_loss = mx.nd.array([0], ctx=ctx)
        if restore_img is not None:
            restore_loss = gan_loss_fun(restore_img, gan_img_label, restore_scale, batch_size)
        
        total_loss = cls_loss + restore_loss + descriminator_cls_loss

    return total_loss, pred_correct, restore_loss, cls_loss, descriminator_cls_loss, descriminator_correct

def parall_feat_mom_udpate(batch_fc1, device_labels, device_feats, feat_mom, ctx_num_cls):
    zeros_pad_lines = [mx.nd.zeros_like(device_feat[0]).reshape(1,-1) for device_feat in device_feats]
    pad_feats = [mx.nd.concat(*[zeros_pad_lines[i], device_feat, zeros_pad_lines[i]], dim=0) for i, device_feat in enumerate(device_feats)]
    clip_labels = [mx.nd.clip(label+1, 0, ctx_num_cls+1) for label in device_labels]
    for pad_feat, clip_label in zip(pad_feats, clip_labels):
        pad_feat[clip_label, :] = feat_mom * pad_feat[clip_label, :] + (1-feat_mom) * batch_fc1.as_in_context(pad_feat.context)
    for device_feat, pad_feat in zip(device_feats, pad_feats):
        device_feat[:] = mx.nd.L2Normalization(pad_feat[1:-1], mode='instance')
    return device_feats

class ParallModule(BaseModule):
    def __init__(self, symbol, data_names, label_names,
                 logger=logging, context=ctx.cpu(), asymbol = None, args = None, config=None,  
                 restore_sym=None, restore_scale=1.0,  model_teacher = None,
                 get_descriminator_cls_sym_fn=None, descriminator_embedding=None, **kwargs):
        super(ParallModule, self).__init__(logger=logger)
        self._symbol = symbol
        self._asymbol = asymbol
        self._data_names = data_names
        self._context = context
        self._batch_size = args.batch_size
        self._verbose = args.verbose
        self._emb_size = config.emb_size
        self._loss_s = config.loss_s

        if ('plus' in args.phi_name) or ('slope' in args.phi_name):
            assert False
            phi_name = args.phi_name
            suffix_idx = phi_name.rfind('_')
            l_n = int(phi_name[suffix_idx+1 : ])
            phi_fn = eval(phi_name[: suffix_idx+1]+'n')
            self._phi_fn = functools.partial(phi_fn, n=l_n)
            self.logger.info("= linear loss {} with {}".format(phi_name, l_n))
        else:
            self._phi_fn = eval(args.phi_name)

        self.logger.info("=== psi fun init {}".format(args.psi_norm_name))
        self._psi_norm_fn = eval(args.psi_norm_name)
        self._parall_target_fn = parall_argmax
        if args.psi_norm_name == 'penalize_linear_psi':
            self.logger.info("=== psi linear slope {}, margin {}".format(config.slope, config.margin))

            self._psi_norm_fn = functools.partial(self._psi_norm_fn, slope=config.slope, margin=config.margin)
            self._parall_target_fn = parall_argmin

        self._local_class_start = args.local_class_start
        assert self._local_class_start == 0
        self._iter = 0

        self._num_ctx = len(self._context)
        self._ctx_num_classes = args.ctx_num_classes
        self._total_cls_num = self._ctx_num_classes * len(self._context)
        self._ctx_single_gpu = self._context[-1]

        label_name = None
        self._backbone_module = Module(self._symbol, self._data_names, label_name, logger=self.logger, context=self._context)
        self._phi_parall_cls_modules = []
        self._ctx_class_start = []

        ## parall cls sym
        for i in range(len(self._context)):
            args._ctxid = i
            _module = Module(self._asymbol(args), self._data_names, label_name, logger=self.logger,
                            context=self._context[i])
            self._phi_parall_cls_modules.append(_module)
            _c = self._local_class_start + i* self._ctx_num_classes
            self._ctx_class_start.append(_c)

        ## restore error analysis
        self._restore_scale = restore_scale
        self._add_gan_loss = False
        self._gan_both_loss = True if 'both' in args.gan_loss_fun else False
        self._gan_loss_fun = eval(args.gan_loss_fun)
        if restore_sym is not None:
            self._add_gan_loss = True
            self.logger.info("==== add gan loss fun {} with scale {} both {} for generative loss ======".format(args.gan_loss_fun, restore_scale, self._gan_both_loss))
            self._restore_img_sym = restore_sym 
            self._restore_module = Module(self._restore_img_sym, ['data'], [], 
                                logger=self.logger, context=self._context)

        ## decode embedding and cls layer
        self._add_descriminator = False
        self._descriminator_cls_num = 2

        if descriminator_embedding is not None:
            assert self._add_gan_loss ## descriminator available only when AE generate image from decoder
            self._add_descriminator = True
            self._add_input2descriminator = True

            self._descriminator_cls_modules = []
            self.logger.info("=== add descriminator layer ======================")
            self._descriminator_batch_mul = 2

            self._descriminator_embedding = descriminator_embedding
            self._descriminator_embedding_module = Module(self._descriminator_embedding, 
                                        ['data'], [], 
                                        logger=self.logger, context=self._context)

            self.logger.info("==== decode cls mul {} because add_input to dec set {}".format(self._descriminator_batch_mul, self._add_input2descriminator))
            args._ctxid = 0
            descriminator_cls_mod = Module(get_descriminator_cls_sym_fn(args), self._data_names, label_name, 
                                            logger=self.logger, context=self._ctx_single_gpu)
            self._descriminator_cls_modules.append(descriminator_cls_mod)

        self._teacher_correct_cnt = 0
        self._teacher_batch_cnt = 0
        self._frequent = args.frequent
        self._model_teacher = model_teacher
        self._teacher_topk = args.teacher_topk
        if self._model_teacher is not None:
            self.logger.info("==== add teacher model with topk setting {}".format(self._teacher_topk))
            self._teacher_backbone_module = Module(self._model_teacher.backbone_sym, self._data_names,
                                                   label_name, context=self._context)
            self._teacher_fc_modules = []
            for i in range(len(self._context)):
                args._ctxid = i
                _teacher_cls_part_mod = Module(self._model_teacher.get_arcface_fun(args), self._data_names, label_name, logger=self.logger,
                             context=self._context[i])
                self._teacher_fc_modules.append(_teacher_cls_part_mod)

        self.logger.info("==== init with scale {} ".format(self._loss_s))

    def _reset_bind(self):
        self.binded = False
        self._backbone_module = None

    @property
    def data_names(self):
        return self._data_names

    @property
    def output_names(self):
        return self._symbol.list_outputs()

    @property
    def data_shapes(self):
        assert self.binded
        return self._backbone_module.data_shapes

    @property
    def label_shapes(self):
        assert self.binded
        return self._backbone_module.label_shapes

    @property
    def output_shapes(self):
        assert self.binded
        return self._backbone_module.output_shapes

    def get_export_params(self):
        assert self.binded and self.params_initialized
        _g, _x = self._backbone_module.get_params()
        g = _g.copy()
        x = _x.copy()
        return g, x

    def _get_dec_cls_params(self):
        _dec_dis_em_params, _dec_dis_em_x = self._descriminator_embedding_module.get_params()
        g = _dec_dis_em_params.copy()
        x = _dec_dis_em_x.copy()

        for _module in self._descriminator_cls_modules:
            _g, _x = _module.get_params()
            ag = _g.copy()
            ax = _x.copy()
            g.update(ag)
            x.update(ax)
        return g,x

    def _get_enc_clsnet_params(self):
        _g, _x = self._backbone_module.get_params()
        g = _g.copy()
        x = _x.copy()

        for _module in self._phi_parall_cls_modules:
            _g, _x = _module.get_params()
            ag = _g.copy()
            ax = _x.copy()
            g.update(ag)
            x.update(ax)
        return g, x

    
    def get_params(self):
        assert self.binded and self.params_initialized
        _enc_g, _enc_x = self._get_enc_clsnet_params()
        g = _enc_g.copy()
        x = _enc_x.copy()
        
        if self._add_gan_loss:
            _k_g, _k_x = self._restore_module.get_params()
            kg = _k_g.copy()
            kx = _k_x.copy()
            g.update(kg)
            x.update(kx)

        if self._add_descriminator:
            _dec_cls_g, _dec_cls_x = self._get_dec_cls_params()
            dec_g = _dec_cls_g.copy()
            dec_x = _dec_cls_x.copy()
            g.update(dec_g)
            x.update(dec_x)
        return g, x


    def set_params(self, arg_params, aux_params, allow_missing=False, force_init=True,
                   allow_extra=False):
        ## encode cls net
        for _enc_cls_module in self._phi_parall_cls_modules: 
            _enc_cls_module.set_params(arg_params, aux_params, allow_missing=allow_missing, allow_extra=allow_extra) 
        self._backbone_module.set_params(arg_params, aux_params, allow_missing=allow_missing, allow_extra=allow_extra) 

        ## decode restore net
        if self._add_gan_loss:
            self._restore_module.set_params(arg_params, aux_params, allow_missing=allow_missing, allow_extra=allow_extra)
        
        ## decode discriminative net
        if self._add_descriminator:
            for _descriminator_cls_mod in self._descriminator_cls_modules: 
                _descriminator_cls_mod.set_params(arg_params, aux_params, allow_missing=allow_missing, allow_extra=allow_extra)
            self._descriminator_embedding_module.set_params(arg_params, aux_params, allow_missing=allow_missing, allow_extra=allow_extra)
        

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False, allow_extra=False):
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'
        #TODO init the same weights with all work nodes
        self._backbone_module.init_params(initializer=initializer,arg_params=arg_params,
                                aux_params=aux_params, allow_missing=False,
                                force_init=force_init, allow_extra=allow_extra)
        for _module in self._phi_parall_cls_modules:
            #_initializer = initializer
            _initializer = mx.init.Normal(0.01)
            _module.init_params(initializer=_initializer, arg_params=arg_params,
                                aux_params=aux_params, allow_missing=allow_missing,
                                force_init=force_init, allow_extra=allow_extra)

        if self._add_gan_loss:
            self._restore_module.init_params(initializer=initializer, arg_params=arg_params,
                                    aux_params=aux_params, allow_missing=allow_missing,
                                   force_init=force_init, allow_extra=allow_extra)

        if self._add_descriminator:
            self._descriminator_embedding_module.init_params(initializer=initializer, arg_params=arg_params,
                                    aux_params=aux_params, allow_missing=allow_missing,
                                   force_init=force_init, allow_extra=allow_extra)
            for _module in self._descriminator_cls_modules:
                _initializer = mx.init.Normal(0.01)
                _module.init_params(initializer=_initializer, arg_params=arg_params,
                                aux_params=arg_params, allow_missing=allow_missing,
                                force_init=force_init, allow_extra=allow_extra)
        
        if self._model_teacher:
            self._teacher_backbone_module.init_params(initializer=initializer, arg_params=self._model_teacher.backbone_arg_params,
                                                      aux_params=self._model_teacher.backbone_aux_params, allow_missing=False,
                                                      force_init=force_init, allow_extra=False)
            for i, _module in enumerate(self._teacher_fc_modules):
                _initializer = mx.init.Normal(0.01)
                arg_params = {}
                arg_params['fc7_%d_weight' % (i)] = self._model_teacher.fc_arg_params['fc7_%d_weight' % (i)]
                _module.init_params(initializer=_initializer, arg_params=arg_params,
                                    aux_params=None, allow_missing=False,
                                    force_init=force_init, allow_extra=False)
        self.params_initialized = True


    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None):
        print('in_bind', self.params_initialized, data_shapes, label_shapes)
        self.logger.info('in_bind {}'.format(self.params_initialized, data_shapes, label_shapes))

        if self.params_initialized:
            arg_params, aux_params = self.get_params()

        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        assert shared_module is None, 'shared_module for MutableModule is not supported'
        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True
        
        label_shapes = None
        self.logger.info('bind backbone data_shape {}, label shape {}'.format( data_shapes, label_shapes))
        self._backbone_module.bind(data_shapes, label_shapes, for_training, inputs_need_grad,
                    force_rebind=False, shared_module=None)
        
        batch_size = data_shapes[0][1][0]

        ## bind parall cls layer
        for i, _module in enumerate(self._phi_parall_cls_modules):
            _module.bind([('data', (batch_size, self._emb_size))], 
                        label_shapes, for_training, True,
                        force_rebind=False, shared_module=None)

        ## bind restore generative net layer
        if self._add_gan_loss:
            self._restore_module.bind([('data', (batch_size, self._emb_size))], 
                                label_shapes, for_training, True, force_rebind=False, shared_module=None)

        ## bind decode cls layer
        if self._add_descriminator:
            img_shape = data_shapes[0][1][1:]
            descriminator_batch_size = self._descriminator_batch_mul * batch_size
            self._descriminator_embedding_module.bind([('data', (descriminator_batch_size, *img_shape))], label_shapes, for_training, True, force_rebind=False, shared_module=None)

            for i, _descriminator_cls_modules in enumerate(self._descriminator_cls_modules):
                _descriminator_cls_modules.bind([('data', (descriminator_batch_size, self._emb_size))], 
                        label_shapes, for_training, True,
                        force_rebind=False, shared_module=None)

        ## bind teacher with data
        if self._model_teacher is not None:
            self._teacher_backbone_module.bind(data_shapes, label_shapes, for_training=False, inputs_need_grad=False,
                                               force_rebind=False, shared_module=None)
            for i, _module in enumerate(self._teacher_fc_modules):
                _module.bind([('data', (batch_size, self._emb_size))],
                             label_shapes, for_training=False, inputs_need_grad=False,
                             force_rebind=False, shared_module=None)

        if self.params_initialized:
            self.set_params(arg_params, aux_params, allow_missing=allow_missing, allow_extra=allow_extra)

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        assert self.binded and self.params_initialized
        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring.')
            return

        self._backbone_module.init_optimizer(kvstore, optimizer, optimizer_params,
                                         force_init=force_init)
        for _module in self._phi_parall_cls_modules:
            _module.init_optimizer(kvstore, optimizer, optimizer_params,
                                           force_init=force_init)
        if self._add_gan_loss:
            self._restore_module.init_optimizer(kvstore, optimizer, optimizer_params,
                                           force_init=force_init)
        if self._add_descriminator:
            self._descriminator_embedding_module.init_optimizer(kvstore, optimizer, optimizer_params,
                                           force_init=force_init)
            for _module in self._descriminator_cls_modules:
                _module.init_optimizer(kvstore, optimizer, optimizer_params,
                                        force_init=force_init)

        self.optimizer_initialized = True

    #forward backbone fc1 and other parts
    def forward(self, data_batch, is_train=None):
        assert self.binded and self.params_initialized

        label = data_batch.label
        input_data = data_batch.data
                
        self._backbone_module.forward(data_batch, is_train=is_train)
        
        backbone_pred = self._backbone_module.get_outputs(merge_multi_context=True)

        if is_train:
            label_len = 2 if self._add_gan_loss else 1
            assert len(label) == label_len
            assert len(backbone_pred) == 1

            self._iter += 1
            self.global_fc1 = backbone_pred[0]
            self.global_label = label[0].as_in_context(self._ctx_single_gpu)

            self.restore_img_buff = None
            self.gan_image_label = None

            if self._add_gan_loss:
                if self._gan_both_loss:
                    assert False
                    ### 0 is dssim , and 1 for l1 regression
                    self.gan_image_label = [(input_data[0]/255.0).as_in_context(self._ctx_single_gpu), label[1].as_in_context(self._ctx_single_gpu)]
                    #self.gan_image_label = [label[1].as_in_context(self._ctx_single_gpu), label[1].as_in_context(self._ctx_single_gpu)]
                    ### 0 is ones constant , and 1 for l1 regression
                    #self.gan_image_label = [mx.nd.ones_like(input_data[0]).as_in_context(self._ctx_single_gpu), label[1].as_in_context(self._ctx_single_gpu)]
                else:
                    self.gan_image_label = label[1].as_in_context(self._ctx_single_gpu)
                db_restore_batch = io.DataBatch([backbone_pred[0]], [])
                self._restore_module.forward(db_restore_batch)
                resotore_mod_output = self._restore_module.get_outputs(merge_multi_context=True)
                assert len(resotore_mod_output) == 1
                self.restore_img_buff = resotore_mod_output[0].as_in_context(self._ctx_single_gpu)

            if self._add_descriminator:
                descriminator_databatch = io.DataBatch([mx.nd.concat(self.restore_img_buff, input_data[0].as_in_context(self._ctx_single_gpu), dim=0)], [])
                self._descriminator_embedding_module.forward(descriminator_databatch)
                descriminator_embedding_pred = self._descriminator_embedding_module.get_outputs(merge_multi_context=True)
                assert len(descriminator_embedding_pred) == 1
                for i, _module in enumerate(self._descriminator_cls_modules):
                    descriminator_cls_batch = io.DataBatch(descriminator_embedding_pred, [])
                    _module.forward(descriminator_cls_batch)

            # teacher module forward
            if self._model_teacher is not None:
                self._teacher_backbone_module.forward(data_batch, is_train=False)
                teacher_backbone_pred = self._teacher_backbone_module.get_outputs(merge_multi_context=True)
                assert len(teacher_backbone_pred) == 1
                for i, _module in enumerate(self._teacher_fc_modules):
                    teacher_fc1_databatch = io.DataBatch([teacher_backbone_pred[0]], [])
                    _module.forward(teacher_fc1_databatch, is_train=False)
            
            
        for i, _module in enumerate(self._phi_parall_cls_modules):
            db_global_fc1 = io.DataBatch([backbone_pred[0]], [])
            _module.forward(db_global_fc1) #fc7 matrix multiple
                

    def backward(self, out_grads=None):
        assert self.binded and self.params_initialized

        ## ============= backward classifier layer ===========
        self._fc_cls_buff_list = []
        for i, _module in enumerate(self._phi_parall_cls_modules):
            mod_output_list = _module.get_outputs(merge_multi_context=True)
            assert len(mod_output_list) == 1
            mod_output_list[0].attach_grad()
            self._fc_cls_buff_list.append(mod_output_list[0])

        ## ============= compute verbose train accuracy and loss ===========
        local_label = self.global_label
        device_labels = [(local_label.as_in_context(device) - self._ctx_class_start[i]) for i, device in enumerate(self._context)]
        
        descriminator_cls_labels = []
        descriminator_cls_global_label = 0*local_label
        if self._add_descriminator:
            descriminator_cls_global_label = mx.nd.concat(descriminator_cls_global_label, descriminator_cls_global_label+1, dim=0)
            descriminator_cls_labels = [descriminator_cls_global_label.as_in_context(self._ctx_single_gpu)]

        if self._add_gan_loss:
            self.restore_img_buff.attach_grad()

        self._descriminator_cls_buff_list = []
        if self._add_descriminator:
            for i, _module in enumerate(self._descriminator_cls_modules):
                mod_output_list = _module.get_outputs(merge_multi_context=True)
                assert len(mod_output_list) == 1
                mod_output_list[0].attach_grad()
                self._descriminator_cls_buff_list.append(mod_output_list[0])


        loss, pred_correct, restore_loss, cls_loss, descriminator_cls_loss, descriminator_correct = \
            parall_total_loss(self._fc_cls_buff_list, device_labels, local_label, 
                self._ctx_single_gpu, self._ctx_num_classes, self._phi_fn, self._psi_norm_fn, self._parall_target_fn, self._loss_s, 
                self.restore_img_buff, self._restore_scale, self.gan_image_label, self._gan_loss_fun, 
                self._descriminator_cls_buff_list, descriminator_cls_labels, 
                self._descriminator_cls_num, self._batch_size)
        
        assert not math.isnan(loss.asscalar())
        assert not math.isnan(restore_loss.asscalar())
        assert not math.isnan(cls_loss.asscalar())
        assert not math.isnan(descriminator_cls_loss.asscalar())


        if self._iter % self._verbose == 0:
            acc = nd.mean(pred_correct).asnumpy()
            dec_acc = nd.mean(descriminator_correct).asnumpy()
            self.logger.info('[Iter {}] train phi acc : {}, dec acc : {}, total loss : {}\n--- restore loss : {}, restore scale : {}, cls loss : {} decode dis loss : {}'.format(
                self._iter, acc, dec_acc, loss.asscalar()/ self._batch_size, 
                restore_loss.asscalar()/self._batch_size, self._restore_scale,
                cls_loss.asscalar()/self._batch_size, descriminator_cls_loss.asscalar()/self._batch_size))

        ##============caculate teacher mask ===============
        if self._model_teacher is not None:
            self._teacher_fc_cls_list = []
            for i, _module in enumerate(self._teacher_fc_modules):
                mod_output_list = _module.get_outputs(merge_multi_context=True)
                assert len(mod_output_list) == 1
                self._teacher_fc_cls_list.append(mod_output_list[0])

            if self._teacher_topk == 10000: # compute teacher pred cos sim as teacher mask
                teacher_pred_correct_mask = parall_pick_teacher_cos_label(self._teacher_fc_cls_list, device_labels, self._ctx_num_classes, self._ctx_single_gpu)
                teacher_pred_correct_mask = mx.nd.reshape(teacher_pred_correct_mask, (self._batch_size, 1))
            else:
                if self._teacher_topk == 1:
                    module_teacher_pred = self._parall_target_fn(self._teacher_fc_cls_list, self._ctx_single_gpu)
                    teacher_pred_correct_mask = mx.nd.reshape(mx.nd.equal(module_teacher_pred, local_label), (self._batch_size, 1))
                else:
                    local_label = mx.nd.reshape(local_label, (self._batch_size, 1))
                    module_teacher_pred_topk = parall_topk_index(self._teacher_fc_cls_list, self._ctx_single_gpu, self._teacher_topk)
                    teacher_pred_correct_mask = mx.nd.sum(mx.nd.broadcast_equal(module_teacher_pred_topk, local_label), axis=1, keepdims=True)
                pred_correct_nums = mx.nd.sum(teacher_pred_correct_mask).asnumpy().astype('int32')
                self._teacher_correct_cnt += pred_correct_nums[0]
                self._teacher_batch_cnt += 1
        else:
            teacher_pred_correct_mask = mx.nd.ones((self._batch_size, 1), ctx=self._ctx_single_gpu)

        
        ## ============= backward large weight classifier layer with gradient ===========
        loss.backward()

        local_fc1_grad = mx.nd.zeros((self._batch_size, self._emb_size), ctx=self._ctx_single_gpu)
        ## =========== backward parall cls layer ================
        for i, _module in enumerate(self._phi_parall_cls_modules):
            phi_cls_grad_with_mask = mx.nd.broadcast_mul(self._fc_cls_buff_list[i].grad, teacher_pred_correct_mask.as_in_context(self._context[i]))
            _module.backward(out_grads=[phi_cls_grad_with_mask])
            local_fc1_grad += _module.get_input_grads()[0].as_in_context(self._ctx_single_gpu)

        ## =========== backward decode net cls model ======
        if self._add_descriminator:
            descriminator_cls_grad_4_descriminator_embedding = mx.nd.zeros((self._descriminator_batch_mul * self._batch_size, self._emb_size), ctx=self._ctx_single_gpu)
            for i, _module in enumerate(self._descriminator_cls_modules):
                _module.backward(out_grads=[self._descriminator_cls_buff_list[i].grad])
                dec_cls_grad = _module.get_input_grads()[0].as_in_context(self._ctx_single_gpu)
                descriminator_cls_grad_4_descriminator_embedding += dec_cls_grad
            
            self._descriminator_embedding_module.backward(out_grads=[descriminator_cls_grad_4_descriminator_embedding])

            dec_cls_net_input_grads = self._descriminator_embedding_module.get_input_grads()
            assert len(dec_cls_net_input_grads) == 1
            dec_cls_net_grad_4_gan_image = mx.nd.split(dec_cls_net_input_grads[0].as_in_context(self._ctx_single_gpu), num_outputs=2, axis=0)[0]

        ## =========== backward restore layer ============
        if self._add_gan_loss:
            restore_grad = self.restore_img_buff.grad

            if self._add_descriminator:
                restore_grad = restore_grad + dec_cls_net_grad_4_gan_image
            
            ##restore_grad = mx.nd.broadcast_mul(restore_grad, teacher_pred_correct_mask.reshape((self._batch_size, 1, 1, 1)).as_in_context(restore_grad.context))
            self._restore_module.backward(out_grads = [restore_grad])
            restore_fc1_grad = self._restore_module.get_input_grads()[0].as_in_context(self._ctx_single_gpu)
            restore_fc1_grad = mx.nd.broadcast_mul(restore_fc1_grad, teacher_pred_correct_mask.as_in_context(self._ctx_single_gpu))
            local_fc1_grad = local_fc1_grad + restore_fc1_grad

        ## ============= backward backbone ===============
        self._backbone_module.backward(out_grads = [local_fc1_grad])


    def update(self):
        assert self.binded and self.params_initialized and self.optimizer_initialized
        self._backbone_module.update()
        for i, _module in enumerate(self._phi_parall_cls_modules):
            _module.update()

        if self._add_gan_loss:
            self._restore_module.update()

        if self._add_descriminator:
            self._descriminator_embedding_module.update()
            for _dec_mod in self._descriminator_cls_modules:
                _dec_mod.update()

        mx.nd.waitall()


    def get_outputs(self, merge_multi_context=True):
        assert self.binded and self.params_initialized
        return self._backbone_module.get_outputs(merge_multi_context=merge_multi_context)

    def get_class_output(self, merge_multi_context=True):
        part_pred_list = [m.get_outputs(merge_multi_context=merge_multi_context)[0]
                                for m in self._phi_parall_cls_modules]

        fc7_pred_label = self._parall_target_fn(part_pred_list, self._ctx_single_gpu)
        return [fc7_pred_label]


    def reset_teacher_metric(self):
        self._teacher_correct_cnt = 0
        self._teacher_batch_cnt = 0



    def get_input_grads(self, merge_multi_context=True):
        assert False
        assert self.binded and self.params_initialized and self.inputs_need_grad
        return self._backbone_module.get_input_grads(merge_multi_context=merge_multi_context)

    def update_metric(self, eval_metric, labels):
        assert self.binded and self.params_initialized

        preds = self.get_class_output(merge_multi_context=True)
        
        label_len = 2 if self._add_gan_loss else 1
        #assert len(labels) == label_len, 'label out len'
        assert len(preds) == 1, 'pred cls out len'
        eval_metric.update(labels=[labels[0]], preds=preds)
        

    def install_monitor(self, mon):
        """ Install monitor on all executors """
        assert self.binded
        self._backbone_module.install_monitor(mon)
        for enc_cls_mod in self._phi_parall_cls_modules:
            enc_cls_mod.install_monitor(mon)

        if self._add_gan_loss:
            self._restore_module.install_monitor(mon)

        if self._add_descriminator:
            self._descriminator_embedding_module.install_monitor(mon)
            for dec_cls_mod in self._descriminator_cls_modules:
                dec_cls_mod.install_monitor(mon)

    def forward_backward(self, data_batch):
        """A convenient function that calls both ``forward`` and ``backward``."""
        self.forward(data_batch, is_train=True) # forward net
        self.backward()

    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None, sparse_row_id_fn=None):
        """Trains the module parameters.

        Checkout `Module Tutorial <http://mxnet.io/tutorials/basic/module.html>`_ to see
        a end-to-end use-case.

        Parameters
        ----------
        train_data : DataIter
            Train DataIter.
        eval_data : DataIter
            If not ``None``, will be used as validation set and the performance
            after each epoch will be evaluated.
        eval_metric : str or EvalMetric
            Defaults to 'accuracy'. The performance measure used to display during training.
            Other possible predefined metrics are:
            'ce' (CrossEntropy), 'f1', 'mae', 'mse', 'rmse', 'top_k_accuracy'.
        epoch_end_callback : function or list of functions
            Each callback will be called with the current `epoch`, `symbol`, `arg_params`
            and `aux_params`.
        batch_end_callback : function or list of function
            Each callback will be called with a `BatchEndParam`.
        kvstore : str or KVStore
            Defaults to 'local'.
        optimizer : str or Optimizer
            Defaults to 'sgd'.
        optimizer_params : dict
            Defaults to ``(('learning_rate', 0.01),)``. The parameters for
            the optimizer constructor.
            The default value is not a dict, just to avoid pylint warning on dangerous
            default values.
        eval_end_callback : function or list of function
            These will be called at the end of each full evaluation, with the metrics over
            the entire evaluation set.
        eval_batch_end_callback : function or list of function
            These will be called at the end of each mini-batch during evaluation.
        initializer : Initializer
            The initializer is called to initialize the module parameters when they are
            not already initialized.
        arg_params : dict
            Defaults to ``None``, if not ``None``, should be existing parameters from a trained
            model or loaded from a checkpoint (previously saved model). In this case,
            the value here will be used to initialize the module parameters, unless they
            are already initialized by the user via a call to `init_params` or `fit`.
            `arg_params` has a higher priority than `initializer`.
        aux_params : dict
            Defaults to ``None``. Similar to `arg_params`, except for auxiliary states.
        allow_missing : bool
            Defaults to ``False``. Indicates whether to allow missing parameters when `arg_params`
            and `aux_params` are not ``None``. If this is ``True``, then the missing parameters
            will be initialized via the `initializer`.
        force_rebind : bool
            Defaults to ``False``. Whether to force rebinding the executors if already bound.
        force_init : bool
            Defaults to ``False``. Indicates whether to force initialization even if the
            parameters are already initialized.
        begin_epoch : int
            Defaults to 0. Indicates the starting epoch. Usually, if resumed from a
            checkpoint saved at a previous training phase at epoch N, then this value should be
            N+1.
        num_epoch : int
            Number of epochs for training.
        sparse_row_id_fn : A callback function
            The function  takes `data_batch` as an input and returns a dict of
            str -> NDArray. The resulting dict is used for pulling row_sparse
            parameters from the kvstore, where the str key is the name of the param,
            and the value is the row id of the param to pull.

        Examples
        --------
        >>> # An example of using fit for training.
        >>> # Assume training dataIter and validation dataIter are ready
        >>> # Assume loading a previously checkpointed model
        >>> sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)
        >>> mod.fit(train_data=train_dataiter, eval_data=val_dataiter, optimizer='sgd',
        ...     optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
        ...     arg_params=arg_params, aux_params=aux_params,
        ...     eval_metric='acc', num_epoch=10, begin_epoch=3)
        """
        assert num_epoch is not None, 'please specify number of epochs'
        #assert arg_params is None and aux_params is None

        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=True, force_init=force_init, allow_extra=True)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if validation_metric is None:
            validation_metric = eval_metric

        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)
            print("=== init eval metirc {}, {}".format(eval_metric, type(eval_metric)))

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            eval_metric.reset()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            while not end_of_batch:
                data_batch = next_data_batch
                if monitor is not None:
                    monitor.tic()
                self.forward_backward(data_batch)
                self.update()
                assert not isinstance(data_batch, list)

                if isinstance(data_batch, list):
                    assert False
                    db_cls_label = mx.nd.concat(*[db.label[0] for db in data_batch], dim=0)
                    self.update_metric(eval_metric,
                                       [db_cls_label],
                                       pre_sliced=True)
                else:
                    self.update_metric(eval_metric, data_batch.label)

                try:
                    # pre fetch next batch
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch, sparse_row_id_fn=sparse_row_id_fn)
                except StopIteration:
                    end_of_batch = True

                if monitor is not None:
                    monitor.toc_print()

                if end_of_batch:
                    eval_name_vals = eval_metric.get_name_value()

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    batch_end_callback(batch_end_params)

                if self._model_teacher and self._teacher_topk != 10000 and (self._teacher_batch_cnt % self._frequent == 0):
                    acc = self._teacher_correct_cnt / (self._teacher_batch_cnt * self._batch_size)
                    self.logger.info('TeacherModule-Accuracy=%f', acc)
                    self.reset_teacher_metric()

                nbatch += 1

            # one epoch of training is finished
            for name, val in eval_name_vals:
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)

            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            # sync aux params across devices
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params, allow_missing=False, allow_extra=True) 


            #----------------------------------------
            # evaluation on validation set
            if eval_data:
                res = self.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                #TODO: pull this into default
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()

            if epoch_end_callback is not None:
                epoch_end_callback(epoch)
                


