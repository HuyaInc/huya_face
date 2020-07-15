'''
refactor train parall main function
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import numpy as np
import functools
from easydict import EasyDict as edict


import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
sys.path.append(os.path.join(os.path.dirname(os.path.abspath('..')), 'common'))
import flops_counter
from config import config, default, generate_config
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
import verification

sys.path.append(os.path.join(os.path.abspath('..'), 'symbol'))
from face_symbol_factory import get_symbol_embedding, get_symbol_arcface


def parse_args():
    parser = argparse.ArgumentParser(description='Train parall face network')
    # general
    parser.add_argument('--dataset', default=default.dataset, help='dataset config')
    parser.add_argument('--network', default=default.network, help='network config')
    parser.add_argument('--loss', default=default.loss, help='loss config')
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset, args.loss)
    parser.add_argument('--models-root', default=default.models_root, help='root directory to save model.')
    parser.add_argument('--pretrained', default=default.pretrained, help='pretrained backbone model to load')
    parser.add_argument('--pretrained-epoch', type=int, default=default.pretrained_epoch, help='pretrained epoch to load')
    parser.add_argument('--ckpt', type=int, default=default.ckpt, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
    parser.add_argument('--verbose', type=int, default=default.verbose, help='do verification testing and model saving every verbose batches')
    parser.add_argument('--lr', type=float, default=default.lr, help='start learning rate')
    parser.add_argument('--lr-steps', type=str, default=default.lr_steps, help='steps of lr changing')
    parser.add_argument('--wd', type=float, default=default.wd, help='weight decay')
    parser.add_argument('--mom', type=float, default=default.mom, help='momentum')
    parser.add_argument('--frequent', type=int, default=default.frequent, help='')
    parser.add_argument('--per-batch-size', type=int, default=default.per_batch_size, help='batch size in each context')
    parser.add_argument('--kvstore', type=str, default=default.kvstore, help='kvstore setting')
    parser.add_argument('--worker-id', type=int, default=0, help='worker id for dist training, starts from 0')
    parser.add_argument('--extra-model-name', type=str, default='', help='extra model name')
    parser.add_argument('--net-output', default='E', help='net_output to fc_1')

    parser.add_argument('--phi-name', default='nd_phi_linear', help='phi func to add margin')
    parser.add_argument('--psi-norm-name', default='penalize_with_cos_psi', help='psi func for non class values')

    parser.add_argument('--add-gan-loss', dest='add_gan_loss', action='store_true', \
            help='whether to add feat restore loss')
    parser.add_argument('--gan-mom', dest='gan_mom', type=float, default=0.9, \
            help='gan label mean avg mom')
    parser.add_argument('--decode-layer', dest='decode_layer', type=int, default=18, \
            help='gan decode net depth')
    parser.add_argument('--gan-loss-fun', dest='gan_loss_fun', type=str, default='l1_gan_loss', \
            help='gan loss function name')

    parser.add_argument('--restore-scale', type=float, default=10.0, \
            help='restore loss scale')
    parser.add_argument('--add-descriminator', dest='add_descriminator', action='store_true', \
            help='whether classify decode image')
    parser.add_argument('--descriminator-layers', type=int, default=18, help='decode cls network layers')

    parser.add_argument('--backbone-only', dest='backbone_only', action='store_true', \
            help='whether only load pretrain backbone')

    parser.add_argument('--model-teacher', type=str, default=default.model_teacher, help='')
    parser.add_argument('--teacher-epoch', type=int, default=1, help='teacher epoch to load')
    parser.add_argument('--teacher-topk',  type=int, default=1, help='teacher predit topk condition')

    args = parser.parse_args()
    return args

def get_data_iter(config, batch_size):
    data_dir = config.dataset_path
    path_imgrec = None
    path_imglist = None
    image_size = config.image_shape[0:2]

    assert len(image_size)==2
    assert image_size[0]==image_size[1]
    print('image_size', image_size)

    path_imgrec = os.path.join(data_dir, "train.rec")
    data_shape = (config.image_shape[2], image_size[0], image_size[1])
    print("====== data shape is ", data_shape)

    val_dataiter = None 
    if config.dataset == 'cifar100':
        ## cifar dataset for debug
        train_dataiter = mx.io.ImageRecordIter(
                path_imgrec=os.path.join(data_dir,"train.rec"), 
                data_name="data", 
                label_name="softmax_label", 
                batch_size=batch_size, 
                data_shape=data_shape, 
                shuffle=True)
    elif 'gan' in config.dataset: ## key point dataset
        from gan_image_iter import GanFaceImageIter
        train_dataiter = GanFaceImageIter(
            batch_size           = batch_size,
            data_shape           = data_shape,
            path_imgrec          = path_imgrec,
            shuffle              = True,
            rand_mirror          = config.data_rand_mirror,
            mean                 = None,
            cutoff               = config.data_cutoff,
            color_jittering      = config.data_color,
            images_filter        = config.data_images_filter,
            gan_mom              = config.gan_mom,
        )
        train_dataiter = mx.io.PrefetchingIter(train_dataiter)
    else:
        from image_iter import FaceImageIter
        train_dataiter = FaceImageIter(
            batch_size           = batch_size,
            data_shape           = data_shape,
            path_imgrec          = path_imgrec, 
            shuffle              = True,
            rand_mirror          = config.data_rand_mirror,
            mean                 = None,
            cutoff               = config.data_cutoff,
            color_jittering      = config.data_color,
            images_filter        = config.data_images_filter,
        )
        train_dataiter = mx.io.PrefetchingIter(train_dataiter)

    if 'cifar100' in config.dataset:
        val_dataiter = mx.io.ImageRecordIter(
                path_imgrec=os.path.join(data_dir,"test.rec"), 
                data_name="data", 
                label_name="softmax_label", 
                batch_size=batch_size, 
                data_shape=data_shape)
    return train_dataiter, val_dataiter

def train_net(args):
    ## =================== parse context ==========================
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
        for i in range(len(cvd.split(','))):
            ctx.append(mx.gpu(i))
    if len(ctx)==0:
        ctx = [mx.cpu()]
        print('use cpu')
    else:
        print('gpu num:', len(ctx))


    ## ==================== get model save prefix and log ============
    loss_name = args.phi_name if 'phi' in args.loss else args.loss

    if len(args.extra_model_name)==0:
        prefix = os.path.join(args.models_root, '%s-%s-%s'%(args.network, loss_name, args.dataset), 'model')
        prefix_fc = os.path.join(args.models_root, '%s-%s-%s' % (args.network, loss_name, args.dataset), 'fc_model_gs%d'%(len(ctx)))
        prefix_gan = os.path.join(args.models_root, '%s-%s-%s' % (args.network, loss_name, args.dataset), 'gan_model')
        prefix_dec = os.path.join(args.models_root, '%s-%s-%s' % (args.network, loss_name, args.dataset), 'dec_model')
    else:
        prefix = os.path.join(args.models_root, '%s-%s-%s-%s'%(args.network, loss_name, args.dataset, args.extra_model_name), 'model')
        prefix_fc = os.path.join(args.models_root, '%s-%s-%s-%s' % (args.network, loss_name, args.dataset, args.extra_model_name), 'fc_model_gs%d'%(len(ctx)))
        prefix_gan = os.path.join(args.models_root, '%s-%s-%s-%s' % (args.network, loss_name, args.dataset, args.extra_model_name), 'gan_model')
        prefix_dec = os.path.join(args.models_root, '%s-%s-%s-%s' % (args.network, loss_name, args.dataset, args.extra_model_name), 'dec_model')


    prefix_dir = os.path.dirname(prefix)
    print('prefix', prefix)
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler("{}.log".format(prefix))
    streamhandler = logging.StreamHandler()
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    ## ================ parse batch size and class info and set pretrain params ======================
    args.ctx_num = len(ctx)
    if args.per_batch_size==0:
        args.per_batch_size = 128
    args.batch_size = args.per_batch_size*args.ctx_num
    
    global_num_ctx = config.num_workers * args.ctx_num
    config.num_ctx = global_num_ctx
    if config.num_classes % global_num_ctx == 0:
        args.ctx_num_classes = config.num_classes//global_num_ctx
    else:
        assert False
        args.ctx_num_classes = config.num_classes//global_num_ctx+1

    args.local_num_classes = args.ctx_num_classes * args.ctx_num
    args.local_class_start = args.local_num_classes * args.worker_id
    
    if len(args.pretrained) != 0:
        args.lr = default.lr_pretrained
        args.lr_steps = default.lr_steps_pretrained

    logger.info("Train model with argument: {}\nconfig : {}".format(args, config))

    config.gan_mom = args.gan_mom
    train_dataiter, val_dataiter = get_data_iter(config, args.batch_size)

    ## =============== get train info ============================
    image_size = config.image_shape[0:2]
    config.per_batch_size = args.per_batch_size
    
    ## ================ get backbone embedding ====================
    embedding_with_label = False

    ## ============= backbone to feature ==========================
    config.net_output = args.net_output
    logger.info("===== use {} as fc1 output ".format(args.net_output))
    esym = get_symbol_embedding(config, embedding_with_label)
    if config.count_flops:
        all_layers = esym.get_internals()
        _sym = all_layers['fc1_output']
        FLOPs = flops_counter.count_flops(_sym, data=(1,3,image_size[0],image_size[1]))
        _str = flops_counter.flops_str(FLOPs)
        logging.info("Network FLOPs : %s" % _str)

    ##=================creat teacher_model ========================
    model_teacher = None
    if len(args.model_teacher) != 0:
        logger.info("== load teacher model to clean data")
        teacher_backbone_prefix = os.path.join(args.model_teacher, 'model')
        teacher_fc_prefix = os.path.join(args.model_teacher, 'fc_model_gs%d'%(len(ctx)))

        model_teacher = edict()
        model_teacher.backbone_sym, model_teacher.backbone_arg_params, model_teacher.backbone_aux_params = mx.model.load_checkpoint(teacher_backbone_prefix, args.teacher_epoch)
        _, model_teacher.fc_arg_params, _ = mx.model.load_checkpoint(teacher_fc_prefix, args.teacher_epoch)
        model_teacher.get_arcface_fun = functools.partial(get_symbol_arcface, config=config, name_prefix='')
   
 
    if len(args.pretrained) == 0: # train from scratch
        logger.info("== train from scratch")
        config.phi_fun_name = args.phi_name
        asym = functools.partial(get_symbol_arcface, config=config)
        arg_params = aux_params = None
    else: # load train model to continue
        config.phi_fun_name = args.phi_name
        asym = functools.partial(get_symbol_arcface, config=config)
        arg_params = {}
        aux_params = {}
        if args.backbone_only:
            logger.info("== load from backbone {} only".format(args.pretrained))
            _, backbone_arg_params, backbone_aux_params = mx.model.load_checkpoint(args.pretrained, args.pretrained_epoch)
            arg_params.update(backbone_arg_params)
            aux_params.update(backbone_aux_params)
        else:
            pretrained_model_prefixs = args.pretrained.split(',')
            for pretrained_model in pretrained_model_prefixs:
                logger.info("== load model from {} ==".format(pretrained_model))
                _, load_arg_params, load_aux_params = mx.model.load_checkpoint(pretrained_model, args.pretrained_epoch)
                arg_params.update(load_arg_params)
                aux_params.update(load_aux_params)


    ## ========== parall loss module =======================
    restore_sym = None
    descriminator_cls_sym_fn = None
    descriminator_embedding = None
    if config.num_workers == 1:
        from parall_module_local import ParallModule
        if args.add_gan_loss:
            from face_symbol_factory import get_decode_img 
            for_face = False if 'cifar' in config.dataset else True
            decode_channel = 1 if config.net_input == 10 else 3
            logger.info("== add gan restore module with channel : {} layer : {}".format(decode_channel, args.decode_layer))
            restore_sym = get_decode_img(config.emb_size, decode_num_layer=args.decode_layer, for_face=for_face, decode_channel=decode_channel)
            if args.add_descriminator:
                from face_symbol_factory import get_descriminator_embedding_sym
                config.num_layers = args.descriminator_layers
                descriminator_prefix = 'descriminator_'
                descriminator_embedding = get_descriminator_embedding_sym(config, name_prefix=descriminator_prefix)
                descriminator_cls_sym_fn = functools.partial(get_symbol_arcface, config=config, name_prefix=descriminator_prefix, cls_num=2)

                if config.count_flops:
                    all_layers = descriminator_embedding.get_internals()
                    _sym = all_layers['%sfc1_output' % descriminator_prefix]
                    FLOPs = flops_counter.count_flops(_sym, data=(1,3,image_size[0],image_size[1]))
                    _str = flops_counter.flops_str(FLOPs)
                    logging.info("Decode Embedding Network FLOPs : %s" % _str)

        
    else: # distribute parall loop
        assert False

    model = ParallModule(
        context       = ctx,
        symbol        = esym,
        data_names    = ['data'],
        label_names   = ['softmax_label'],
        asymbol       = asym,
        args          = args,
        logger        = logger,
        config        = config,
        restore_sym   = restore_sym,
        restore_scale = args.restore_scale,
        get_descriminator_cls_sym_fn = descriminator_cls_sym_fn,
        descriminator_embedding = descriminator_embedding,
        model_teacher = model_teacher,
    )
    

    ## ============ get optimizer =====================================
    if config.net_name=='fresnet' or config.net_name=='fmobilefacenet':
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    else:
        assert False
        initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)

    opt = optimizer.SGD(learning_rate=args.lr, momentum=args.mom, wd=args.wd, rescale_grad=1.0/args.batch_size)
    _cb = mx.callback.Speedometer(args.batch_size, args.frequent)

    ver_list = []
    ver_name_list = []
    ver_data_dir = config.valsets_path
    print('=== setting test data dir ', ver_data_dir)
    for name in config.val_targets:
        path = os.path.join(ver_data_dir, name+".bin")
        if os.path.exists(path):
            data_set = verification.load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)
            logger.info("ver {}".format(name))


    def ver_test(nbatch, test_model, batch_size):
        results = []
        for i in range(len(ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], test_model, batch_size, 10, None, None)
            logger.info("[{}][{}]XNorm: {}".format(ver_name_list[i], nbatch, xnorm))
            logger.info('[{}][{}]Accuracy-Flip: {}+-{}'.format(ver_name_list[i], nbatch, acc2, std2))
            results.append(acc2)
        return results

    highest_acc = [0.0, 0.0]  #lfw and target
    

    global_step = [0]
    save_step = [0]
    lr_steps = [int(x) for x in args.lr_steps.split(',')]
    logger.info("lr_steps {}".format(lr_steps))

    ## =============== batch end callback definition ===================================
    def _save_params(msave):
        logger.info('saving {}'.format(msave))
        _arg, _aux = model.get_params()
        if config.save_fc:
            _asym = model._asymbol(args).get_internals()
            fc_arg = {}
            for k in _arg:
                if k.startswith('fc7'):
                    fc_arg[k] = _arg[k]
            mx.model.save_checkpoint(prefix_fc, msave, _asym, fc_arg, _aux)

        if args.add_gan_loss:
            gan_sym = model._restore_img_sym
            gan_args = {}
            for k in _arg:
                if k.startswith('decode'):
                    gan_args[k] = _arg[k]
            gan_auxs = {}
            for k in _aux:
                if k.startswith('decode'):
                    gan_auxs[k] = _aux[k]
            mx.model.save_checkpoint(prefix_gan, msave, gan_sym, gan_args, gan_auxs)

            if args.add_descriminator:
                dec_em_sym = model._descriminator_embedding
                dec_args, dec_auxs = model._get_dec_cls_params()
                mx.model.save_checkpoint(prefix_dec, msave, dec_em_sym, dec_args, dec_auxs)

        arg, aux = model.get_export_params()
        all_layers = model.symbol.get_internals()
        _sym = all_layers['{}fc1_output'.format(config.name_prefix)]
        mx.model.save_checkpoint(prefix, msave, _sym, arg, aux)

    def _epoch_callback(epoch_num):
        _save_params(epoch_num)

    def _batch_callback(param):
        #global global_step
        global_step[0] += 1
        mbatch = global_step[0]
        for step in lr_steps:
            if mbatch==step:
                opt.lr *= 0.1
                logger.info('lr change to {}'.format(opt.lr))
                break

        _cb(param)
        if mbatch % 1000 == 0:
            logger.info('lr-batch-epoch: {}, {}, {}'.format(opt.lr, param.nbatch, param.epoch))

        if mbatch>=0 and mbatch % args.verbose==0:
            acc_list = ver_test(mbatch, model, args.batch_size)
            save_step[0]+=1
            msave = save_step[0]
            do_save = False
            is_highest = False

            def judge_is_highest(acc_list, is_highest):
                if len(acc_list) > 0:
                    score = sum(acc_list)
                    if acc_list[-1]  >= highest_acc[-1]:
                        if acc_list[-1] > highest_acc[-1]:
                            is_highest = True
                        else:
                            if score >= highest_acc[0]:
                                is_highest = True
                                highest_acc[0] = score
                        highest_acc[-1] = acc_list[-1]
                return is_highest

            is_highest = judge_is_highest(acc_list, is_highest)
            
            if args.add_descriminator:
                logger.info('============== Descriminator Embedding Val test ============================')
                '''
                dec_batch_size = args.batch_size
                if args.add_input2dec:
                    dec_batch_size = 2* args.batch_size
                acc_list_dec = ver_test(mbatch, model._descriminator_embedding_module, dec_batch_size)
                is_highest = judge_is_highest(acc_list_dec, is_highest)
                '''
                    
            if is_highest:
                do_save = True
            if args.ckpt==0:
                do_save = False
            elif args.ckpt==2:
                do_save = True
            elif args.ckpt==3:
                msave = 1

            if do_save:
                _save_params(msave)
                
            logger.info('[{}]Accuracy-Highest: {}'.format(mbatch, highest_acc[-1]))

        if config.max_steps>0 and mbatch>config.max_steps:
            sys.exit(0)

    model.fit(train_dataiter,
        begin_epoch        = 0,
        num_epoch          = 999999,
        eval_data          = val_dataiter,
        kvstore            = args.kvstore,
        optimizer          = opt,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        allow_missing      = True,
        batch_end_callback = _batch_callback,
        epoch_end_callback = _epoch_callback)

def main():
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

