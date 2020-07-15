# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import traceback

from easydict import EasyDict as edict
import time
import sys
import numpy as np
import argparse
import struct
import cv2
import multiprocessing
import sklearn
from sklearn.preprocessing import normalize
import mxnet as mx
from mxnet import ndarray as nd

feature_dim = 512
feature_ext = 1

def put_feature(imgs, nets, out_put_list, q_exc):
    try:
        count = len(imgs)
        data = mx.nd.zeros(shape=(count * 2, 3, imgs[0].shape[1], imgs[0].shape[2]))
        for idx, img in enumerate(imgs):
            for flipid in [0, 1]:
                _img = nd.array(img)
                if flipid == 1:
                    _img = _img[:, :, ::-1]
                data[count * flipid + idx] = _img

        F = []
        for net in nets:
            db = mx.io.DataBatch(data=(data,))
            net.model.forward(db, is_train=False)
            x = net.model.get_outputs()[0].asnumpy()
            embedding = x[0:count, :] + x[count:, :]
            embedding = sklearn.preprocessing.normalize(embedding)
            F.append(embedding)
        F = np.concatenate(F, axis=1)
        F = sklearn.preprocessing.normalize(F)
        for i, k in enumerate(out_put_list):
            q_work = q_exc[i % len(q_exc)]
            data = (F[i], k)
            while True:
                if q_work.full():
                    continue
                else:
                    q_work.put(data)
                    break

    except Exception as e:
        traceback.print_exc()
        print('det_img error:', e)
        for q in q_exc:
            q.put(None)
        return


def write(args, q_exc):
    while True:
        data = q_exc.get()
        if data is None:
            break
        v= data[0]
        path, label = data[1][0], data[1][1]
        if label==1:
            feature = np.full( (feature_dim+feature_ext,), 100, dtype=np.float32)
            feature[0:feature_dim] = v
        else:
            feature = np.full((feature_dim + feature_ext,), 0, dtype=np.float32)
            feature[0:feature_dim] = v
        feature = list(feature)
        with open(path, 'wb') as f:
            f.write(struct.pack('4i', len(feature), 1, 4, 5))
            f.write(struct.pack("%df" % len(feature), *feature))


def generate_output_dic(args, img_list):
    out_dic = {}
    mf_noise_map = {}
    for line in open(args.megaface_noises, 'r'):
        if line.startswith('#'):
            continue
        line = line.strip()
        _vec = line.split("\t")
        if len(_vec) > 1:
            line = _vec[1]
        mf_noise_map[line] = 1

    print("Creating dictionary start")
    for line in img_list:
        clean_label = 0
        line = [i.strip() for i in line.strip().split('\t')]
        img_path = line[-1]
        image_path = img_path.strip()
        _path = image_path.split('/')
        a_pre, a, b = _path[-3], _path[-2], _path[-1]
        dataset_out = os.path.join(args.output, args.dataset)
        out_dir = os.path.join(dataset_out, a_pre, a)
        out_path = os.path.join(out_dir, b + "_%s.bin" % (args.algo))
        bb = '/'.join([a_pre, a, b])
        if bb in mf_noise_map:
            clean_label = 1
        out_dic[int(line[0])] = (out_path, clean_label)
    print("Creating dictionary end, the length of dictionary is", len(out_dic))
    return out_dic


def main(args):
    print(args)
    if len(args.gpu) == 1:
        gpuid = int(args.gpu)
        ctx = mx.gpu(gpuid)
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpu.split(',')]
    nets = []
    image_shape = [int(x) for x in args.image_size.split(',')]
    for model in args.model.split('|'):
        vec = model.split(',')
        assert len(vec) > 1
        prefix = vec[0]
        epoch = int(vec[1])
        print('loading', prefix, epoch)
        net = edict()
        net.ctx = ctx
        net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = net.sym.get_internals()
        net.sym = all_layers['{}fc1_output'.format(args.name_prefix)]
        net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names=None)
        net.model.bind(data_shapes=[('data', (args.batch_size, 3, image_shape[1], image_shape[2]))])
        net.model.set_params(net.arg_params, net.aux_params)
        nets.append(net)

    with open(args.dataset_lst) as f:
        img_lst = f.readlines()
    dataset_dic = generate_output_dic(args, img_lst)
    total_nums = len(img_lst)
    i, j = total_nums // args.batch_size, total_nums % args.batch_size
    count = 0
    q_exc = [multiprocessing.Queue(2048) for v in range(args.num_threads)]
    write_process = [multiprocessing.Process(target=write, args=(args, q_exc[v])) \
                     for v in range(args.num_threads)]
    for p in write_process:
        p.start()

    data_iter = mx.image.ImageIter(batch_size=args.batch_size, data_shape=(3, 112, 112),
                                   path_imgrec=args.rec_path,
                                   part_index=args.idx_path)
    data_iter.reset()
    while count <= i:
        start = time.time()
        batch = data_iter.next()
        data = batch.data[0]
        data.asnumpy()
        out_path_list = []
        for value in batch.label[0]:
            idx = int(value.asnumpy())
            out_path = dataset_dic.get(idx)
            out_path_list.append(out_path)
            if count == i and j != 0:
                out_path_list = out_path_list[:j]
        put_feature(data, nets, out_path_list, q_exc)
        elapse = time.time() - start
        print(count, '/', i, 'Total Time used:', elapse)
        count += 1

    for q in q_exc:
        q.put(None)
    for p in write_process:
        p.join()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--rec-path', type=str, help='',
                        default='~/huya_face/face_datasets/megaface_data/eval_set/megaface.rec')
    parser.add_argument('--idx-path', type=str, help='',
                        default='~/huya_face/face_datasets/megaface_data/eval_set/megaface.idx')
    parser.add_argument('--batch-size', type=int, help='', default=32)
    parser.add_argument('--image-size', type=str, help='', default='3,112,112')
    parser.add_argument('--gpu', type=str, help='', default=0)
    parser.add_argument('--num-threads', type=int, help='', default=16)
    parser.add_argument('--algo', type=str, help='', default='insightface')
    parser.add_argument('--dataset', type=str, help='', default='megaface')
    parser.add_argument('--dataset_lst', type=str, help='',
                        default='~/huya_face/face_datasets/megaface_data/eval_set/megaface.lst')
    parser.add_argument('--output', type=str, help='',
                        default='~/huya_face/feature_out_clean')
    parser.add_argument('--model', type=str, help='',
                        default='~/huya_face/models/model-r100-ii/model,0')
    parser.add_argument('--megaface-noises', type=str, help='',
                        default='~/huya_face/face_datasets/megaface_data/eval_set/megaface_noises.txt')
    parser.add_argument('--name_prefix', type=str, default='')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


