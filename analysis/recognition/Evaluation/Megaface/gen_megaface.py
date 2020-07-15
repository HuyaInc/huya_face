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


def get_feature(imgs, nets):
  count = len(imgs)
  data = mx.nd.zeros(shape = (count*2, 3, imgs[0].shape[1], imgs[0].shape[2]))
  for idx, img in enumerate(imgs):
    img = img[::-1,:,:] #to rgb
    # img = nd.transpose( img, (2,0,1) )
    for flipid in [0,1]:
      _img = nd.array(img)
      if flipid==1:
        _img = _img[:,:,::-1]
      # _img = nd.array(_img)
      data[count*flipid+idx] = _img

  F = []
  for net in nets:
    db = mx.io.DataBatch(data=(data,))
    net.model.forward(db, is_train=False)
    x = net.model.get_outputs()[0].asnumpy()
    embedding = x[0:count,:] + x[count:,:]
    embedding = sklearn.preprocessing.normalize(embedding)
    #print('emb', embedding.shape)
    F.append(embedding)
  F = np.concatenate(F, axis=1)
  F = sklearn.preprocessing.normalize(F)
  #print('F', F.shape)
  return F


def put_feature(imgs, nets, out_put_list, q_exc):
  try:
    _st = time.time()
    count = len(imgs)
    data = mx.nd.zeros(shape=(count * 2, 3, imgs[0].shape[1], imgs[0].shape[2]))
    for idx, img in enumerate(imgs):
      #img = img[::-1, :, :]  # to rgb
      # img = nd.transpose( img, (2,0,1) )
      for flipid in [0, 1]:
        _img = nd.array(img)
        if flipid == 1:
          _img = _img[:, :, ::-1]
        # _img = nd.array(_img)
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
    ft = time.time() - _st
    print('Feature extracting time used:', ft)
    for i, k in enumerate(out_put_list):
      q_work = q_exc[i%len(q_exc)]
      data = (F[i], k)
      while True:
        if q_work.full():
          continue
        else:
          q_work.put(data)
          break  
   # et = time.time() - _st - ft
   # print('Entering queue time used:', et)

  except Exception as e:
    traceback.print_exc()
    print('det_img error:', e)
    for q in q_exc:
      q.put(None)
    return


def write(args, q_exc):
  while True:
    #st = time.time()
    data = q_exc.get()
    if data is None:
      break
    feature, path = data[0], data[1]
    feature = list(feature)
    with open(path, 'wb') as f:
      f.write(struct.pack('4i', len(feature), 1, 4, 5))
      f.write(struct.pack("%df" % len(feature), *feature))
    #et = time.time() - st
    #print('Writing time used:', et)

def write_bin(buffer):
  feature, path = buffer[0], buffer[1]
  feature = list(feature)
  with open(path, 'wb') as f:
    f.write(struct.pack('4i', len(feature),1,4,5))
    f.write(struct.pack("%df"%len(feature), *feature))


def generate_dic(args, img_list):
  out_dic = {}
  print("Creating dictionary start")
  for line in img_list:
    line = [i.strip() for i in line.strip().split('\t')]
    img_path = line[-1]
    image_path = img_path.strip()
    _path = image_path.split('/')
    if args.dataset =='facescrub':
      a, b = _path[-2], _path[-1]
      dataset_out = os.path.join(args.output, args.dataset)
      out_dir = os.path.join(dataset_out, a)
     # if not os.path.exists(out_dir):
     #   os.makedirs(out_dir)
    else:
      a_pre, a, b =_path[-3], _path[-2], _path[-1]
      dataset_out = os.path.join(args.output, args.dataset)
      out_dir = os.path.join(dataset_out, a_pre, a)
      #if not os.path.exists(out_dir):
      #  os.makedirs(out_dir)
    out_path = os.path.join(out_dir, b + "_%s.bin" % (args.algo))
    out_dic[int(line[0])] = out_path
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
    assert len(vec)>1
    prefix = vec[0]
    epoch = int(vec[1])
    print('loading',prefix, epoch)
    net = edict()
    net.ctx = ctx
    net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = net.sym.get_internals()
    net.sym = all_layers['{}fc1_output'.format(args.name_prefix)]
    net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names = None)
    net.model.bind(data_shapes=[('data', (48, 3, image_shape[1], image_shape[2]))])
    net.model.set_params(net.arg_params, net.aux_params)
    nets.append(net)

  with open(args.dataset_lst) as f:
    img_lst = f.readlines()
  dataset_dic = generate_dic(args, img_lst)
  total_nums = len(img_lst)
  i, j = total_nums // args.batch_size, total_nums % args.batch_size
  count = 0
  q_exc = [multiprocessing.Queue(1024) for v in range(args.num_threads)]
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
    print('Data generation time used:', time.time()-start)
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

  parser.add_argument('--rec_path', type=str, help='', default='~/huya_face/face_datasets/megaface_data/eval_set/facescrub.rec')
  parser.add_argument('--idx_path', type=str, help='', default='~/huya_face/face_datasets/megaface_data/eval_set/facescrub.idx')
  parser.add_argument('--batch_size', type=int, help='', default=32)
  parser.add_argument('--image_size', type=str, help='', default='3,112,112')
  parser.add_argument('--gpu', type=str, help='', default=0)
  parser.add_argument('--num_threads', type=int, help='', default=8)
  parser.add_argument('--algo', type=str, help='', default='insightface')
  parser.add_argument('--dataset', type=str, help='', default='facescrub')
  parser.add_argument('--dataset_lst', type=str, help='', default='~/huya_face/face_datasets/megaface_data/eval_set/facescrub.lst')
  parser.add_argument('--output', type=str, help='', default='~/huya_face/feature_out')
  parser.add_argument('--model', type=str, help='', default='~/huya_face/models/model-r100-ii/model,0')
  parser.add_argument('--name_prefix', type=str, help='', default='')
  return parser.parse_args(argv)

if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))

