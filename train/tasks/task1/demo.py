#!/usr/bin/python3
from __future__ import print_function

import os
import sys
import torch
import torch.backends.cudnn as cudnn
import argparse
import cv2
import numpy as np
from collections import OrderedDict

sys.path.append(os.getcwd() + '/../../src')

from config import cfg
from prior_box import PriorBox
from detection import Detect
from nms import nms
from utils import decode
from timer import Timer
from yufacedetectnet import YuFaceDetectNet


parser = argparse.ArgumentParser(description='Face and Mask Detection')
parser.add_argument('-m', '--trained_model', default='weights/yunet_final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--image_file', default='t1.jpg', type=str, help='the image file to be detected')
parser.add_argument('--conf_thresh', default=0.5, type=float, help='conf_thresh')
parser.add_argument('--top_k', default=20, type=int, help='top_k')
parser.add_argument('--nms_thresh', default=0.5, type=float, help='nms_thresh')
parser.add_argument('--keep_top_k', default=20, type=int, help='keep_top_k')
parser.add_argument('--device', default='cuda:0', help='which device the program will run on. cuda:0, cuda:1, ...')
args = parser.parse_args()

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


labels = ('_background_', 'face', 'mask')
num_classes = 3
detect = Detect(num_classes, 0, args.top_k, args.conf_thresh, args.nms_thresh)

if __name__ == '__main__':
    # img_dim = 320
    device = torch.device(args.device) 
    torch.set_grad_enabled(False)

    # net and model
    net = YuFaceDetectNet(phase='test', size=None )    # initialize detector
    net = load_model(net, args.trained_model, True)
    # net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')

    ## Print model's state_dict
    #print("Model's state_dict:")
    #for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    
    cudnn.benchmark = True
    net = net.to(device)

    _t = {'forward_pass': Timer(), 'misc': Timer()}
    # testing begin
    img_raw = cv2.imread(args.image_file, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape

    #img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    _t['forward_pass'].tic()
    loc, conf = net(img)  # forward pass
    _t['forward_pass'].toc()
    _t['misc'].tic()

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    
    detections = detect(loc, conf, priors)
    # detections = out.data
    print(detections.size())
    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    # scale = scale.to(device)
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6:
            score = detections[0,i,j,0]
            label_name = labels[i]
            display_txt = '%s: %.2f'%(label_name, score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            j+=1
            pts = (int(pt[0]), int(pt[1]))
            pte = (int(pt[2]), int(pt[3]))
            cx = int(pt[0])
            cy = int(pt[1]) + 12
            cv2.rectangle(img_raw, pts, pte, (0, 255, 0), 2)
            cv2.putText(img_raw, label_name, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    cv2.imshow('facemask', img_raw)
    cv2.waitKey(0)
