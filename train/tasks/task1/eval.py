import os
import sys
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd() + '/../../src')
from config import cfg
from evaluator import Evaluator
from prior_box import PriorBox
from timer import Timer
from detection import Detect
from data import FaceRectLMDataset, detection_collate
from yufacedetectnet import YuFaceDetectNet

parser = argparse.ArgumentParser(description='Face and Mask Detection')
parser.add_argument('-m', '--trained_model', default='weights/yunet_final.pth', type=str, help='Trained state_dict file path to open')
parser.add_argument('--data_dir', default='../../data/val', type=str, help='the image file to be detected')
parser.add_argument('--conf_thresh', default=0.6, type=float, help='conf_thresh')
parser.add_argument('--top_k', default=100, type=int, help='top_k')
parser.add_argument('--nms_thresh', default=0.5, type=float, help='nms_thresh')
parser.add_argument('--device', default='cuda:0', help='which device the program will run on. cuda:0, cuda:1, ...')
parser.add_argument('--save_folder', default='eval/', type=str, help='File path to save results')
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

def plot_save_result(cfg, results, classes, savePath):
    
    
    plt.rcParams['savefig.dpi'] = 80
    plt.rcParams['figure.dpi'] = 130

    acc_AP = 0
    validClasses = 0
    fig_index = 0

    for cls_index, result in enumerate(results):
        if result is None:
            raise IOError('Error: Class %d could not be found.' % classId)

        cls = result['class']
        precision = result['precision']
        recall = result['recall']
        average_precision = result['AP']
        acc_AP = acc_AP + average_precision
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']
        npos = result['total positives']
        total_tp = result['total TP']
        total_fp = result['total FP']

        fig_index+=1
        plt.figure(fig_index)
        plt.plot(recall, precision, cfg['colors'][cls_index], label='Precision')
        plt.xlabel('recall')
        plt.ylabel('precision')
        ap_str = "{0:.2f}%".format(average_precision * 100)
        plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(cls), ap_str))
        plt.legend(shadow=True)
        plt.grid()
        plt.savefig(os.path.join(savePath, cls + '.png'))
        plt.show()
        plt.pause(0.05)


    mAP = acc_AP / fig_index
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)

def evaluate_net(net, dataset, detect):
    num_images = len(dataset)
    # print("num_images: %d" %(num_images))
    classes = []
    num_cls_gt = {}
    gt_dict = {}
    det_dict = {}

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    for i in range(num_images):
        im, gts, h, w = dataset.pull_item(i)
        img = im.unsqueeze(0)
        img = img.to(device)
        # print("image w: %d, h: %d" %(w, h))

        # ground truth
        idxOfImage = i
        for gt in gts:
            gcls = labels[np.int(gt[-1])]
            gbox = gt[:4].tolist()
            if gcls not in classes:
                classes.append(gcls)
                gt_dict[gcls] = {}
                num_cls_gt[gcls] = 0             
            num_cls_gt[gcls] += 1
            gbox.append(0)
            if idxOfImage not in gt_dict[gcls]:
                gt_dict[gcls][idxOfImage] = []
            gt_dict[gcls][idxOfImage].append(gbox)

        _t['im_detect'].tic()
        loc, conf = net(img)
        detect_time = _t['im_detect'].toc(average=False)

        priorbox = PriorBox(cfg, image_size=(h, w))
        priors = priorbox.forward()
        priors = priors.to(device)

        detections = detect(loc, conf, priors)
        scale = torch.Tensor([w, h, w, h])
        # skip l = 0, because it's the background class
        for k in range(1, detections.size(1)):
            l = 0
            while detections[0,k,l,0] >= args.conf_thresh:
                score = detections[0,k,l,0].cpu().numpy()
                bndbox = (detections[0,k,l,1:]*scale).cpu().numpy()
                dcls = labels[k]
                l+=1
                one_box = bndbox.tolist()
                one_box.append(np.float(score))
                one_box.append(idxOfImage)

                if gcls not in det_dict:
                    det_dict[gcls]=[]
                det_dict[gcls].append(one_box)
        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))
    evaluator = Evaluator(cfg)
    return evaluator.GetPascalVOCMetrics(classes, gt_dict, num_cls_gt, det_dict)


img_dim = 320
rgb_mean = (0, 0, 0)
num_classes = 3
labels = ('_background_', 'face', 'mask')
device = torch.device(args.device) 
detect = Detect(num_classes, 0, args.top_k, args.conf_thresh, args.nms_thresh)

if __name__ == '__main__':
    # load net
    num_classes = len(labels)   
    torch.set_grad_enabled(False)                  
    net = YuFaceDetectNet(phase='test', size=None)    # initialize detector
    net = load_model(net, args.trained_model, False)
    net.eval()
    cudnn.benchmark = True
    net = net.to(device)
    print('Finished loading model!')

    # load data
    valdata = FaceRectLMDataset(args.data_dir, img_dim, rgb_mean)

    # evaluation
    results, classes = evaluate_net(net, valdata, detect)
    savePath = 'weights'
    plot_save_result(cfg, results, classes, savePath)
