import os
import sys
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import pickle
import argparse
import numpy as np

sys.path.append(os.getcwd() + '/../../src')
from prior_box import PriorBox
from timer import Timer
from detection import Detect
from data import FaceRectLMDataset, detection_collate
from yufacedetectnet import YuFaceDetectNet

parser = argparse.ArgumentParser(description='Face and Mask Detection')
parser.add_argument('-m', '--trained_model', default='weights/yunet_final.pth', type=str, help='Trained state_dict file path to open')
parser.add_argument('--data_dir', default='../../data/val', type=str, help='the image file to be detected')
parser.add_argument('--conf_thresh', default=0.5, type=float, help='conf_thresh')
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

def bbox_iou(box1, box2):
    mx = min(box1[0], box2[0])
    Mx = max(box1[2], box2[2])
    my = min(box1[1], box2[1])
    My = max(box1[3], box2[3])
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def evaluate_net(save_folder, net, dataset, detect):
    num_images = len(dataset)
    # print("num_images: %d" %(num_images))
    gt_dict = {}
    est_dict = {}

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)
        img = im.unsqueeze(0)
        img = img.to(device)
        # print("image w: %d, h: %d" %(w, h))

        _t['im_detect'].tic()
        loc, conf = net(img)
        detect_time = _t['im_detect'].toc(average=False)

        priorbox = PriorBox(cfg, image_size=(h, w))
        priors = priorbox.forward()
        priors = priors.to(device)

        detections = detect(loc, conf, priors)
        scale = torch.Tensor([w, h, w, h])
        # skip l = 0, because it's the background class
        eval_det = np.empty((0, 5))
        for k in range(1, detections.size(1)):
            l = 0
            while detections[0,k,l,0] >= args.conf_thresh:
                score = detections[0,k,l,0].cpu().numpy()
                bndbox = (detections[0,k,l,1:]*scale).cpu().numpy()
                label_name = labels[k]
                l+=1
                cls_dets = np.hstack((bndbox, np.array(k))).astype(np.float32, copy=False)
                eval_det = np.vstack((eval_det, cls_dets))
        est_dict[i] = eval_det
        gt_dict[i] = gt
        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))
    evaluate_detections(est_dict, gt_dict)

def evaluate_detections(allDets, allGT):
    iou_thresh = 0.5
    proposals = 0.0
    total = 0.0
    total_face = 0.0
    correct = 0.0
    correct_face = 0.0
    eps = 1e-5
    for idx, cls_dets in allDets.items():
        GT = allGT[idx]
        proposals = proposals + allDets[idx].size
        total = total + len(GT)
        for i in range(len(GT)):
            box_gt = GT[i][:4]
            label_gt = GT[i][-1]
            best_iou = 0
            best_j = -1
            if (label_gt == 1):
                total_face = total_face + 1
            for j in range(len(cls_dets)):
                bb = cls_dets[j][:4]
                label = cls_dets[j][4]
                iou = bbox_iou(box_gt, bb)
                if iou > best_iou:
                    best_j = j
                    best_iou = iou
            if best_iou > iou_thresh and label == label_gt:
                correct = correct + 1
                if (label == 1):
                    correct_face = correct_face + 1

    precision = 1.0 * correct / (proposals + eps)
    recall = 1.0 * correct/ (total + eps)
    fscore = 2.0 * precision * recall / (precision + recall + eps)
    face_mAP = correct_face / (total_face + eps)
    mask_mAP = (correct - correct_face) / (total - total_face + eps)
    mAP = correct / total
    print("precision:{:.3f} recall:{:.3f} fscore:{:.3f}\
        \nface mAP:{:.3f} mask mAP:{:.3f} mAP:{:.3f}".format(\
            precision, recall, fscore, face_mAP, mask_mAP, mAP))


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
    evaluate_net(args.save_folder, net, valdata, detect)
