from __future__ import division
import sys
import os
import numpy as np
from PIL import Image
import src.siamese as siam
from src.tracker import tracker
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
#from skimage.color import rgb2gray
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from src.visualization import show_frame, show_crops, show_scores
import cv2

def main():
    # avoid printing TF debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # TODO: allow parameters from command line or leave everything in json files?
    hp, evaluation, run, env, design = parse_arguments()
    # Set size for use with tf.image.resize_images with align_corners=True.
    # For example,
    #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
    # instead of
    # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    # build TF graph once for all
    filename, image, templates_z, scores = siam.build_tracking_graph(final_score_sz, design, env)

    # iterate through all videos of evaluation.dataset
    if evaluation.video == 'all':
        dataset_folder = os.path.join(env.root_dataset, evaluation.dataset)
        videos_list = [v for v in os.listdir(dataset_folder)]
        videos_list.sort()
        nv = np.size(videos_list)
        speed = np.zeros(nv * evaluation.n_subseq)
        precisions = np.zeros(nv * evaluation.n_subseq)
        precisions_auc = np.zeros(nv * evaluation.n_subseq)
        ious = np.zeros(nv * evaluation.n_subseq)
        lengths = np.zeros(nv * evaluation.n_subseq)
        for i in range(nv):
            images_arr, gt, frame_name_list, frame_sz, n_frames = _init_video(env, evaluation, videos_list[i])
            starts = np.rint(np.linspace(0, n_frames - 1, evaluation.n_subseq + 1))
            starts = starts[0:evaluation.n_subseq]
            for j in range(evaluation.n_subseq):
                start_frame = int(starts[j])
                gt_ = gt[start_frame:, :]
                frame_name_list_ = frame_name_list[start_frame:]
                pos_x, pos_y, target_w, target_h = region_to_bbox(gt_[0])
                idx = i * evaluation.n_subseq + j
                bboxes, speed[idx] = tracker(hp, run, design, frame_name_list_, pos_x, pos_y,
                                                                     target_w, target_h, final_score_sz, filename,
                                                                     image, templates_z, scores, start_frame)
                lengths[idx], precisions[idx], precisions_auc[idx], ious[idx] = compile_results(gt, bboxes, evaluation.dist_threshold)
                print(str(i) + ' -- ' + videos_list[i] + ' -- Precision: ' + "%.2f" % precisions[idx] + ' -- Precisions AUC: ' + "%.2f" % precisions_auc[idx] + ' -- IOU: ' + "%.2f" % ious[idx] + ' -- Speed: ' + "%.2f" % speed[idx] + ' --')
        tot_frames = np.sum(lengths)
        mean_precision = np.sum(precisions * lengths) / tot_frames
        mean_precision_auc = np.sum(precisions_auc * lengths) / tot_frames
        mean_iou = np.sum(ious * lengths) / tot_frames
        mean_speed = np.sum(speed * lengths) / tot_frames
        print('-- Overall stats (averaged per frame) on ' + str(nv) + ' videos (' + str(tot_frames) + ' frames) --')
        print(' -- Precision ' + "(%d px)" % evaluation.dist_threshold + ': ' + "%.2f" % mean_precision + ' -- Precisions AUC: ' + "%.2f" % mean_precision_auc + ' -- IOU: ' + "%.2f" % mean_iou + ' -- Speed: ' + "%.2f" % mean_speed + ' --')

    else:
        images_arr, gt, frame_name_list, _, _ = _init_video(env, evaluation, evaluation.video)
        pos_x, pos_y, target_w, target_h = region_to_bbox(gt[evaluation.start_frame]) 
      
        bboxes, speed = tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz,
                                filename, image, templates_z, scores, evaluation.start_frame)        
                            
        num_frames = np.size(frame_name_list)
        bboxes_final = np.zeros((num_frames,4))

        lk_params = dict( winSize  = (5,5),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 0.03))

        bboxes_final = np.zeros((num_frames, 4))
        for i in range(1,len(images_arr)-1):
            # Create some random colors
            color = np.random.randint(0,255,(100,3))

            # Take first frame and find corners in it
            #ret, old_frame = cap.read()
            frame = images_arr[i+1]
            old_frame = images_arr[i]
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            p0 = np.zeros((1,1,2), dtype=np.float32)
            bbox_i = bboxes[i]
            c, r, w, h = int(bbox_i[0]), int(bbox_i[1]), int(bbox_i[2]), int(bbox_i[3])
            p0[0,0,0] = c
            p0[0,0,1] = r
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st==1]
            bboxes_final[i,:] = p1[0][0][0], p1[0][0][1], w, h
            good_old = p0[st==1]   
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            r1 = patches.ConnectionPatch((a,b),(c,d),'data','data',arrowstyle="-|>") 
            r2 = patches.Circle((a,b),5,color='r')#((x,y), w, h, linewidth=2, edgecolor='r', fill=False)
            ax.imshow(np.uint8(frame))
            ax.add_patch(r2)
            ax.add_patch(r1)
            plt.ion()
            plt.show()
            plt.pause(0.001)
            plt.clf()
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
        _, precision, precision_auc, iou = _compile_results(gt, bboxes_final, evaluation.dist_threshold)
        print(evaluation.video + ' -- Precision ' + "(%d px)" % evaluation.dist_threshold + ': ' + "%.2f" % precision + ' -- Precision AUC: ' + "%.2f" % precision_auc + ' -- IOU: ' + "%.2f" % iou + ' -- Speed: ' + "%.2f" % speed + ' --')


def _compile_results(gt, bboxes, dist_threshold):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)

    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(bboxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
    precision = sum(new_distances < dist_threshold)/np.size(new_distances) * 100

    # find above result for many thresholds, then report the AUC
    thresholds = np.linspace(0, 25, n_thresholds+1)
    thresholds = thresholds[-n_thresholds:]
    # reverse it so that higher values of precision goes at the beginning
    thresholds = thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i] = sum(new_distances < thresholds[i])/np.size(new_distances)

    # integrate over the thresholds
    precision_auc = np.trapz(precisions_ths)    

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100

    return l, precision, precision_auc, iou

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def _init_video(env, evaluation, video):
    video_folder = os.path.join(env.root_dataset, evaluation.dataset, video)
    frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
    
    frame_name_list = [os.path.join(env.root_dataset, evaluation.dataset, video, '') + s for s in frame_name_list]
    frame_name_list.sort()

    images_arr = []
    for i in range(len(frame_name_list)):
        im = cv2.imread(frame_name_list[i])
        #im = rgb2gray(im)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        images_arr.append(im)

    with Image.open(frame_name_list[0]) as img:
        plt.imshow(img)
        frame_sz = np.asarray(img.size)
        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    # read the initialization from ground truth
    gt_file = os.path.join(video_folder, 'groundtruth.txt')
    gt = np.genfromtxt(gt_file, delimiter=',')
    n_frames = len(frame_name_list)
    assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'

    return images_arr,gt, frame_name_list, frame_sz, n_frames


def _compute_distance(boxA, boxB):
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist


def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou


if __name__ == '__main__':
    sys.exit(main())