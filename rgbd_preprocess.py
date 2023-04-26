import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np

from SegFormer.mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from SAM.segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from natsort import natsorted
from tqdm import tqdm

sys.path.append("..")


def main():
    parser = ArgumentParser()
    parser.add_argument('path', help='Dataset path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('sf_checkpoint', help='Checkpoint file')
    parser.add_argument('sam_checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # SegFormer model
    sfm = init_segmentor(args.config, args.sf_checkpoint, device=args.device)
    # segment-anything model
    sam = sam_model_registry["default"](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    len_classes = len(sfm.CLASSES)

    folder = "result"
    try:
        os.makedirs(folder)
    except:
        pass

    out = folder + '/' + os.path.basename(args.path) + '_out'
    try:
        os.makedirs(out)
    except:
        pass

    rgb_list = natsorted(os.listdir(args.path+'/rgb'))
    depth_list = natsorted(os.listdir(args.path+'/depth'))

    for i, (rgb_file, depth_file) in tqdm(enumerate(zip(rgb_list, depth_list))):
        rgb_file = args.path + '/rgb/' + rgb_file
        depth_file = args.path + '/depth/' + depth_file

        # SegFormer inference
        segmentation = inference_segmentor(sfm, rgb_file)
        # segment-anything inference
        image = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)
        sam_masks = mask_generator.generate(image)

        cls = np.zeros(image.shape[0:2] + (len_classes,))

        for mask in sam_masks:
            seg_mask = mask['segmentation']
            vals, cnts = np.unique(segmentation * seg_mask, return_counts=True)
            cnts = cnts[vals != 0]
            vals = vals[vals != 0]

            for val, cnt in zip(vals, cnts):
                cls[:, :, val][seg_mask] += cnt

        # normalization
        total_cnt = np.sum(cls, axis=-1, keepdims=True)
        total_cnt[total_cnt == 0] = 1
        cls /= total_cnt

        cls = np.concatenate((np.expand_dims(np.load(depth_file), axis=-1), cls), axis=-1)

        np.save(out+'/'+str(i), cls)


if __name__ == '__main__':
    main()
