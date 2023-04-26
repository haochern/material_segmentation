import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np

from SAM.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from SegFormer.mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot

sys.path.append("segment_anything")


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
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

    # SegFormer inference
    segmentation = inference_segmentor(sfm, args.img)
    # segment-anything inference
    image = cv2.cvtColor(cv2.imread(args.img), cv2.COLOR_BGR2RGB)
    sam_masks = mask_generator.generate(image)

    # matrix with shape(HxWxCLASSES)
    len_classes = len(sfm.CLASSES)
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

    folder = "result"
    try:
        os.makedirs(folder)
    except:
        pass
    np.save('result/demo', cls)

    # show the results
    show_result_pyplot(sfm, args.img, [cls.argmax(-1).astype(int)], sfm.PALETTE)


if __name__ == '__main__':
    main()
