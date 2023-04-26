from argparse import ArgumentParser
from SegFormer.mmseg.apis import init_segmentor, show_result_pyplot
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = ArgumentParser()
    parser.add_argument('npy', help='npy file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    mat = np.load(args.npy)

    depth = mat[:, :, 0]
    cls = mat[:, :, -len(model.CLASSES):]

    plt.imshow(depth)
    plt.show()

    # show the results
    show_result_pyplot(model, np.zeros(depth.shape + (3,)), [cls.argmax(-1).astype(int)], model.PALETTE)


if __name__ == '__main__':
    main()
