import sys
sys.path.insert(1, "./lib")

import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.nn import functional as F
from lib.models.ddrnet_23_slim import DualResNet, BasicBlock


color_palette = [
    (0, 0, 0),
    (150, 100, 100),
    (220, 20, 60),
    (128, 64, 128),
    (157, 234, 50),
    (244, 35, 232),
    (220, 220, 0),
    (250, 170, 30),
    (0, 0, 142)
]


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", default="", help="Folder containig images to be predicted")
    parser.add_argument("--image_ext", default="png", help="Extension of your images")
    parser.add_argument("--resize", default="", help="(width, height) resize size, if not, leave blank")
    parser.add_argument("--show_image", action="store_true", help="Show live predicting images")
    parser.add_argument("--save_folder", default="", help="folder to save predicted images")
    parser.add_argument("--num_classes", default=9, type=int, help="Number of classes")
    parser.add_argument("--model_file", default="", help="Path to file containing your model")

    return parser.parse_args()


def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    return image


def fill_color_palette(labels, segmentation):
    canvas = np.zeros((*segmentation.shape, 3), np.uint8)
    for idx, label in enumerate(labels):
        indices = np.where(segmentation == idx)
        canvas[indices[0], indices[1], :] = label
    return canvas

def apply_canvas(image, segmentation):
    h, w, _ = image.shape
    canvas = np.empty((h*2, w, 3), np.uint8)
    canvas[:h] = image
    canvas[h:] = segmentation

    return canvas


def main(args):
    data_folder = Path(args.data_folder)
    assert data_folder.exists(), "Data folder not found bruh"

    images = list(data_folder.rglob(f"*.{args.image_ext}"))
    print(f"Got {len(images)} images")

    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=args.num_classes, planes=32, spp_planes=128, head_planes=64, augment=True)
    pretrained_dict = torch.load(args.model_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        print('=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.cuda()
    print("Done loading model")

    model.eval()
    with torch.no_grad():
        for image in tqdm(images):
            file_name = image.name
            image = cv2.imread(str(image))
            image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_NEAREST)
            pred_image = input_transform(image.copy())
            pred_image = pred_image.transpose((2, 0, 1))
            pred_image = np.expand_dims(pred_image, axis=0)
            pred_image = torch.from_numpy(pred_image).cuda()

            result = model(pred_image)[1]
            result = F.interpolate(
                input=result, size=(512, 1024),
                mode='bilinear', align_corners=True
            )
            _, pred = torch.max(result, dim=1)
            pred = pred.squeeze(0).cpu().numpy()
            palette = fill_color_palette(color_palette, pred)
            predicted_image = apply_canvas(image, palette)
            if args.show_image:
                cv2.imshow("Segmented", predicted_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if args.save_folder != "":
                save_file = Path(args.save_folder)/file_name
                if not save_file.parent.exists():
                    save_file.parent.mkdir(parents=True)
                cv2.imwrite(str(save_file), predicted_image)

    if args.show_image:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = arguments()
    main(args)