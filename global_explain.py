import os
import cv2
import random
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import datetime
from settings import mean, std
import matplotlib.pyplot as plt
from utils.helpers import makedir, find_high_activation_crop


def init_seeds(seed=0):
    random.seed(seed)  # seed for module random
    np.random.seed(seed)  # seed for numpy
    torch.manual_seed(seed)  # seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # seed for current PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # seed for all PyTorch GPUs
    if seed == 0:
        # if True, causes cuDNN to only use deterministic convolution algorithms.
        torch.backends.cudnn.deterministic = True
        # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
        torch.backends.cudnn.benchmark = False

from settings import img_size, class_prototype_num

if __name__ == '__main__':

    result_path = './Result_img/'
    time_path = str(datetime.datetime.now()).replace(':', '.')
    makedir(os.path.join(result_path, time_path))
    train_test_path = 'Train_and_Test'
    push_test_path = 'Push_Test'
    iteration_train_path = 'Iteration_train'

    model_path = './trained_model_file.pth' # trained_model_file.pth

    ppnet = torch.load(model_path)

    test_path = './datasets/cub200_cropped/test_cropped'

    test_img = cv2.imread(test_path)

    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    train_dataset = datasets.ImageFolder(
        test_path,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=False)

    image = torch.zeros([3, 224, 224])

    img_list = os.listdir(test_path)

    box = []

    for i, (image, label) in enumerate(train_loader):
        image = image.cuda()

        i1 = label.item()

        if(i1 in box):
            continue

        box.append(i1)

        image_1 = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        prototype_num = class_prototype_num
        feature_maps = ppnet.conv_features(image)
        feature_maps_list = feature_maps[0, :, :, :]


        for feature_maps_list_num in range(feature_maps_list.shape[0]):
            f_m = feature_maps_list[feature_maps_list_num].detach().cpu().numpy()
            f_m_umsample = cv2.resize(f_m, (224, 224), interpolation=cv2.INTER_LINEAR)
            makedir(os.path.join('./global_explain_result', str(i1), 'feature_map'))
            makedir(os.path.join('./global_explain_result', str(i1), 'cam'))
            makedir(os.path.join('./global_explain_result', str(i1), 'cam_bbox'))
            makedir(os.path.join('./global_explain_result', str(i1), 'img_bbox'))
            makedir(os.path.join('./global_explain_result', str(i1), 'prototype_patch'))
            makedir(os.path.join('./global_explain_result', str(i1), 'original_image'))

            f_mask = f_m_umsample
            f_mask = f_mask - np.min(f_mask)
            f_mask = f_mask / np.max(f_mask)

            f_heatmap = cv2.applyColorMap(np.uint8(255 * f_mask), cv2.COLORMAP_JET)
            f_heatmap = np.float32(f_heatmap) / 255
            f_heatmap = f_heatmap[..., ::-1]
            cam = 0.5 * image_1 + 0.3 * f_heatmap
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)

            plt.imsave(os.path.join('./global_explain_result', str(i1), 'feature_map', str(feature_maps_list_num) + '.jpg'), f_m_umsample)
            plt.imsave(os.path.join('./global_explain_result', str(i1), 'cam', str(feature_maps_list_num) + '.jpg'), cam)

            f_mask_proto_bound_j = find_high_activation_crop(f_mask)

            original_image_patch = image_1[f_mask_proto_bound_j[0]:f_mask_proto_bound_j[1], f_mask_proto_bound_j[2]:f_mask_proto_bound_j[3], :]
            original_image_patch = original_image_patch - np.min(original_image_patch)
            original_image_patch = original_image_patch / np.max(original_image_patch)
            from find_nearest import imsave_with_bbox

            makedir(os.path.join('./global_explain_result', str(i1)))

            imsave_with_bbox(os.path.join('./global_explain_result', str(i1), 'cam_bbox', str(feature_maps_list_num) + '.jpg'),
                             cam, f_mask_proto_bound_j[0], f_mask_proto_bound_j[1], f_mask_proto_bound_j[2],
                             f_mask_proto_bound_j[3])

            image_x = image_1
            image_x = image_x - np.min(image_x)
            image_x = image_x / np.max(image_x)

            imsave_with_bbox(os.path.join('./global_explain_result', str(i1), 'img_bbox', str(feature_maps_list_num) + '.jpg'),
                             image_x, f_mask_proto_bound_j[0], f_mask_proto_bound_j[1], f_mask_proto_bound_j[2],
                             f_mask_proto_bound_j[3])

            plt.imsave(os.path.join('./global_explain_result', str(i1), 'prototype_patch', str(feature_maps_list_num) + '.jpg'),
                       original_image_patch)

            plt.imsave(os.path.join('./global_explain_result', str(i1), 'original_image', str(feature_maps_list_num) + '.jpg'),
                image_x)