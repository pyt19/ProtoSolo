import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time
from utils.helpers import makedir, find_high_activation_crop
import numpy as np
import torch
from utils.preprocess import  undo_one_image_preprocess_input_function
from find_nearest import imsave_with_bbox

def push_prototypes(dataloader,
                    prototype_network_parallel,
                    class_specific=True,
                    preprocess_input_function=None,
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None,
                    epoch_number=None,
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True,
                    log=print,
                    prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()

    log('\tpush')

    start = time.time()


    prototype_shape = prototype_network_parallel.module.prototype_shape

    n_prototypes = prototype_network_parallel.module.num_prototypes

    global_min_proto_dist = np.full(n_prototypes, np.inf)

    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                 fill_value=-1)

        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)

    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                 fill_value=-1)

        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)

    if root_dir_for_saving_prototypes != None:

        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-' + str(epoch_number))

            makedir(proto_epoch_dir)

        else:
            proto_epoch_dir = root_dir_for_saving_prototypes

    else:
        proto_epoch_dir = None


    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.module.num_classes

    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''

        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:

        np.save(file=os.path.join(proto_epoch_dir,
                             proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                arr=proto_rf_boxes)

        np.save(file=os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                arr=proto_bound_boxes)

    # 处理push
    log('\tExecuting push ...')

    '''
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))

    prototype_network_parallel.module.prototype_vectors.data.copy_(
        torch.tensor(prototype_update, dtype=torch.float32).cuda())
    '''

    # prototype_network_parallel.cuda()
    end = time.time()

    log('\tpush time: \t{0}'.format(end - start))



# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist,
                               global_min_fmap_patches,
                               proto_rf_boxes,
                               proto_bound_boxes,
                               class_specific=True,
                               search_y=None,
                               num_classes=None,
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:

        search_batch = search_batch_input

    else:
        search_batch = search_batch_input

    with torch.no_grad():

        search_batch = search_batch.cuda()
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())
    prototype_ = np.copy(prototype_network_parallel.module.prototype_vectors.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    if class_specific:

        class_to_img_index_dict = {key: [] for key in range(num_classes)}

        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape

    n_prototypes = prototype_shape[0]

    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    for j in range(n_prototypes):

        if class_specific:  # 特定类别

            target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()

            if len(class_to_img_index_dict[target_class]) == 0:
                continue

            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:, j, :]  # 原型j 对应类的 距离图

        else:

            proto_dist_j = proto_dist_[:, j, :]

        batch_min_proto_dist_j = np.amin(proto_dist_j)

        if batch_min_proto_dist_j < global_min_proto_dist[j]:

            batch_argmin_proto_dist_j = \
                list(np.unravel_index(indices=np.argmin(proto_dist_j, axis=None),
                                      shape=proto_dist_j.shape))

            if class_specific:
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            img_index_in_batch = batch_argmin_proto_dist_j[0]

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch, batch_argmin_proto_dist_j[1]]
            prototype_h_w_j = prototype_[j].reshape(7, 7)
            feature_prototype_similarity = batch_min_fmap_patch_j - prototype_h_w_j
            proto_dist_img_j = feature_prototype_similarity ** 2
            batch_min_fmap_patch_j = batch_min_fmap_patch_j.reshape(-1, 1, 1)

            global_min_proto_dist[j] = batch_min_proto_dist_j

            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            original_img_j = search_batch_input[batch_argmin_proto_dist_j[0]]

            original_img_j = undo_one_image_preprocess_input_function(original_img_j)

            original_img_j = original_img_j.numpy()

            original_img_j = np.transpose(original_img_j, (1, 2, 0))

            original_img_size = original_img_j.shape[0]

            feature_maps_min_index = np.argmin(proto_dist_[img_index_in_batch, j, :])

            prototype_j_feature_maps = protoL_input_[img_index_in_batch, feature_maps_min_index, :, :]

            makedir(os.path.join(dir_for_saving_prototypes, 'feature_maps', 'original_img'))
            makedir(os.path.join(dir_for_saving_prototypes, 'feature_maps', 'prototype_patch'))
            makedir(os.path.join(dir_for_saving_prototypes, 'feature_maps', 'act'))
            makedir(os.path.join(dir_for_saving_prototypes, 'feature_maps', 'prototype_patch_original_img'))
            makedir(os.path.join(dir_for_saving_prototypes, 'feature_maps', 'prototype_patch_act_img'))
            makedir(os.path.join(dir_for_saving_prototypes, 'feature_maps', 'mask'))

            proto_act_img_j = prototype_j_feature_maps

            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)

            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)

            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                          proto_bound_j[2]:proto_bound_j[3], :]

            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]

            if dir_for_saving_prototypes is not None:
                if prototype_img_filename_prefix is not None:
                    plt.imsave(os.path.join(dir_for_saving_prototypes, 'feature_maps', 'original_img',
                                            prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                               original_img_j,
                               vmin=0.0,
                               vmax=1.0)

                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)

                    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap

                    plt.imsave(os.path.join(dir_for_saving_prototypes, 'feature_maps', 'act',
                                            prototype_img_filename_prefix + '-original_with_self_act' + str(
                                                j) + '.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)


                    plt.imsave(os.path.join(dir_for_saving_prototypes, 'feature_maps', 'mask',
                                            prototype_img_filename_prefix + '-mask' + str(
                                                j) + '.png'),
                               rescaled_act_img_j,
                               vmin=0.0,
                               vmax=1.0)

                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(os.path.join(dir_for_saving_prototypes, 'feature_maps', 'prototype_patch',
                                            prototype_img_filename_prefix + str(j) + '.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)

                    imsave_with_bbox(os.path.join(dir_for_saving_prototypes, 'feature_maps', 'prototype_patch_original_img',
                                            prototype_img_filename_prefix + '-org-box-' + str(j) + '.png'),
                               original_img_j, proto_bound_j[0], proto_bound_j[1], proto_bound_j[2], proto_bound_j[3])

                    imsave_with_bbox(os.path.join(dir_for_saving_prototypes, 'feature_maps', 'prototype_patch_act_img',
                                                  prototype_img_filename_prefix + '-act-box-' + str(j) + '.png'),
                                     overlayed_original_img_j, proto_bound_j[0], proto_bound_j[1], proto_bound_j[2],
                                     proto_bound_j[3])

    if class_specific:
        del class_to_img_index_dict
