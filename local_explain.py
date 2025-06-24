import os
import cv2
import random
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from utils.helpers import makedir
import datetime
from settings import mean, std
import matplotlib.pyplot as plt
from settings import class_prototype_num

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


from settings import img_size, num_classes

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
    print(len(img_list))
    box = []

    for i, (image, label) in enumerate(train_loader):
        image = image.cuda()

        i1 = label.item()

        if (i1 in box):
            continue

        box.append(i1)
        image_1 = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        prototype_num = class_prototype_num
        feature_maps = ppnet.conv_features(image)

        last_layer = ppnet.last_layer
        output, min_distances = ppnet(image)

        pred = output.argmax()
        pred_value = output[0, pred]
        prototype_similarity_score = min_distances[0, pred * prototype_num: (pred + 1) * prototype_num]

        x = image
        x1 = x
        x = ppnet.conv_features(x)
        feature_maps_1c = x
        x = x.view(x.size(0), x.size(1), -1)
        prototype_vector_tmp = ppnet.prototype_vectors.squeeze(-1).squeeze(-1)
        x = x.unsqueeze(1)
        x = x.expand(-1, num_classes * prototype_num, -1, -1)
        prototype_expanded = prototype_vector_tmp.unsqueeze(0).unsqueeze(2)
        distance = torch.sum((x - prototype_expanded) ** 2, dim=-1)

        prototype_activations = ppnet.distance_2_similarity(min_distances)

        class_prototype_num = int(prototype_activations.shape[1] / ppnet.num_classes)

        reshaped_activations = prototype_activations.view(prototype_activations.shape[0], ppnet.num_classes,
                                                          class_prototype_num)

        max_activations, _ = reshaped_activations.max(dim=2)

        wkk = last_layer.weight[pred, pred]

        Gk = max_activations[0, pred]

        indeice_u = prototype_similarity_score.argmax()

        prototype_k_u = prototype_vector_tmp[pred * prototype_num + indeice_u]

        prototype_k_u = prototype_vector_tmp[pred * prototype_num + indeice_u]


        prototype_mulu = './saved_models/resnet34/004/img/epoch-24/feature_maps/act/prototype-img-original_with_self_act' + str((pred * prototype_num + indeice_u).item()) + '.png'

        prototype_mask = './saved_models/resnet34/004/img/epoch-24/feature_maps/mask/prototype-img-mask' + str(
            (pred * prototype_num + indeice_u).item()) + '.png'

        prototype_full_image = './saved_models/resnet34/004/img/epoch-24/feature_maps/original_img/prototype-img-original' + str(
            (pred * prototype_num + indeice_u).item()) + '.png'

        prototype_patch = './saved_models/resnet34/004/img/epoch-24/feature_maps/prototype_patch/prototype-img' + str(
            (pred * prototype_num + indeice_u).item()) + '.png'

        prototype_patch_act_img = './saved_models/resnet34/004/img/epoch-24/feature_maps/prototype_patch_act_img/prototype-img-act-box-' + str(
            (pred * prototype_num + indeice_u).item()) + '.png'

        prototype_patch_original_img = './saved_models/resnet34/004/img/epoch-24/feature_maps/prototype_patch_original_img/prototype-img-org-box-' + str(
            (pred * prototype_num + indeice_u).item()) + '.png'

        I_prototype = cv2.imread(prototype_mulu)[..., ::-1]
        I_prototype = I_prototype - np.min(I_prototype)
        I_prototype = I_prototype / np.max(I_prototype)

        I_prototype_mask = cv2.imread(prototype_mask)[..., ::-1]
        I_prototype_mask = I_prototype_mask - np.min(I_prototype_mask)
        I_prototype_mask = I_prototype_mask / np.max(I_prototype_mask)

        I_prototype_full_image = cv2.imread(prototype_full_image)[..., ::-1]
        I_prototype_full_image = I_prototype_full_image - np.min(I_prototype_full_image)
        I_prototype_full_image = I_prototype_full_image / np.max(I_prototype_full_image)

        I_prototype_patch = cv2.imread(prototype_patch)[..., ::-1]
        I_prototype_patch = I_prototype_patch - np.min(I_prototype_patch)
        I_prototype_patch = I_prototype_patch / np.max(I_prototype_patch)

        I_prototype_patch_act_img = cv2.imread(prototype_patch_act_img)[..., ::-1]
        I_prototype_patch_act_img = I_prototype_patch_act_img - np.min(I_prototype_patch_act_img)
        I_prototype_patch_act_img = I_prototype_patch_act_img / np.max(I_prototype_patch_act_img)

        I_prototype_patch_original_img = cv2.imread(prototype_patch_original_img)[..., ::-1]
        I_prototype_patch_original_img = I_prototype_patch_original_img - np.min(I_prototype_patch_original_img)
        I_prototype_patch_original_img = I_prototype_patch_original_img / np.max(I_prototype_patch_original_img)

        prototype_k_u_hw = prototype_k_u.view(7, 7).detach().cpu().numpy()

        prototype_k_u_hw_sali_maps = cv2.resize(prototype_k_u_hw, (224, 224), interpolation=cv2.INTER_LINEAR)

        distances = ppnet.prototype_distances(x1)

        min_distances, min_distances_index = torch.min(distances, dim=-1)

        fea_map = distance[0, pred * prototype_num + indeice_u]

        fea_map_min_distance, fea_map_min_distance_index = torch.min(fea_map, dim=-1)

        feature_maps_c = feature_maps_1c[0, fea_map_min_distance_index].detach().cpu().numpy()

        feature_maps_c_sali_maps = cv2.resize(feature_maps_c, (224, 224), interpolation=cv2.INTER_LINEAR)

        image_1 = image_1 - np.min(image_1)
        image_1 = image_1 / np.max(image_1)

        feature_maps_heatmap = cv2.applyColorMap(np.uint8(255 * feature_maps_c_sali_maps), cv2.COLORMAP_JET)
        feature_maps_heatmap = np.float32(feature_maps_heatmap) / 255
        feature_maps_heatmap = feature_maps_heatmap[..., ::-1]
        I_feature_maps = 0.5 * image_1 + 0.3 * feature_maps_heatmap
        I_feature_maps = I_feature_maps - np.min(I_feature_maps)
        I_feature_maps = I_feature_maps / np.max(I_feature_maps)

        feature_maps_c_sali_maps = feature_maps_c_sali_maps - np.min(feature_maps_c_sali_maps)
        feature_maps_c_sali_maps = feature_maps_c_sali_maps / np.max(feature_maps_c_sali_maps)
        from utils.helpers import makedir, find_high_activation_crop

        proto_bound_j = find_high_activation_crop(feature_maps_c_sali_maps)


        original_image_patch = image_1[proto_bound_j[0]:proto_bound_j[1], proto_bound_j[2]:proto_bound_j[3], :]
        from find_nearest import imsave_with_bbox

        makedir(os.path.join('./local_explain_result', str(i1)))

        imsave_with_bbox(os.path.join('./local_explain_result', str(i1), 'feature_image_with_bbox.jpg'),
                         image_1, proto_bound_j[0], proto_bound_j[1], proto_bound_j[2],
                         proto_bound_j[3])

        imsave_with_bbox(os.path.join('./local_explain_result', str(i1), 'Image_with_bbox.jpg'),
                         image_1, proto_bound_j[0], proto_bound_j[1], proto_bound_j[2],
                         proto_bound_j[3])

        plt.imsave(os.path.join('./local_explain_result', str(i1), 'I_feature_maps.jpg'), I_feature_maps)
        plt.imsave(os.path.join('./local_explain_result', str(i1), 'input_image.jpg'), image_1)
        plt.imsave(os.path.join('./local_explain_result', str(i1), 'original_image_patch.jpg'), original_image_patch)
        plt.imsave(os.path.join('./local_explain_result', str(i1), 'original_image_act.jpg'), feature_maps_c_sali_maps)

        plt.imsave(os.path.join('./local_explain_result', str(i1), 'I_prototype.jpg'), I_prototype)
        plt.imsave(os.path.join('./local_explain_result', str(i1), 'I_prototype_mask.jpg'), I_prototype_mask)
        plt.imsave(os.path.join('./local_explain_result', str(i1), 'I_prototype_full_image.jpg'), I_prototype_full_image)
        plt.imsave(os.path.join('./local_explain_result', str(i1), 'I_prototype_patch.jpg'), I_prototype_patch)
        plt.imsave(os.path.join('./local_explain_result', str(i1), 'I_prototype_patch_act_img.jpg'), I_prototype_patch_act_img)
        plt.imsave(os.path.join('./local_explain_result', str(i1), 'I_prototype_patch_original_img.jpg'), I_prototype_patch_original_img)

        print("Gk = ", Gk.item())
        print("wkk = ", wkk.item())
        print("pred_value = ", pred_value.item())

        import datetime

        filename = "lcaol_explain_data.txt"

        content = "Gk = " + str(Gk.item()) + "\n"
        content += "wkk = " + str(wkk.item()) + "\n"
        content += "pred_value = " + str(pred_value.item()) + "\n"
        content += "i1 = " + str(i1) + "\n"

        with open(os.path.join('./local_explain_result', str(i1), filename), 'w') as file:
            file.write(content)