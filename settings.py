datasets_name_txt = 'CUB-200-2011'
training_set_size = 244320
push_set_size = 8144
test_set_size = 8041

train_batch_size = 80
test_batch_size = 80
train_push_batch_size = 80

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

iteration_ = 5
base_architecture = 'resnet34'
img_size = 224
final_add_layer_channel = 64
prototype_shape = (2000, 49, 1, 1)
prototype_num = 2000
num_classes = 200
class_prototype_num = 10
prototype_activation_function = 'log'
add_on_layers_type = 'regular'
experiment_run = '004'

data_path = './datasets/cub200_cropped/'
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}

joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 1000
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 1 == 0]
