import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from models.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from models.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from models.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features

from receptive_field import compute_proto_layer_rf_info_v2
from settings import final_add_layer_channel

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

class PPNet(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4

        self.prototype_activation_function = prototype_activation_function

        assert (self.num_prototypes % self.num_classes == 0)

        self.prototype_class_identity_s = torch.zeros(self.num_classes,
                                                    self.num_classes)

        num_prototypes_per_class_s = self.num_classes // self.num_classes

        for j in range(self.num_classes):
            self.prototype_class_identity_s[j, j // num_prototypes_per_class_s] = 1


        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes

        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1


        self.proto_layer_rf_info = proto_layer_rf_info
        self.features = features
        self.final_add_layer_channel = final_add_layer_channel
        features_name = str(self.features).upper()

        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels

        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []

            current_in_channels = first_add_on_layer_in_channels

            while (current_in_channels > self.prototype_shape[1]) or (
                    len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (
                        current_in_channels // 2))

                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))

                add_on_layers.append(nn.ReLU())

                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))

                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert (current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())

                current_in_channels = current_in_channels // 2

            self.add_on_layers = nn.Sequential(*add_on_layers)

        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.final_add_layer_channel,
                          kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.final_add_layer_channel, out_channels=self.final_add_layer_channel, kernel_size=1),
                nn.Sigmoid()
            )

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(in_features=self.num_classes, out_features=self.num_classes,
                                    bias=False)  # do not use bias

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)

        x = self.add_on_layers(x)

        return x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):

        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)
        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)
        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''

        x = x.view(x.size(0), x.size(1), -1)
        prototype_vector_tmp = self.prototype_vectors.squeeze(-1).squeeze(-1)
        x = x.unsqueeze(1)
        prototype_expanded = prototype_vector_tmp.unsqueeze(0).unsqueeze(2)
        distances = F.relu(torch.sum((x - prototype_expanded) ** 2, dim=-1))

        return distances

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log(
                (distances + 1) / (distances + self.epsilon))

        elif self.prototype_activation_function == 'linear':
            return -distances

        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        distances = self.prototype_distances(x)
        min_distances, min_distances_index = torch.min(distances, dim=-1)
        prototype_activations = self.distance_2_similarity(min_distances)
        class_prototype_num = int(prototype_activations.shape[1] / self.num_classes)
        reshaped_activations = prototype_activations.view(prototype_activations.shape[0], self.num_classes, class_prototype_num)
        max_activations, _ = reshaped_activations.max(dim=2)
        logits = self.last_layer(max_activations)

        return logits, min_distances

    def push_forward(self, x):
        ''' this method is needed for the pushing operation '''
        conv_output = self.conv_features(x)

        distances = self._l2_convolution(conv_output)

        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):

        prototypes_to_keep = list(
            set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        self.prototype_shape = list(self.prototype_vectors.size())

        self.num_prototypes = self.prototype_shape[0]

        self.last_layer.in_features = self.num_prototypes

        self.last_layer.out_features = self.num_classes

        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...], requires_grad=False)

        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):

        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):

        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''

        positive_one_weights_locations = torch.t(self.prototype_class_identity_s)

        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1

        incorrect_class_connection = incorrect_strength

        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):

        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

def construct_PPNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 512, 1, 1), num_classes=200,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):

    features = base_architecture_to_features[base_architecture](pretrained=pretrained)

    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()

    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])

    return PPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)
