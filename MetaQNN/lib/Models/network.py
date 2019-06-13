########################
# importing libraries
########################
# system libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

# custom libraries
import lib.Models.state_space_parameters as state_space_parameters


def spatial_pyramid_pooling(inp, level):
    """
    spatial pyramid pooling (spp) layer
    
    Parameters:
        inp (tuple): 4 channel input (bt,ch,r,c)
        level (int): no of levels of pooling
    
    Returns:
        output (torch.Tensor) from the spp layer
    """

    if inp.size(2) != inp.size(3):
        raise ValueError('spp only works on square/symmetric feature inputs')

    assert inp.dim() == 4
    output = []

    # iterating over spp scales
    for i in range(1, level + 1):
        kernel_size = (int(np.ceil(inp.size(2) / (1.0 * i))), int(np.ceil(inp.size(3) / (1.0 * i))))
        stride_size = (int(np.floor(inp.size(2) / (1.0 * i))), int(np.floor(inp.size(3) / (1.0 * i))))
        level_out = F.max_pool2d(inp, kernel_size=kernel_size, stride=stride_size)
        output.append(level_out.view(inp.size()[0], -1))

    final_out = torch.cat(output, 1)
    return final_out.view(final_out.size(0), -1)


class WRNBasicBlock(nn.Module):
    """
    builds a wrn basic block ('Deep Residual Learning for Image Recognition': https://arxiv.org/abs/1512.03385)

    Parameters:
        in_planes (int): number of input channels
        out_planes (int): number of output channels
        stride (int): stride value for conv filters in wrn block
        dropout (float): weight drop probability for dropout
        batchnorm (float): epsilon value for batch normalization

    """

    def __init__(self, in_planes, out_planes, stride, dropout=0.0, batchnorm=1e-3):
        super(WRNBasicBlock, self).__init__()

        self.batchnorm = batchnorm

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        if batchnorm > 0.0:
            self.bn1 = nn.BatchNorm2d(in_planes, eps=batchnorm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        if batchnorm > 0.0:
            self.bn2 = nn.BatchNorm2d(out_planes, eps=batchnorm)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop_rate = dropout
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                                                stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        """
        overloaded forward method to carry out the forward pass with the network
        
        Parameters:
            x (torch.Tensor): input to the wrn block
        
        Returns:
            output (torch.Tensor) of the wrn block
        """

        if not self.equalInOut:
            if self.batchnorm > 0.0:
                x = self.relu1(self.bn1(x))
            else:
                x = F.relu(x)

        else:
            if self.batchnorm > 0.0:
                out = self.relu1(self.bn1(x))
            else:
                out = F.relu(x)
        if self.batchnorm > 0.0:
            out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        else:
            out = self.relu2(self.conv1(out if self.equalInOut else x))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class Net(nn.Module):
    """
    builds a network from a state_list

    Parameters:
        state_list (list): list of states in the state sequence
        num_classes (int): number of data classes
        net_input (batch of data): batch of data for setting values of batch size, number of colors and image size
        bn_val (float): epsilon value for batch normalization
        do_drop (float): weight drop probability for dropout

    Attributes:
        gpu_mem_req (float): estimated gpu memory requirement for the model by adding memory for storing model
            weights/biases and activations
        feature_extractor (torch.nn.Sequential): sequential container holding the layers in the feature extractor
                                                of a network
        spp_size (int): no. of scales in the spp layer
        classifier (torch.nn.Sequential): sequential container holding the layers in the classifier of a network
    """

    def __init__(self, state_list, num_classes, net_input, bn_val, do_drop):
        super(Net, self).__init__()
        batch_size = net_input.size(0)
        num_colors = net_input.size(1)
        image_size = net_input.size(2)

        # class attribute for storing total gpu memory requirement of the model
        # (4 bytes/ 32 bits per floating point no.)
        self.gpu_mem_req = 32 * batch_size * num_colors * image_size * image_size

        # lists for appending layer definitions
        feature_extractor_list = []
        classifier_list = []

        wrn_no = conv_no = fc_no = relu_no = bn_no = 0
        out_channel = num_colors
        no_feature = num_colors * (image_size ** 2)
        last_image_size = image_size

        # for pretty-printing
        print('*' * 80)

        for state_no, state in enumerate(state_list):
            # last layer is classifier (linear with sigmoid)
            if state_no == len(state_list)-1:
                break

            if state.layer_type == 'wrn':
                wrn_no += 1
                in_channel = out_channel
                out_channel = state.filter_depth
                no_feature = (state.image_size ** 2) * out_channel
                last_image_size = state.image_size

                feature_extractor_list.append(('wrn_' + str(wrn_no), WRNBasicBlock(in_channel, out_channel,
                                                                                   stride=state.stride,
                                                                                   batchnorm=bn_val)))

                # gpu memory requirement for wrn block due to layer parameters (batchnorm parameters have been
                # ignored)
                self.gpu_mem_req += 32 * (3 * 3 * in_channel * out_channel + 3 * 3 * out_channel * out_channel +
                                          int(in_channel != out_channel) * in_channel * out_channel)

                # gpu memory requirement for wrn block due to layer feature output
                self.gpu_mem_req += 32 * batch_size * state.image_size * state.image_size * state.filter_depth\
                                    * (2 + int(in_channel != out_channel))

            elif state.layer_type == 'conv':
                conv_no += 1
                in_channel = out_channel
                out_channel = state.filter_depth
                no_feature = (state.image_size ** 2) * out_channel
                last_image_size = state.image_size

                # conv filters without padding
                if state_space_parameters.conv_padding == 'VALID':
                    feature_extractor_list.append(('conv' + str(conv_no), nn.Conv2d(in_channel, out_channel,
                                                                                    state.filter_size,
                                                                                    stride=state.stride,
                                                                                    bias=False)))
                else:
                    raise NotImplementedError("conv layers for retaining feature dimensionality yet to be"
                                              "implemented")
                bn_no += 1
                feature_extractor_list.append(('batchnorm' + str(bn_no), nn.BatchNorm2d(num_features=out_channel,
                                                                                        eps=bn_val)))
                relu_no += 1
                feature_extractor_list.append(('relu' + str(relu_no), nn.ReLU(inplace=True)))

                # gpu memory requirement for conv layer due to layer parameters (batchnorm parameters have been
                # ignored)
                self.gpu_mem_req += 32 * in_channel * out_channel * state.filter_size * state.filter_size

                # gpu memory requirement for conv layer due to layer feature output
                self.gpu_mem_req += 32 * batch_size * state.image_size * state.image_size * state.filter_depth

            elif state.layer_type == 'spp':
                temp = torch.randn(batch_size, out_channel, last_image_size, last_image_size)
                no_feature = spatial_pyramid_pooling(temp, state.filter_size).size(1)

                # gpu memory requirement for spp layer
                self.gpu_mem_req += 32 * no_feature * batch_size

                self.spp_size = state.filter_size

            elif state.layer_type == 'fc':
                fc_no += 1
                in_feature = no_feature
                no_feature = state.fc_size

                classifier_list.append(('fc' + str(fc_no), nn.Linear(in_feature, no_feature, bias=False)))
                classifier_list.append(('batchnorm_fc' + str(fc_no), nn.BatchNorm1d(num_features=no_feature,
                                                                                    eps=bn_val)))
                classifier_list.append(('relu_fc' + str(fc_no), nn.ReLU(inplace=True)))

                # gpu memory requirement for FC layer due to layer parameters (batchnorm parameters have been ignored)
                self.gpu_mem_req += 32 * batch_size * no_feature

                # gpu memory requirement for FC layer due to layer feature output
                self.gpu_mem_req += 32 * in_feature * no_feature

        fc_no += 1

        # dropout just before the final linear layer
        classifier_list.append(('dropout', nn.Dropout(p=do_drop)))
        classifier_list.append(('fc' + str(fc_no), nn.Linear(no_feature, num_classes, bias=False)))

        # gpu memory requirement for classifier layer due to layer parameters
        self.gpu_mem_req += 32 * no_feature * num_classes

        # gpu memory requirement for classifier layer due to layer output
        self.gpu_mem_req += 32 * batch_size * num_classes

        # converting bits to GB
        self.gpu_mem_req /= (8.*1024*1024*1024)

        self.feature_extractor = nn.Sequential(collections.OrderedDict(feature_extractor_list))
        self.classifier = nn.Sequential(collections.OrderedDict(classifier_list))

    def forward(self, x):
        """
        overloaded forward method to carry out the forward pass with the network
        
        Parameters:
            x (torch.Tensor): input to the network
        
        Returns:
            output (torch.Tensor) of the network
        """
        x = spatial_pyramid_pooling(self.feature_extractor(x), self.spp_size)
        x = torch.sigmoid(self.classifier(x))
        return x
