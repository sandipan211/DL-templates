import torch
import torch.nn as nn

# While defining the model class, it should extend the Module class of Pytorch. However, here we are going to use functions defined in the nn module. So using that while defining the class:
class myVGG(nn.Module):


  def __init__(self, conv_maps, attributes):


    """ Since in the VGGNet models different modes have been tried that vary the depth of the network but use the same filter sizes and paddings, the conv_chunks [CONV2D - RELU - POOL] have been defined in the get_conv_chunks() method and returned as conv_maps to be part of the model's __init__(). And finally the FCLs have been defined in __init__() to complete the layer architecture. Also, needs attributes of the dataset. 
    
    VGG11 with LRN has not been implemented as it did not prove to be much useful according to authors of VGGNet
    """

    # The constructor of the class (__init__) defines the layers of the model

    # As you construct a Net class by inheriting from the Module class and you override the default behavior of the __init__ constructor, you also need to explicitly call the parent's one with super(Net, self).__init__(). See for details: "https://datascience.stackexchange.com/questions/58415/how-does-the-forward-method-get-called-in-this-pytorch-conv-net"
    super(myVGG, self).__init__()

    self.conv_maps = conv_maps
    # conv_maps has a shape [7,7,512]

    # since in Conv2d implementation of Pytorch it is independent of the input size (it takes only number of channels as input), hence to be able to work fine with images of different sizes, we need to add Adaptive average pooling. The layer structure obtained till now assumes that the input is always 224x224 (as required in VGG, and we explicitly resized images in VGG.ipynb). If we work with any other image size, this structure of convNet might not work and give us an error, since we are always expecting 7x7 output maps from the conv structure. Adding an adaptive average pooling works as per the input and output dimensions.
    # for more details on this, see "https://forums.fast.ai/t/adaptivepooling-vs-maxpooling/14727" and "https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work"
    self.aa_pooling = nn.AdaptiveAvgPool2d((7,7)) 

    # got the convolutional units [CONV-RELU-POOL] till now from conv_maps....now add the fully-connected layers (FCLs)
    self.classifier = nn.Sequential(
        nn.Linear(7*7*512, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),   
        nn.Linear(4096, attributes['num_classes'])    
        )

    self.initialization()



  def forward(self, x):

    # The forward() function is the override that defines how to forward propagate input through the defined layers of the model
    x = self.conv_maps(x)
    x = self.aa_pooling(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)

    return x

  def initialization(self):

    # in the actual paper, the authors trained the network in mode A with random initialization, and then used it training weights as the initial weights for the other configurations....but later they found that xavier initialization works very well....so using that here

    # iterate over all the modules in the network defined using modules() method of nn.Module. See this for details: "https://discuss.pytorch.org/t/pytorch-self-module/49677"
    for m in self.modules():
      
      # two kinds of layers to be initialized: conv layers and FCLs
      # check if a layer is an instance of a particular class using isinstance(object, class) and initialize accordingly
      if isinstance(m, nn.Conv2d):
        
        # Pytorch modules store their state in a dictionary accessed using state_dict(). If we write module.state_dict().keys(), we get the stored state names: ['bias', 'weight']. Hence, using these keys, we can access the weights and biases of any layer. For more info, look for state_dict() in "https://pytorch.org/docs/stable/nn.html#parameters"
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
      

      elif isinstance(m, nn.Linear):

        # authors do random normal initialization with mean = 0 and variance = 0.01
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.constant_(m.bias, 0)




# the following are the five modes of VGG models that have been described in the original paper
cfg_params = {
    
    'A': [64, 'MP', 128, 'MP', 256, 256, 'MP', 512, 512, 'MP', 512, 512, 'MP'],
    # input to output: using the formula (W-F+2P)/2 + 1 while reducing map size using CONV or POOL
    # For CONV layers: F = 3x3, S = 1, P = 1
    # For Maxpool layers: F = 2x2, stride = 2, no padding 

    # [224*224*3] ==> [224*224*64]
    # [224*224*64] ===> [112*112*64]
    # [112*112*64] ===> [112*112*128]
    # [112*112*128] ===> [56*56*128]
    # [56*56*128] ===> [56*56*256]
    # [56*56*256] ===> [56*56*256]
    # [56*56*256] ===> [28*28*256]
    # [28*28*256] ===> [28*28*512]
    # [28*28*512] ===> [28*28*512]
    # [28*28*512] ===> [14*14*512]
    # [14*14*512] ===> [14*14*512]
    # [14*14*512] ===> [14*14*512]
    # [14*14*512] ===> [7*7*512]

    #similar calculations follow for the other modes as well

    'B': [64, 64, 'MP', 128, 128, 'MP', 256, 256, 'MP', 512, 512, 'MP', 512, 512, 'MP'],
    'C': [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 'MP', 512, 512, 512, 'MP', 512, 512, 512, 'MP'],
    'D': [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 'MP', 512, 512, 512, 'MP', 512, 512, 512, 'MP'],
    'E': [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 256, 'MP', 512, 512, 512, 512, 'MP', 512, 512, 512, 512, 'MP']

    # at the end of every mode, we would get feature maps of size [7*7*512]
}



def get_conv_chunks(mode, cfg, attributes):

  in_channels = attributes['num_channels']
  layer_stack = []

  if mode == 'C':

    # this is a special case where some conv layers (L8, L12, L16 - indexed from 0) have filters of size 1*1 (as told in paper)

    layer_counter = 0
    # stack up the layers as per the configuration in corresponding cfg
    for i in cfg:

      if i == 'MP':
        # add a maxpooling layer
        layer_stack += [nn.MaxPool2d(kernel_size=2, stride=2)]
    
      else:
        # add a conv layer followed by ReLU activation
        if layer_counter in [8, 12, 16]:
          conv_layer = nn.Conv2d(in_channels, i, kernel_size = 1, stride = 1, padding = 0)
        else:
          conv_layer = nn.Conv2d(in_channels, i, kernel_size = 3, stride = 1, padding = 1)
        # inplace = True apparently saves memory See for details: "https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948/9"
        activation_layer = nn.ReLU(inplace = True)
        layer_stack += [conv_layer, activation_layer]
 
        # set the new in_channels to be the previous out_channels (as desired) - not needed for pooling layers though as spatial size remains same after pooling
        in_channels = i

      layer_counter += 1

  
  else:
    # stack up the layers as per the configuration in corresponding cfg
    for i in cfg:

      if i == 'MP':
        # add a maxpooling layer
        layer_stack += [nn.MaxPool2d(kernel_size=2, stride=2)]
    
      else:
        # add a conv layer followed by ReLU activation
        conv_layer = nn.Conv2d(in_channels, i, kernel_size = 3, stride = 1, padding = 1)
        # inplace = True apparently saves memory See for details: "https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948/9"
        activation_layer = nn.ReLU(inplace = True)
        layer_stack += [conv_layer, activation_layer]
 
        # set the new in_channels to be the previous out_channels (as desired)
        in_channels = i

  # unpacked layer_stack (using *) returned to be a part of the sequntial layer structure. For reference, see "https://stackoverflow.com/questions/46141690/how-to-write-a-pytorch-sequential-model" and "https://www.datacamp.com/community/tutorials/usage-asterisks-python"
  return nn.Sequential(*layer_stack)


def makeVGG(mode, cfg_params, attributes):

  vgg_model = myVGG(get_conv_chunks(mode, cfg_params[mode], attributes), attributes)
  return vgg_model


def myVGG11(attributes):
  
  return makeVGG('A', cfg_params, attributes)


def myVGG13(attributes):
  
  return makeVGG('B', cfg_params, attributes)


def myVGG16_less_conv(attributes):
  
  return makeVGG('C', cfg_params, attributes)

  
def myVGG16(attributes):
  
  return makeVGG('D', cfg_params, attributes)

  
def myVGG19(attributes):
  
  return makeVGG('E', cfg_params, attributes)

  
