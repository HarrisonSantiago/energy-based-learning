import numpy

from model.function.interaction import SumSeparableFunction
from model.variable.layer import InputLayer, LinearLayer
from model.DCHN_Denoiser.layer import HardSigmoidLayer, SigmoidLayer, SoftMaxLayer, dSiLULayer
from model.variable.parameter import Bias, DenseWeight, ConvWeight
from model.DCHN_Denoiser.interaction import BiasInteraction, DenseHopfield, ConvAvgPoolHopfield, ConvMaxPoolHopfield, ConvSoftPoolHopfield, ModernHopfield, ConvHopfield



def create_layer(shape, activation='hard-sigmoid'):
    """Adds a layer to the network

    Args:
        shape (tuple of ints): shape of the layer
        activation (str, optional): the layer's activation function, either the identity ('linear'), the 'hard-sigmoid', or the `silu'. Default: 'hard-sigmoid'.
    """

    if activation == 'linear': layer = LinearLayer(shape)
    elif activation == 'hard-sigmoid': layer = HardSigmoidLayer(shape)
    elif activation == 'sigmoid': layer = SigmoidLayer(shape)
    elif activation == 'softmax': layer = SoftMaxLayer(shape)
    elif activation == 'silu': layer = dSiLULayer(shape)
    elif activation == 'input': layer = InputLayer(shape)
    else: raise ValueError("expected `linear', `hard-sigmoid' or `silu' but got {}".format(activation))

    return layer

def create_edge(layers, interaction_type, indices, gain, shape=None, padding=0):
    """Adds an interaction between two layers of the network.

    Adding an interaction also adds the associated parameter (weight or bias)

    Args:
        interaction_type (str): either `bias', `dense', `conv_avg_pool' or `conv_max_pool'
        indices (list of int): indices of layer_pre (the `pre-synaptic' layer) and layer_post (the `post-synaptic' layer)
        gain (float16): the gain (scaling factor) of the param at initialization
        shape (tuple of ints, optional): the shape of the param tensor. Required in the case of convolutional params. Default: None
        padding (int, optional): the padding of the convolution, if applicable. Default: 0
    """

    if interaction_type == "bias":
        layer = layers[indices[0]]
        if shape == None: shape = layer.shape  # if no shape is provided for the bias, we use the layer's shape by default
        param = Bias(shape, gain=gain, device=None)
        interaction = BiasInteraction(layer, param)
    elif interaction_type == "dense":
        layer_pre = layers[indices[0]]
        layer_post = layers[indices[1]]
        param = DenseWeight(layer_pre.shape, layer_post.shape, gain, device=None)
        interaction = DenseHopfield(layer_pre, layer_post, param)
    elif interaction_type == "conv_avg_pool":
        layer_pre = layers[indices[0]]
        layer_post = layers[indices[1]]
        param = ConvWeight(shape, gain, device=None)
        interaction = ConvAvgPoolHopfield(layer_pre, layer_post, param, padding)
    elif interaction_type == "conv_max_pool":
        layer_pre = layers[indices[0]]
        layer_post = layers[indices[1]]
        param = ConvWeight(shape, gain, device=None)
        interaction = ConvMaxPoolHopfield(layer_pre, layer_post, param, padding)
    elif interaction_type == "conv_soft_pool":
        layer_pre = layers[indices[0]]
        layer_post = layers[indices[1]]
        param = ConvWeight(shape, gain, device=None)
        interaction = ConvSoftPoolHopfield(layer_pre, layer_post, param, padding)
    elif interaction_type == "modern_hopfield":
        layer = layers[indices[0]]
        shape = (1024,)
        param = DenseWeight(layer.shape, shape, gain, device=None)
        interaction = ModernHopfield(layer, param)
    elif interaction_type == 'conv_hopfield':
        layer_pre = layers[indices[0]]
        layer_post = layers[indices[1]]
        param = ConvWeight(shape, gain, device = None)
        interaction = ConvHopfield(layer_pre, layer_post, param, padding)
    else:
        raise ValueError("expected `bias', `dense', `conv_avg_pool', `conv_max_pool' or `conv_soft_pool' but got {}".format(interaction_type))

    return param, interaction




class ConvHopfieldEnergyRGB(SumSeparableFunction):
    """Energy function of a convolutional Hopfield network (CHN) with RGB pixel input images

    The model consists of 3 layers:
        0. input layer has shape (3, H, W)
        1. first hidden layer has shape (num_hidden_1, H/2, W/2)
        2. second hidden layer has shape (num_hidden_2, H/4, W/4)
        3. third hidden layer has shape (num_hidden_3, H/8, W/8)
        4. output layer has shape (3, H, W)

    The first three weight tensors are 3x3 convolutional kernels (with padding 1), followed by 2x2 pooling.
    The last weight tensor is dense.
    """

    def __init__(self, image_shape,  num_inputs=3, num_hiddens_1=32, num_hiddens_2=64, num_hiddens_3=128, num_outputs = 3, activation='hard-sigmoid', pool_type='conv_hopfield', weight_gains=[0.6, 0.6, 0.5, 0.5]):
        """Creates an instance of a convolutional Hopfield network 28x28 (CHN28)

        Args:
            num_inputs (int, optional): number of input filters. Default: 1
            num_hiddens_1 (int, optional): number of filters in the first hidden layer. Default: 32
            num_hiddens_2 (int, optional): number of filters in the second hidden layer. Default: 64
            num_hiddens_3 (int, optional): number of filters in the third layer
            activation (str, optional): activation function used for the hidden layers. Default: 'hard-sigmoid'
        """

        self._size = [num_inputs, num_hiddens_1, num_hiddens_2, num_hiddens_3, num_outputs]
        self._activation = activation
        self._pool_type = pool_type
        self._weight_gains = weight_gains

        # creates the layers of the network
        #TODO make a test for this
        C, H, W = image_shape
        layer_shapes = [(num_inputs, H, W), (num_hiddens_1, H, W), (num_hiddens_2, H, W), (num_hiddens_3, H, W), (num_outputs, H, W)]
        activations = ['input', activation, activation, activation, 'linear']

        # define the biases of the network
        bias_shapes = [(num_hiddens_1,), (num_hiddens_2,), (num_hiddens_3,), (num_outputs,)]
        bias_gains = [0.5/numpy.sqrt(num_inputs*3*3), 0.5/numpy.sqrt(num_hiddens_1*3*3), 0.5/numpy.sqrt(num_hiddens_2*3*3), 0.5/numpy.sqrt(num_hiddens_3*2*2)]


        # define the weights of the network
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        weight_shapes = [(num_hiddens_1, num_inputs, 3, 3), (num_hiddens_2, num_hiddens_1, 3, 3), (num_hiddens_3, num_hiddens_2, 3, 3), (num_outputs, 2, 2, num_hiddens_3)]
        weight_types = [pool_type, pool_type, pool_type, 'dense']
        paddings = [1, 1, 1, None]

        # create the layers, biases, and weights
        layers = [create_layer(shape, activation) for shape, activation in zip(layer_shapes, activations)]
        biases = [Bias(shape, gain=gain, device=None) for shape, gain in zip(bias_shapes, bias_gains)]
        bias_interactions = [BiasInteraction(layer, bias) for layer, bias in zip(layers[1:], biases)]

        params = biases
        interactions = bias_interactions

        for indices, weight_type, gain, shape, padding, in zip(edges, weight_types, weight_gains, weight_shapes, paddings):
            param, interaction = create_edge(layers, weight_type, indices, gain, shape, padding)
            params.append(param)
            interactions.append(interaction)

        # creates an instance of a SumSeparableFunction
        SumSeparableFunction.__init__(self, layers, params, interactions)

    def __str__(self):
        return 'ConvHopfieldEnergyRGB -- size={}, activation={}, pooling={}, gains={}'.format(self._size, self._activation, self._pool_type, self._weight_gains)
