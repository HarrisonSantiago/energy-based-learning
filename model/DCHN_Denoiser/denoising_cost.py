from abc import ABC, abstractmethod
import torch
from model.function.interaction import Function, QFunction


class CostFunction(ABC):
   """Abstract class for cost functions

    Attributes
    ----------
    img_size (Tensor): Shape of images
    _predicted (Tensor): Predicted image, Shape is (batch size, 3, H, W)
    _target (Tensor): Target Image. Shape is (batch size, 3, H, W). Type is float.

    Methods
    -------
    set_target(label)
        Set target values
    error_fn()
        Computes the error value for the current state configuration
    _get_output()
        Returns the output layer, or the prediction
    """
   def __init__(self, shape):
        """Initializes an instance of CostFunction

        Args:
            shape (Tensor): CxHxW shape of the image
        """

        self._shape = shape

        self._predicted = None  # FIXME
        self._target = None  # FIXME

   def set_target(self, target):
        """Set label and target values

        Args:
            label: Image tensor
        """
        output = self._get_output()  # the output layer's state
        device = output.device  # device on which the output layer Tensor is, and on which we put the layer and target Tensors
        self._target = target.to(device).type(torch.float16)

   def error_fn(self):
      """Returns the error value for the current output configuration.

        Returns:
            Tensor of shape (batch_size,) and type bool. Vector of error values for each of the examples in the current mini-batch
        """

      output = self._get_output()
      mse = torch.mean((output - self._target) ** 2, dim=[1, 2, 3])
      return mse

   def top_five_error_fn(self):
    """
    Returns fake top-5 error values for compatibility reasons.

    Returns:
        Tensor of shape (batch_size,) and type bool. Vector of fake top-5 error values for each of the examples in the current mini-batch
    """
    # Assuming we can still get the batch size from the output
    output = self._get_output()
    batch_size = output.size(0)

    # Generate random boolean values
    fake_errors = torch.rand(batch_size) > 0.8  # 20% chance of being True

    return fake_errors

   @abstractmethod
   def _get_output(self):
        """Returns the output layer, or the prediction"""
        pass



class SquaredError(CostFunction, QFunction):
    """Class for the squared error cost function between the output layer and the target layer.

    Methods
    -------
    eval()
        Returns the squared error between the output layer and the target
    """

    def __init__(self, layer):
        """Initializes an instance of SquaredError

        Args:
            layer (Layer): the layer playing the role of `output layer', or prediction
        """

        self._layer = layer

        image_shape = layer.state.shape  # number of categories in the classification task
        CostFunction.__init__(self, image_shape)

        Function.__init__(self, [layer], [])

    def eval(self):
        """Returns the cost value (the squared error) of the current state configuration.

        Returns:
            Tensor of shape (batch_size,) and type float16. Vector of cost values for each of the examples in the current mini-batch
        """

        output = self._layer.state  # state of output layer: shape is (batch_size, 3, H, W)
        target = self._target  # desired output: shape is (batch_size, 3, H, W)
        return torch.mean((target - output) ** 2) # Vector of shape (batch_size,)

    def _get_output(self):
        """Returns the output layer, or the prediction"""
        return self._layer.state

    def grad_layer_fn(self, layer):
        """Overrides the default implementation of Function"""
        dictionary = {self._layer: self._grad_layer}
        return dictionary[layer]

    def a_coef_fn(self, layer):
        """Overrides the default implementation of QFunction"""
        dictionary = {
            self._layer: self._a_coef,
            }
        return dictionary[layer]

    def b_coef_fn(self, layer):
        """Overrides the default implementation of QFunction"""
        dictionary = {
            self._layer: self._b_coef,
            }
        return dictionary[layer]

    def _grad_layer(self):
        """Returns the gradient of the cost function wrt the outputs"""
        return self._layer.state - self._target

    def _a_coef(self):
        """Returns the interaction's linear influence on the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float16: the linear contribution on layer_pre
        """

        return 0.5

    def _b_coef(self):
        """Returns the interaction's linear influence on the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float16: the linear contribution on layer_pre
        """

        return - self._target

    def __str__(self):
        return 'Squared Error (MSE)'