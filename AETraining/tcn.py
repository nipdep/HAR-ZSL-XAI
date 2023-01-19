import torch
from torch import nn
from torch.nn.utils import weight_norm

def is_power_of_two(num: int):
    return num != 0 and ((num & (num - 1)) == 0)


def adjust_dilations(dilations: list):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class ResidualBlock(nn.Module):
    def __init__(self,
                 input_size: int,
                 nb_filters: int,
                 dilation_rate: int,
                 kernel_size: int,
                 padding: str,
                 padding_type:str = "zeros",
                 activation: str = 'relu',
                 dropout_rate: float = 0,
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_weight_norm: bool = False,
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        
        """Defines the residual block for the WaveNet TCN
        Args:
            x: The previous layer in the model
            training: boolean indicating whether the layer should behave in training mode or in inference mode
            dilation_rate: The dilation power of 2 we are using for this residual block
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            padding: The padding used in the convolutional layers, 'same' or 'causal'.
            activation: The final activation used in o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            use_weight_norm: Whether to use weight normalization in the residual layers or not.
            kwargs: Any initializers for Layer class.
        """
        
        self.input_size = input_size
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding_type = padding_type
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None
        
        assert not (self.use_batch_norm and self.use_layer_norm)
        self.build_layers()
        
    def build_layers(self):
        self.layers = []
        self.layer_name = []
        inp_size = self.input_size
        
        for i in range(2):
            conv1d_1 = nn.Conv1d(
                inp_size,self.nb_filters,kernel_size=self.kernel_size,
                dilation=self.dilation_rate,padding=self.padding,padding_mode = self.padding_type)
            
            if self.use_weight_norm:
                conv1d_1 = weight_norm(self.conv1d_1)

            if self.use_batch_norm:
                norm_layer = (nn.BatchNorm1d(self.nb_filters))
            elif self.use_layer_norm:
                norm_layer = (nn.LayerNorm(self.nb_filters))
            else:
                pass

            activation_1 = nn.ReLU()
            dropout_1 = nn.Dropout()
            
            if self.use_batch_norm or self.use_layer_norm:
                self.layers += [conv1d_1,norm_layer,activation_1,dropout_1]
            else:
                self.layers += [conv1d_1,activation_1,dropout_1]
                
            inp_size = self.nb_filters
            
        self.conv1x1 = nn.Conv1d(
                self.input_size,self.nb_filters,kernel_size=1,
                dilation=self.dilation_rate,padding=self.padding,padding_mode = "replicate")
        
        self.conv1x1_act = nn.ReLU()
        self.final_act = nn.ReLU()
             
        
    def forward(self,x):
        x1 = x
        for __id,layer in enumerate(self.layers):
            x1 = layer(x1)
        x2 = self.conv1x1(x)
        x2 = self.conv1x1_act(x2)
        
        x1_x2 = self.final_act(x1+x2)
        return [x1_x2, x1]
    
    
    


    
    
    
class TCN(nn.Module):
    """Creates a TCN layer.
        Input shape:
            A 3D tensor with shape (batch_size, timesteps, input_dim).
        Args:
            nb_filters: The number of filters to use in the convolutional layers. Can be a list.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual blocK.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            activation: The activation used in the residual blocks o = Activation(x + F(x)).
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            use_weight_norm: Whether to use weight normalization in the residual layers or not.
            go_backwards: Boolean (default False). If True, process the input sequence backwards and
            return the reversed sequence.
            return_state: Boolean. Whether to return the last state in addition to the output. Default: False.
            kwargs: Any other arguments for configuring parent class Layer. For example "name=str", Name of the model.
                    Use unique names when using multiple TCN.
        Returns:
            A TCN layer.
        """

    def __init__(self,
                 n_features,
                 nb_filters=64,
                 kernel_size=3,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='reflect',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='relu',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False,
                 go_backwards=False,
                 return_state=False,
                 **kwargs):
        super(TCN, self).__init__(**kwargs)
        
        self.n_features = n_features
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation_name = activation
        self.padding = padding
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.go_backwards = go_backwards
        self.return_state = return_state
        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None
        self.slicer_layer = None  # in case return_sequence=False
        self.output_slice_index = None  # in case return_sequence=False
        self.padding_same_and_time_dim_unknown = False  # edge case if padding='same' and time_dim = None

        if self.use_batch_norm + self.use_layer_norm + self.use_weight_norm > 1:
            raise ValueError('Only one normalization can be specified at once.')

        if isinstance(self.nb_filters, list):
            assert len(self.nb_filters) == len(self.dilations)
            if len(set(self.nb_filters)) > 1 and self.use_skip_connections:
                raise ValueError('Skip connections are not compatible '
                                 'with a list of filters, unless they are all equal.')
            
        self.build_model()

    
    def build_model(self):
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1
            
        if isinstance(self.nb_filters, list):
            input_sizes_list = self.nb_filters[:]
            input_sizes_list.insert(0,self.n_features)
            input_sizes_list = input_sizes_list[:-1]
            
        else:
            input_sizes_list = self.nb_filters
        
        for i, d in enumerate(self.dilations):
            res_block_filters = self.nb_filters[i] if isinstance(self.nb_filters, list) else self.nb_filters
            input_value = input_sizes_list[i] if isinstance(input_sizes_list, list) else input_sizes_list
            n_features = input_value if i != 0 else self.n_features
            self.residual_blocks.append(ResidualBlock(input_size = n_features,
                                                      nb_filters=res_block_filters,
                                                      kernel_size=self.kernel_size,
                                                      padding=self.padding,
                                                      dilation_rate=d,
                                                      activation=self.activation_name,
                                                      dropout_rate=self.dropout_rate,
                                                      use_batch_norm=self.use_batch_norm,
                                                      use_layer_norm=self.use_layer_norm,
                                                      use_weight_norm=self.use_weight_norm))
            

    
    def forward(self,x):
        if self.go_backwards:
            # reverse x in the time axis
            x = tf.reverse(x, axis=[1])

        self.layers_outputs = [x]
        self.skip_connections = []
        for res_block in self.residual_blocks:
            x, skip_out = res_block(x)
            
            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)
        if self.use_skip_connections:
            if len(self.skip_connections) > 1:
                # Keras: A merge layer should be called on a list of at least 2 inputs. Got 1 input.
                x = torch.stack(self.skip_connections)
                x = torch.sum(x, dim=0)
            else:
                x = self.skip_connections[0]
            self.layers_outputs.append(x)

        if not self.return_sequences:
            # case: time dimension is unknown. e.g. (bs, None, input_dim).
            if self.padding_same_and_time_dim_unknown:
                self.output_slice_index = K.shape(self.layers_outputs[-1])[1] // 2
            x = self.slicer_layer(x)
            self.layers_outputs.append(x)
        return x
        
    