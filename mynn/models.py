from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_rates=None):
        self.size_list = size_list
        self.act_func = act_func
        self.lambda_list = lambda_list
        self.dropout_rates = dropout_rates

        if size_list is not None and act_func is not None:
            self.layers = []
            num_hidden_layers = len(size_list)-2
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                self.layers.append(layer)

                # 非输出层添加激活函数和Dropout
                if i < len(size_list) - 2:
                    if act_func == 'Logistic':
                        raise NotImplementedError
                    elif act_func == 'ReLU':
                        self.layers.append(ReLU())
                    if dropout_rates is not None and i < len(dropout_rates):
                        self.layers.append(Dropout(p=dropout_rates[i]))

    def train_mode(self):
        for layer in self.layers:
            layer.train()

    def eval_mode(self):
        for layer in self.layers:
            layer.eval()

            
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def compute_l2_loss(self):
        """
        Compute the total L2 regularization loss for all layers
        """
        l2_loss = 0.0
        for layer in self.layers:
            if isinstance(layer, Linear) and layer.weight_decay: 
                lambda_val = layer.weight_decay_lambda
                l2_loss += 0.5 * lambda_val * np.sum(layer.params['W'] ** 2)
        return l2_loss

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        
        # 重建模型结构（包括 Dropout）
        self.layers = []
        num_hidden_layers = len(self.size_list) - 2
        for i in range(len(self.size_list) - 1):
            # 加载线性层
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i + 2]['lambda']
            self.layers.append(layer)
            
            
            if i < len(self.size_list) - 2:
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    self.layers.append(ReLU())
            
            
            if self.dropout_rates is not None and i < len(self.dropout_rates):
                self.layers.append(Dropout(p=self.dropout_rates[i]))
            
    
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):  # 只保存可优化层（Linear）
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
            # 如果需要保存 Dropout 的 p 值，可以额外添加逻辑
            # elif isinstance(layer, Dropout):
            #     param_list.append({'p': layer.p})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, conv_configs=None, fc_configs=None, act_func='ReLU', use_global_avg_pool=True):
        super().__init__()
        self.layers = []
        self.optimizable = True
        self.has_initialized = False
        self.conv_configs = conv_configs
        self.fc_configs = fc_configs
        self.act_func = act_func
        self.use_global_avg_pool = use_global_avg_pool
        
        if conv_configs is not None and fc_configs is not None:
            self.has_initialized = True
            for config in conv_configs:
                layer_type = config.get('type', 'conv') 
                
                if layer_type == 'conv':
                    in_channels = config['in_channels']
                    out_channels = config['out_channels']
                    kernel_size = config['kernel_size']
                    stride = config.get('stride', 1)
                    padding = config.get('padding', 0)
                    weight_decay = config.get('weight_decay', False)
                    weight_decay_lambda = config.get('weight_decay_lambda', 1e-8)
                    
                    conv_layer = conv2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        weight_decay=weight_decay,
                        weight_decay_lambda=weight_decay_lambda
                    )
                    self.layers.append(conv_layer)
                    
                    if act_func == 'ReLU':
                        self.layers.append(ReLU())
                    else:
                        raise NotImplementedError(f"Activation function {act_func} not implemented")
                
                elif layer_type == 'pool':
                    pool_type = config.get('pool_type', 'max')
                    kernel_size = config['kernel_size']
                    stride = config.get('stride', kernel_size) 
                    padding = config.get('padding', 0)
                    
                    pool_layer = PoolLayer(
                        pool_type=pool_type,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding
                    )
                    self.layers.append(pool_layer)
            
            if use_global_avg_pool:
                self.layers.append(GlobalAvgPool())
            else:
                # Add a Flatten layer if not using global average pooling
                self.layers.append(Flatten())
            for i, (in_dim, out_dim) in enumerate(fc_configs):
                fc_layer = Linear(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    weight_decay=config.get('weight_decay', False),
                    weight_decay_lambda=config.get('weight_decay_lambda', 1e-8)
                )
                self.layers.append(fc_layer)
                if i < len(fc_configs) - 1 and act_func == 'ReLU':
                    self.layers.append(ReLU())

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if not self.has_initialized:
            raise ValueError("Model has not been initialized. Use model.load_model to load a model "
                           "or create a new model with conv_configs and fc_configs provided.")
        
        output = X
        for layer in self.layers:
            output = layer(output)
        
        return output
    

    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        return grad
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        
        self.conv_configs = param_list[0]
        self.fc_configs = param_list[1]
        self.act_func = param_list[2]
        self.use_global_avg_pool = param_list[3]
        
        self.__init__(self.conv_configs, self.fc_configs, self.act_func, self.use_global_avg_pool)
        param_idx = 4 
        
        for i, layer in enumerate(self.layers):
            if layer.optimizable:
                layer.params['W'] = param_list[param_idx]['W']
                layer.params['b'] = param_list[param_idx]['b']
                layer.weight_decay = param_list[param_idx]['weight_decay']
                layer.weight_decay_lambda = param_list[param_idx]['lambda']
                param_idx += 1
        
        self.has_initialized = True
        
    def save_model(self, save_path):
        param_list = [self.conv_configs, self.fc_configs, self.act_func, self.use_global_avg_pool]
        
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Flatten(Layer):
    """
    Layer that flattens the input tensor from shape [batch_size, channels, height, width]
    to shape [batch_size, channels*height*width]
    """
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.input_shape = None
    
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        # Preserve batch size but flatten all other dimensions
        return X.reshape(X.shape[0], -1)
    
    def backward(self, grad):
        # Reshape gradient back to the input shape
        return grad.reshape(self.input_shape)