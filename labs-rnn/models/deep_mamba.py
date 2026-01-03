"""
深层Mamba模型
支持多层堆叠、残差连接、层归一化等高级特性
"""
import numpy as np


class MambaLayer:
    """单层Mamba模块"""
    
    def __init__(self, input_size, hidden_size, state_size=64):
        """初始化单层Mamba
        
        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            state_size: 状态空间大小
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        
        # Xavier初始化函数
        def xavier_init(shape):
            in_dim, out_dim = shape
            limit = np.sqrt(6 / (in_dim + out_dim))
            return np.random.uniform(-limit, limit, shape)
        
        # 输入投影权重
        self.W_in = xavier_init((hidden_size, input_size))
        self.b_in = np.zeros((hidden_size, 1))
        
        # 选通门权重
        self.W_gate = xavier_init((hidden_size, input_size))
        self.b_gate = np.zeros((hidden_size, 1))
        
        # 状态空间参数
        self.A = np.ones((state_size, 1)) * -0.5
        self.B = xavier_init((hidden_size, state_size))
        self.C = xavier_init((hidden_size, state_size))
        
        # 输出投影（将hidden_size映射回input_size，用于残差连接）
        self.W_proj = xavier_init((input_size, hidden_size))
        self.b_proj = np.zeros((input_size, 1))
        
        # 保存中间结果
        self.cache = {}
    
    def silu(self, x):
        """SiLU激活函数"""
        return x * (1 / (1 + np.exp(-np.clip(x, -500, 500))))
    
    def forward(self, x, s_prev):
        """前向传播
        
        Args:
            x: 输入数据，形状(seq_len, input_size, batch_size)
            s_prev: 初始状态，形状(state_size, batch_size)
            
        Returns:
            output: 输出，形状(seq_len, input_size, batch_size)
            s_final: 最终状态，形状(state_size, batch_size)
        """
        seq_len, _, batch_size = x.shape
        outputs = []
        
        # 保存中间结果
        self.cache['x'] = x
        self.cache['s_prev'] = s_prev
        self.cache['gate'] = []
        self.cache['s'] = []
        self.cache['h'] = []
        
        for t in range(seq_len):
            x_t = x[t, :, :]  # (input_size, batch_size)
            
            # 输入投影
            z = np.dot(self.W_in, x_t) + self.b_in  # (hidden_size, batch_size)
            
            # 选通门
            gate = self.silu(np.dot(self.W_gate, x_t) + self.b_gate)
            
            # 状态空间更新
            s_t = s_prev * np.exp(self.A) + np.dot(self.B.T, gate)
            
            # 隐藏层输出
            h_t = gate * np.dot(self.C, s_t)
            
            # 输出投影
            out_t = np.dot(self.W_proj, h_t) + self.b_proj  # (input_size, batch_size)
            
            outputs.append(out_t)
            self.cache['gate'].append(gate)
            self.cache['s'].append(s_t)
            self.cache['h'].append(h_t)
            
            s_prev = s_t
        
        output = np.array(outputs)  # (seq_len, input_size, batch_size)
        return output, s_prev
    
    def backward(self, dout, ds_next):
        """反向传播
        
        Args:
            dout: 输出梯度，形状(seq_len, input_size, batch_size)
            ds_next: 下一时间步状态梯度
            
        Returns:
            dx: 输入梯度
            ds_prev: 初始状态梯度
        """
        seq_len, _, batch_size = dout.shape
        
        # 初始化梯度
        self.cache['dW_in'] = np.zeros_like(self.W_in)
        self.cache['db_in'] = np.zeros_like(self.b_in)
        self.cache['dW_gate'] = np.zeros_like(self.W_gate)
        self.cache['db_gate'] = np.zeros_like(self.b_gate)
        self.cache['dB'] = np.zeros_like(self.B)
        self.cache['dC'] = np.zeros_like(self.C)
        self.cache['dW_proj'] = np.zeros_like(self.W_proj)
        self.cache['db_proj'] = np.zeros_like(self.b_proj)
        
        dx = np.zeros((seq_len, self.input_size, batch_size))
        ds_prev = ds_next
        
        for t in reversed(range(seq_len)):
            x_t = self.cache['x'][t, :, :]
            gate_t = self.cache['gate'][t]
            s_t = self.cache['s'][t]
            h_t = self.cache['h'][t]
            s_prev_t = self.cache['s_prev'] if t == 0 else self.cache['s'][t-1]
            
            # 输出投影梯度
            self.cache['dW_proj'] += np.dot(dout[t], h_t.T)
            self.cache['db_proj'] += np.sum(dout[t], axis=1, keepdims=True)
            
            # 隐藏层梯度
            dh_t = np.dot(self.W_proj.T, dout[t])
            
            # 选通门梯度
            dgate_t1 = dh_t * np.dot(self.C, s_t)
            
            # C梯度
            self.cache['dC'] += np.dot(dh_t * gate_t, s_t.T)
            
            # 状态梯度
            ds_t = np.dot(self.C.T, dh_t * gate_t) + ds_prev
            ds_prev = ds_t * np.exp(self.A)
            
            # B梯度
            self.cache['dB'] += np.dot(gate_t, ds_t.T)
            
            # 选通门梯度
            dgate_t2 = np.dot(self.B, ds_t)
            dgate_t = dgate_t1 + dgate_t2
            
            # SiLU导数
            gate_input_t = np.dot(self.W_gate, x_t) + self.b_gate
            sigmoid_val = 1 / (1 + np.exp(-np.clip(gate_input_t, -500, 500)))
            silu_deriv = sigmoid_val * (1 + gate_input_t * (1 - sigmoid_val))
            
            # 选通门权重梯度
            self.cache['dW_gate'] += np.dot(dgate_t * silu_deriv, x_t.T)
            self.cache['db_gate'] += np.sum(dgate_t * silu_deriv, axis=1, keepdims=True)
            
            # 输入梯度
            dx[t, :, :] = np.dot(self.W_gate.T, dgate_t * silu_deriv)
        
        return dx, ds_prev
    
    def update(self, lr, weight_decay=0.0):
        """参数更新"""
        self.W_in -= lr * (self.cache['dW_in'] + weight_decay * self.W_in)
        self.b_in -= lr * self.cache['db_in']
        self.W_gate -= lr * (self.cache['dW_gate'] + weight_decay * self.W_gate)
        self.b_gate -= lr * self.cache['db_gate']
        self.B -= lr * (self.cache['dB'] + weight_decay * self.B)
        self.C -= lr * (self.cache['dC'] + weight_decay * self.C)
        self.W_proj -= lr * (self.cache['dW_proj'] + weight_decay * self.W_proj)
        self.b_proj -= lr * self.cache['db_proj']


class LayerNorm:
    """层归一化"""
    
    def __init__(self, normalized_shape, eps=1e-5):
        """初始化层归一化
        
        Args:
            normalized_shape: 归一化的维度大小
            eps: 数值稳定性常数
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = np.ones((normalized_shape, 1))
        self.beta = np.zeros((normalized_shape, 1))
        self.cache = {}
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入，形状(..., normalized_shape, batch_size)
            
        Returns:
            out: 归一化后的输出
        """
        # 计算均值和方差
        mean = np.mean(x, axis=-2, keepdims=True)
        var = np.var(x, axis=-2, keepdims=True)
        
        # 归一化
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # 缩放和平移
        out = self.gamma * x_norm + self.beta
        
        # 保存中间结果
        self.cache['x'] = x
        self.cache['x_norm'] = x_norm
        self.cache['mean'] = mean
        self.cache['var'] = var
        
        return out
    
    def backward(self, dout):
        """反向传播"""
        x = self.cache['x']
        x_norm = self.cache['x_norm']
        mean = self.cache['mean']
        var = self.cache['var']
        
        N = x.shape[-2]
        
        # 计算梯度
        self.cache['dgamma'] = np.sum(dout * x_norm, axis=-1, keepdims=True)
        self.cache['dbeta'] = np.sum(dout, axis=-1, keepdims=True)
        
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + self.eps) ** (-1.5), axis=-2, keepdims=True)
        dmean = np.sum(dx_norm * -1 / np.sqrt(var + self.eps), axis=-2, keepdims=True) + \
                dvar * np.mean(-2 * (x - mean), axis=-2, keepdims=True)
        
        dx = dx_norm / np.sqrt(var + self.eps) + dvar * 2 * (x - mean) / N + dmean / N
        
        return dx
    
    def update(self, lr, weight_decay=0.0):
        """参数更新"""
        self.gamma -= lr * self.cache['dgamma']
        self.beta -= lr * self.cache['dbeta']


class DeepMamba:
    """深层Mamba模型
    
    支持多层堆叠、残差连接、层归一化
    """
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, 
                 state_size=64, dropout_rate=0.0, use_residual=True, 
                 use_layer_norm=True, learning_rate=0.01):
        """初始化深层Mamba模型
        
        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            num_layers: 层数
            state_size: 状态空间大小
            dropout_rate: Dropout比率（训练时使用）
            use_residual: 是否使用残差连接
            use_layer_norm: 是否使用层归一化
            learning_rate: 学习率
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.state_size = state_size
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.learning_rate = learning_rate
        
        # 输入投影层（将input_size映射到hidden_size）
        self.input_proj = self._xavier_init((hidden_size, input_size))
        self.input_proj_bias = np.zeros((hidden_size, 1))
        
        # 创建多层Mamba
        self.layers = []
        self.layer_norms = []
        
        for i in range(num_layers):
            layer = MambaLayer(hidden_size, hidden_size, state_size)
            self.layers.append(layer)
            
            if use_layer_norm:
                ln = LayerNorm(hidden_size)
                self.layer_norms.append(ln)
        
        # 输出层
        self.W_out = self._xavier_init((output_size, hidden_size))
        self.b_out = np.zeros((output_size, 1))
        
        # 保存中间结果
        self.cache = {}
    
    def _xavier_init(self, shape):
        """Xavier初始化"""
        in_dim, out_dim = shape
        limit = np.sqrt(6 / (in_dim + out_dim))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, x, states=None, training=True):
        """前向传播
        
        Args:
            x: 输入数据，形状(seq_len, input_size, batch_size)
            states: 各层初始状态列表，每个形状(state_size, batch_size)
            training: 是否为训练模式
            
        Returns:
            y_pred: 预测输出，形状(seq_len, output_size, batch_size)
            final_states: 各层最终状态列表
        """
        seq_len, _, batch_size = x.shape
        
        # 初始化状态
        if states is None:
            states = [np.zeros((self.state_size, batch_size)) for _ in range(self.num_layers)]
        
        # 输入投影
        x_proj = np.zeros((seq_len, self.hidden_size, batch_size))
        for t in range(seq_len):
            x_proj[t] = np.dot(self.input_proj, x[t]) + self.input_proj_bias
        
        self.cache['x'] = x
        self.cache['x_proj'] = x_proj
        self.cache['layer_inputs'] = []
        self.cache['layer_outputs'] = []
        self.cache['dropout_masks'] = []
        
        # 通过各层
        h = x_proj
        final_states = []
        
        for i, layer in enumerate(self.layers):
            self.cache['layer_inputs'].append(h.copy())
            
            # Mamba层前向传播
            layer_out, s_final = layer.forward(h, states[i])
            final_states.append(s_final)
            
            # 残差连接
            if self.use_residual:
                layer_out = layer_out + h
            
            # 层归一化
            if self.use_layer_norm:
                # 对每个时间步进行归一化
                normalized_out = np.zeros_like(layer_out)
                for t in range(seq_len):
                    normalized_out[t] = self.layer_norms[i].forward(layer_out[t])
                layer_out = normalized_out
            
            # Dropout（仅训练时）
            if training and self.dropout_rate > 0:
                mask = (np.random.rand(*layer_out.shape) > self.dropout_rate).astype(float)
                layer_out = layer_out * mask / (1 - self.dropout_rate)
                self.cache['dropout_masks'].append(mask)
            else:
                self.cache['dropout_masks'].append(None)
            
            self.cache['layer_outputs'].append(layer_out)
            h = layer_out
        
        # 输出层
        y_pred = np.zeros((seq_len, self.output_size, batch_size))
        for t in range(seq_len):
            y_pred[t] = np.dot(self.W_out, h[t]) + self.b_out
        
        self.cache['h_final'] = h
        
        return y_pred, final_states
    
    def backward(self, dy, ds_nexts=None):
        """反向传播
        
        Args:
            dy: 输出梯度，形状(seq_len, output_size, batch_size)
            ds_nexts: 各层下一时间步状态梯度列表
            
        Returns:
            dx: 输入梯度
            ds_prevs: 各层初始状态梯度列表
        """
        seq_len, _, batch_size = dy.shape
        
        if ds_nexts is None:
            ds_nexts = [np.zeros((self.state_size, batch_size)) for _ in range(self.num_layers)]
        
        # 输出层梯度
        h_final = self.cache['h_final']
        self.cache['dW_out'] = np.zeros_like(self.W_out)
        self.cache['db_out'] = np.zeros_like(self.b_out)
        
        dh = np.zeros((seq_len, self.hidden_size, batch_size))
        for t in range(seq_len):
            self.cache['dW_out'] += np.dot(dy[t], h_final[t].T)
            self.cache['db_out'] += np.sum(dy[t], axis=1, keepdims=True)
            dh[t] = np.dot(self.W_out.T, dy[t])
        
        # 反向通过各层
        ds_prevs = []
        
        for i in reversed(range(self.num_layers)):
            # Dropout梯度
            if self.cache['dropout_masks'][i] is not None:
                dh = dh * self.cache['dropout_masks'][i] / (1 - self.dropout_rate)
            
            # 层归一化梯度
            if self.use_layer_norm:
                dh_norm = np.zeros_like(dh)
                for t in range(seq_len):
                    dh_norm[t] = self.layer_norms[i].backward(dh[t])
                dh = dh_norm
            
            # 残差连接梯度
            if self.use_residual:
                dh_residual = dh.copy()
            
            # Mamba层反向传播
            dh_layer, ds_prev = self.layers[i].backward(dh, ds_nexts[i])
            ds_prevs.insert(0, ds_prev)
            
            # 残差连接
            if self.use_residual:
                dh = dh_layer + dh_residual
            else:
                dh = dh_layer
        
        # 输入投影梯度
        self.cache['dinput_proj'] = np.zeros_like(self.input_proj)
        self.cache['dinput_proj_bias'] = np.zeros_like(self.input_proj_bias)
        
        dx = np.zeros((seq_len, self.input_size, batch_size))
        for t in range(seq_len):
            self.cache['dinput_proj'] += np.dot(dh[t], self.cache['x'][t].T)
            self.cache['dinput_proj_bias'] += np.sum(dh[t], axis=1, keepdims=True)
            dx[t] = np.dot(self.input_proj.T, dh[t])
        
        return dx, ds_prevs
    
    def update(self, lr=None, weight_decay=0.0):
        """参数更新
        
        Args:
            lr: 学习率
            weight_decay: L2正则化系数
        """
        if lr is None:
            lr = self.learning_rate
        
        # 更新输入投影
        self.input_proj -= lr * (self.cache['dinput_proj'] + weight_decay * self.input_proj)
        self.input_proj_bias -= lr * self.cache['dinput_proj_bias']
        
        # 更新各层
        for i, layer in enumerate(self.layers):
            layer.update(lr, weight_decay)
            if self.use_layer_norm:
                self.layer_norms[i].update(lr, weight_decay)
        
        # 更新输出层
        self.W_out -= lr * (self.cache['dW_out'] + weight_decay * self.W_out)
        self.b_out -= lr * self.cache['db_out']
    
    def get_l2_loss(self, weight_decay=0.0001):
        """计算L2正则化损失"""
        l2_loss = 0.0
        l2_loss += 0.5 * weight_decay * np.sum(self.input_proj ** 2)
        l2_loss += 0.5 * weight_decay * np.sum(self.W_out ** 2)
        
        for layer in self.layers:
            l2_loss += 0.5 * weight_decay * np.sum(layer.W_in ** 2)
            l2_loss += 0.5 * weight_decay * np.sum(layer.W_gate ** 2)
            l2_loss += 0.5 * weight_decay * np.sum(layer.B ** 2)
            l2_loss += 0.5 * weight_decay * np.sum(layer.C ** 2)
            l2_loss += 0.5 * weight_decay * np.sum(layer.W_proj ** 2)
        
        return l2_loss
    
    def get_num_parameters(self):
        """获取模型参数数量"""
        num_params = 0
        
        # 输入投影
        num_params += self.input_proj.size + self.input_proj_bias.size
        
        # 各层参数
        for layer in self.layers:
            num_params += layer.W_in.size + layer.b_in.size
            num_params += layer.W_gate.size + layer.b_gate.size
            num_params += layer.B.size + layer.C.size
            num_params += layer.W_proj.size + layer.b_proj.size
        
        # 层归一化参数
        if self.use_layer_norm:
            for ln in self.layer_norms:
                num_params += ln.gamma.size + ln.beta.size
        
        # 输出层
        num_params += self.W_out.size + self.b_out.size
        
        return num_params


class BidirectionalMamba:
    """双向Mamba模型"""
    
    def __init__(self, input_size, hidden_size, output_size, state_size=64, 
                 learning_rate=0.01):
        """初始化双向Mamba
        
        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度（每个方向）
            output_size: 输出维度
            state_size: 状态空间大小
            learning_rate: 学习率
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.state_size = state_size
        self.learning_rate = learning_rate
        
        # 前向和后向Mamba层
        self.forward_layer = MambaLayer(input_size, hidden_size, state_size)
        self.backward_layer = MambaLayer(input_size, hidden_size, state_size)
        
        # 输出层（合并双向输出）
        self.W_out = self._xavier_init((output_size, hidden_size * 2))
        self.b_out = np.zeros((output_size, 1))
        
        self.cache = {}
    
    def _xavier_init(self, shape):
        in_dim, out_dim = shape
        limit = np.sqrt(6 / (in_dim + out_dim))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, x, states=None):
        """前向传播
        
        Args:
            x: 输入数据，形状(seq_len, input_size, batch_size)
            states: (forward_state, backward_state)
            
        Returns:
            y_pred: 预测输出
            final_states: (forward_final_state, backward_final_state)
        """
        seq_len, _, batch_size = x.shape
        
        if states is None:
            s_fwd = np.zeros((self.state_size, batch_size))
            s_bwd = np.zeros((self.state_size, batch_size))
        else:
            s_fwd, s_bwd = states
        
        # 前向传播
        fwd_out, s_fwd_final = self.forward_layer.forward(x, s_fwd)
        
        # 后向传播（反转输入序列）
        x_reversed = x[::-1].copy()
        bwd_out, s_bwd_final = self.backward_layer.forward(x_reversed, s_bwd)
        bwd_out = bwd_out[::-1]  # 反转回来
        
        # 合并双向输出
        self.cache['fwd_out'] = fwd_out
        self.cache['bwd_out'] = bwd_out
        
        # 输出层
        y_pred = np.zeros((seq_len, self.output_size, batch_size))
        for t in range(seq_len):
            combined = np.concatenate([fwd_out[t], bwd_out[t]], axis=0)
            y_pred[t] = np.dot(self.W_out, combined) + self.b_out
        
        self.cache['combined'] = np.concatenate([fwd_out, bwd_out], axis=1)
        
        return y_pred, (s_fwd_final, s_bwd_final)
    
    def backward(self, dy, ds_nexts=None):
        """反向传播"""
        seq_len, _, batch_size = dy.shape
        
        if ds_nexts is None:
            ds_fwd = np.zeros((self.state_size, batch_size))
            ds_bwd = np.zeros((self.state_size, batch_size))
        else:
            ds_fwd, ds_bwd = ds_nexts
        
        # 输出层梯度
        self.cache['dW_out'] = np.zeros_like(self.W_out)
        self.cache['db_out'] = np.zeros_like(self.b_out)
        
        d_combined = np.zeros((seq_len, self.hidden_size * 2, batch_size))
        for t in range(seq_len):
            combined = self.cache['combined'][t]
            self.cache['dW_out'] += np.dot(dy[t], combined.T)
            self.cache['db_out'] += np.sum(dy[t], axis=1, keepdims=True)
            d_combined[t] = np.dot(self.W_out.T, dy[t])
        
        # 分离前向和后向梯度
        d_fwd = d_combined[:, :self.hidden_size, :]
        d_bwd = d_combined[:, self.hidden_size:, :]
        
        # 前向层反向传播
        dx_fwd, ds_fwd_prev = self.forward_layer.backward(d_fwd, ds_fwd)
        
        # 后向层反向传播
        d_bwd_reversed = d_bwd[::-1].copy()
        dx_bwd_reversed, ds_bwd_prev = self.backward_layer.backward(d_bwd_reversed, ds_bwd)
        dx_bwd = dx_bwd_reversed[::-1]
        
        # 合并输入梯度
        dx = dx_fwd + dx_bwd
        
        return dx, (ds_fwd_prev, ds_bwd_prev)
    
    def update(self, lr=None, weight_decay=0.0):
        """参数更新"""
        if lr is None:
            lr = self.learning_rate
        
        self.forward_layer.update(lr, weight_decay)
        self.backward_layer.update(lr, weight_decay)
        
        self.W_out -= lr * (self.cache['dW_out'] + weight_decay * self.W_out)
        self.b_out -= lr * self.cache['db_out']

