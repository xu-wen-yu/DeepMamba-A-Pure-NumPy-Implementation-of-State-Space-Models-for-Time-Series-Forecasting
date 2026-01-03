import numpy as np

class Mamba:
    def __init__(self, input_size, hidden_size, output_size, state_size=64, kernel_size=4, learning_rate=0.01):
        """初始化Mamba模型
        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            state_size: 状态空间大小，默认为64
            kernel_size: 卷积核大小，默认为4
            learning_rate: 学习率
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.state_size = state_size
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        
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
        self.A = np.ones((state_size, 1)) * -0.5  # 固定为负实数，确保状态衰减
        self.B = xavier_init((hidden_size, state_size))
        self.C = xavier_init((hidden_size, state_size))
        
        # 输出投影权重
        self.W_out = xavier_init((output_size, hidden_size))
        self.b_out = np.zeros((output_size, 1))
        
        # 保存中间结果用于反向传播
        self.cache = {}
    
    def silu(self, x):
        """SiLU激活函数"""
        return x * (1 / (1 + np.exp(-x)))
    
    def forward(self, x, s_prev):
        """前向传播
        Args:
            x: 输入数据，形状(seq_len, input_size, batch_size)
            s_prev: 初始状态，形状(state_size, batch_size)
        Returns:
            y_pred: 预测输出，形状(seq_len, output_size, batch_size)
            s: 状态序列，形状(seq_len, state_size, batch_size)
        """
        seq_len, _, batch_size = x.shape
        y_pred = []
        s = []
        
        # 保存中间结果
        self.cache['x'] = x
        self.cache['s_prev'] = s_prev
        self.cache['gate'] = []  # 选通门
        self.cache['s'] = []  # 状态
        self.cache['h'] = []  # 隐藏层输出
        
        for t in range(seq_len):
            # 获取当前时间步输入
            x_t = x[t, :, :]  # 形状(input_size, batch_size)
            
            # 输入投影
            z = np.dot(self.W_in, x_t) + self.b_in  # 形状(hidden_size, batch_size)
            
            # 选通门
            gate = self.silu(np.dot(self.W_gate, x_t) + self.b_gate)  # 形状(hidden_size, batch_size)
            
            # 状态空间更新：s = s·exp(A) + Bᵀ·gate
            s_t = s_prev * np.exp(self.A) + np.dot(self.B.T, gate)  # 形状(state_size, batch_size)
            
            # 隐藏层输出：h = gate * (C · s)
            h_t = gate * np.dot(self.C, s_t)  # 形状(hidden_size, batch_size)
            
            # 输出投影
            y_t = np.dot(self.W_out, h_t) + self.b_out  # 形状(output_size, batch_size)
            
            # 保存结果
            y_pred.append(y_t)
            s.append(s_t)
            self.cache['gate'].append(gate)
            self.cache['s'].append(s_t)
            self.cache['h'].append(h_t)
            
            # 更新前一状态
            s_prev = s_t
        
        # 转换为数组
        y_pred = np.array(y_pred)  # 形状(seq_len, output_size, batch_size)
        s = np.array(s)  # 形状(seq_len, state_size, batch_size)
        
        return y_pred, s
    
    def backward(self, dy, ds_next):
        """反向传播
        Args:
            dy: 输出梯度，形状(seq_len, output_size, batch_size)
            ds_next: 下一时间步状态梯度，形状(state_size, batch_size)
        Returns:
            dx: 输入梯度，形状(seq_len, input_size, batch_size)
            ds_prev: 初始状态梯度，形状(state_size, batch_size)
        """
        seq_len, output_size, batch_size = dy.shape
        
        # 初始化梯度
        dW_in = np.zeros_like(self.W_in)
        db_in = np.zeros_like(self.b_in)
        dW_gate = np.zeros_like(self.W_gate)
        db_gate = np.zeros_like(self.b_gate)
        dC = np.zeros_like(self.C)
        dB = np.zeros_like(self.B)
        dW_out = np.zeros_like(self.W_out)
        db_out = np.zeros_like(self.b_out)
        
        dx = np.zeros((seq_len, self.input_size, batch_size))
        ds_prev = ds_next
        
        # 从后向前遍历
        for t in reversed(range(seq_len)):
            # 获取当前时间步中间结果
            x_t = self.cache['x'][t, :, :]
            gate_t = self.cache['gate'][t]
            s_t = self.cache['s'][t]
            h_t = self.cache['h'][t]
            s_prev_t = self.cache['s_prev'] if t == 0 else self.cache['s'][t-1]
            
            # 输出层梯度
            dW_out += np.dot(dy[t], h_t.T)
            db_out += np.sum(dy[t], axis=1, keepdims=True)
            
            # 隐藏层梯度
            dh_t = np.dot(self.W_out.T, dy[t])  # 形状(hidden_size, batch_size)
            
            # 选通门梯度（来自h_t）
            dgate_t1 = dh_t * np.dot(self.C, s_t)  # 形状(hidden_size, batch_size)
            
            # C梯度
            dC += np.dot(dh_t * gate_t, s_t.T)  # 形状(hidden_size, state_size)
            
            # 状态梯度（来自h_t）
            ds_t = np.dot(self.C.T, dh_t * gate_t) + ds_prev  # 形状(state_size, batch_size)
            
            # 状态更新的梯度
            ds_prev = ds_t * np.exp(self.A)  # 形状(state_size, batch_size)
            
            # B梯度
            dB += np.dot(gate_t, ds_t.T)  # 形状(hidden_size, state_size)
            
            # 选通门梯度（来自s_t）
            dgate_t2 = np.dot(self.B, ds_t)  # 形状(hidden_size, batch_size)
            
            # 选通门总梯度
            dgate_t = dgate_t1 + dgate_t2  # 形状(hidden_size, batch_size)
            
            # SiLU导数
            silu_deriv = self.silu(x_t) + (1 - self.silu(x_t)) * x_t  # 这里需要修正，应该是对gate的输入求导
            # 重新计算选通门的输入
            gate_input_t = np.dot(self.W_gate, x_t) + self.b_gate
            silu_deriv = self.silu(gate_input_t) * (1 + gate_input_t * (1 - self.silu(gate_input_t)))
            
            # 选通门权重梯度
            dW_gate += np.dot(dgate_t * silu_deriv, x_t.T)
            db_gate += np.sum(dgate_t * silu_deriv, axis=1, keepdims=True)
            
            # 输入梯度（来自选通门）
            dx[t, :, :] += np.dot(self.W_gate.T, dgate_t * silu_deriv)
            
            # 输入投影权重梯度（如果有需要）
            # 注意：这里的z在当前实现中没有被使用，所以暂时注释掉
            # dz_t = ...  # 来自后续计算
            # dW_in += np.dot(dz_t, x_t.T)
            # db_in += np.sum(dz_t, axis=1, keepdims=True)
            # dx[t, :, :] += np.dot(self.W_in.T, dz_t)
        
        # 保存权重梯度
        self.cache['dW_in'] = dW_in
        self.cache['db_in'] = db_in
        self.cache['dW_gate'] = dW_gate
        self.cache['db_gate'] = db_gate
        self.cache['dC'] = dC
        self.cache['dB'] = dB
        self.cache['dW_out'] = dW_out
        self.cache['db_out'] = db_out
        
        return dx, ds_prev
    
    def update(self, lr=None, weight_decay=0.0):
        """参数更新
        
        Args:
            lr: 学习率，如果为None则使用初始化时的学习率
            weight_decay: L2正则化系数，默认0（不使用正则化）
        """
        if lr is None:
            lr = self.learning_rate
        
        # 使用梯度下降更新权重和偏置，并添加L2正则化
        # 权重矩阵添加L2正则化
        self.W_in -= lr * (self.cache['dW_in'] + weight_decay * self.W_in)
        self.b_in -= lr * self.cache['db_in']  # 偏置不加正则化
        
        self.W_gate -= lr * (self.cache['dW_gate'] + weight_decay * self.W_gate)
        self.b_gate -= lr * self.cache['db_gate']  # 偏置不加正则化
        
        self.C -= lr * (self.cache['dC'] + weight_decay * self.C)
        self.B -= lr * (self.cache['dB'] + weight_decay * self.B)
        
        self.W_out -= lr * (self.cache['dW_out'] + weight_decay * self.W_out)
        self.b_out -= lr * self.cache['db_out']  # 偏置不加正则化
    
    def get_l2_loss(self, weight_decay=0.0001):
        """计算L2正则化损失
        
        Args:
            weight_decay: L2正则化系数
            
        Returns:
            l2_loss: L2正则化损失值
        """
        l2_loss = 0.0
        l2_loss += 0.5 * weight_decay * np.sum(self.W_in ** 2)
        l2_loss += 0.5 * weight_decay * np.sum(self.W_gate ** 2)
        l2_loss += 0.5 * weight_decay * np.sum(self.B ** 2)
        l2_loss += 0.5 * weight_decay * np.sum(self.C ** 2)
        l2_loss += 0.5 * weight_decay * np.sum(self.W_out ** 2)
        return l2_loss