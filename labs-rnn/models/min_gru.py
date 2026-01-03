import numpy as np

class MinGRU:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """初始化MinGRU模型
        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            learning_rate: 学习率
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 初始化权重和偏置（随机初始化×0.01）
        # 更新门权重
        self.W_z = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_z = np.zeros((hidden_size, 1))
        
        # 重置门权重
        self.W_r = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_r = np.zeros((hidden_size, 1))
        
        # 候选隐藏状态权重
        self.W_h = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        
        # 输出层权重
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))
        
        # 保存中间结果用于反向传播
        self.cache = {}
    
    def sigmoid(self, x):
        """sigmoid激活函数"""
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        """tanh激活函数"""
        return np.tanh(x)
    
    def forward(self, x, h_prev):
        """前向传播
        Args:
            x: 输入数据，形状(seq_len, input_size, batch_size)
            h_prev: 初始隐藏状态，形状(hidden_size, batch_size)
        Returns:
            y_pred: 预测输出，形状(seq_len, output_size, batch_size)
            h: 隐藏状态序列，形状(seq_len, hidden_size, batch_size)
        """
        seq_len, _, batch_size = x.shape
        y_pred = []
        h = []
        
        # 保存中间结果
        self.cache['x'] = x
        self.cache['h_prev'] = h_prev
        self.cache['z'] = []  # 更新门
        self.cache['r'] = []  # 重置门
        self.cache['h_tilde'] = []  # 候选隐藏状态
        self.cache['h'] = []  # 隐藏状态
        
        for t in range(seq_len):
            # 获取当前时间步输入
            x_t = x[t, :, :]  # 形状(input_size, batch_size)
            
            # 拼接输入和前一隐藏状态
            combined = np.concatenate([h_prev, x_t], axis=0)  # 形状(hidden_size + input_size, batch_size)
            
            # 更新门计算
            z_t = self.sigmoid(np.dot(self.W_z, combined) + self.b_z)  # 形状(hidden_size, batch_size)
            
            # 重置门计算
            r_t = self.sigmoid(np.dot(self.W_r, combined) + self.b_r)  # 形状(hidden_size, batch_size)
            
            # 候选隐藏状态计算
            combined_r = np.concatenate([r_t * h_prev, x_t], axis=0)  # 形状(hidden_size + input_size, batch_size)
            h_tilde_t = self.tanh(np.dot(self.W_h, combined_r) + self.b_h)  # 形状(hidden_size, batch_size)
            
            # 新隐藏状态计算
            h_t = (1 - z_t) * h_prev + z_t * h_tilde_t  # 形状(hidden_size, batch_size)
            
            # 输出计算
            y_t = np.dot(self.W_y, h_t) + self.b_y  # 形状(output_size, batch_size)
            
            # 保存结果
            y_pred.append(y_t)
            h.append(h_t)
            self.cache['z'].append(z_t)
            self.cache['r'].append(r_t)
            self.cache['h_tilde'].append(h_tilde_t)
            self.cache['h'].append(h_t)
            
            # 更新前一隐藏状态
            h_prev = h_t
        
        # 转换为数组
        y_pred = np.array(y_pred)  # 形状(seq_len, output_size, batch_size)
        h = np.array(h)  # 形状(seq_len, hidden_size, batch_size)
        
        return y_pred, h
    
    def backward(self, dy, dh_next):
        """反向传播
        Args:
            dy: 输出梯度，形状(seq_len, output_size, batch_size)
            dh_next: 下一时间步隐藏状态梯度，形状(hidden_size, batch_size)
        Returns:
            dx: 输入梯度，形状(seq_len, input_size, batch_size)
            dh_prev: 初始隐藏状态梯度，形状(hidden_size, batch_size)
        """
        seq_len, output_size, batch_size = dy.shape
        
        # 初始化梯度
        dW_z = np.zeros_like(self.W_z)
        db_z = np.zeros_like(self.b_z)
        dW_r = np.zeros_like(self.W_r)
        db_r = np.zeros_like(self.b_r)
        dW_h = np.zeros_like(self.W_h)
        db_h = np.zeros_like(self.b_h)
        dW_y = np.zeros_like(self.W_y)
        db_y = np.zeros_like(self.b_y)
        
        dx = np.zeros((seq_len, self.input_size, batch_size))
        dh_prev = dh_next
        
        # 从后向前遍历
        for t in reversed(range(seq_len)):
            # 获取当前时间步中间结果
            x_t = self.cache['x'][t, :, :]
            h_t = self.cache['h'][t]
            h_prev_t = self.cache['h_prev'] if t == 0 else self.cache['h'][t-1]
            z_t = self.cache['z'][t]
            r_t = self.cache['r'][t]
            h_tilde_t = self.cache['h_tilde'][t]
            
            # 输出层梯度
            dW_y += np.dot(dy[t], h_t.T)
            db_y += np.sum(dy[t], axis=1, keepdims=True)
            
            # 隐藏状态梯度
            dh_t = np.dot(self.W_y.T, dy[t]) + dh_prev
            
            # 候选隐藏状态梯度
            dh_tilde_t = dh_t * z_t * (1 - h_tilde_t ** 2)
            
            # 更新门梯度
            dz_t = dh_t * (h_tilde_t - h_prev_t) * z_t * (1 - z_t)
            
            # 重置门梯度
            dr_t = np.dot(self.W_h[:, :self.hidden_size].T, dh_tilde_t) * h_prev_t * r_t * (1 - r_t)
            
            # 拼接输入和前一隐藏状态
            combined_r = np.concatenate([r_t * h_prev_t, x_t], axis=0)
            combined = np.concatenate([h_prev_t, x_t], axis=0)
            
            # 权重梯度计算
            dW_h += np.dot(dh_tilde_t, combined_r.T)
            db_h += np.sum(dh_tilde_t, axis=1, keepdims=True)
            
            dW_z += np.dot(dz_t, combined.T)
            db_z += np.sum(dz_t, axis=1, keepdims=True)
            
            dW_r += np.dot(dr_t, combined.T)
            db_r += np.sum(dr_t, axis=1, keepdims=True)
            
            # 输入梯度和前一隐藏状态梯度
            dcombined_r = np.dot(self.W_h.T, dh_tilde_t)
            dcombined = np.dot(self.W_z.T, dz_t) + np.dot(self.W_r.T, dr_t)
            
            dx[t, :, :] = dcombined_r[self.hidden_size:, :] + dcombined[self.hidden_size:, :]
            dh_prev = dcombined_r[:self.hidden_size, :] * r_t + dcombined[:self.hidden_size, :]
        
        # 保存权重梯度
        self.cache['dW_z'] = dW_z
        self.cache['db_z'] = db_z
        self.cache['dW_r'] = dW_r
        self.cache['db_r'] = db_r
        self.cache['dW_h'] = dW_h
        self.cache['db_h'] = db_h
        self.cache['dW_y'] = dW_y
        self.cache['db_y'] = db_y
        
        return dx, dh_prev
    
    def update(self, lr=None, weight_decay=0.0):
        """参数更新
        
        Args:
            lr: 学习率，如果为None则使用初始化时的学习率
            weight_decay: L2正则化系数，默认0（不使用正则化）
        """
        if lr is None:
            lr = self.learning_rate
        
        # 使用梯度下降更新权重和偏置，并添加L2正则化
        # 权重矩阵添加L2正则化，偏置不加正则化
        self.W_z -= lr * (self.cache['dW_z'] + weight_decay * self.W_z)
        self.b_z -= lr * self.cache['db_z']
        
        self.W_r -= lr * (self.cache['dW_r'] + weight_decay * self.W_r)
        self.b_r -= lr * self.cache['db_r']
        
        self.W_h -= lr * (self.cache['dW_h'] + weight_decay * self.W_h)
        self.b_h -= lr * self.cache['db_h']
        
        self.W_y -= lr * (self.cache['dW_y'] + weight_decay * self.W_y)
        self.b_y -= lr * self.cache['db_y']
    
    def get_l2_loss(self, weight_decay=0.0001):
        """计算L2正则化损失
        
        Args:
            weight_decay: L2正则化系数
            
        Returns:
            l2_loss: L2正则化损失值
        """
        l2_loss = 0.0
        l2_loss += 0.5 * weight_decay * np.sum(self.W_z ** 2)
        l2_loss += 0.5 * weight_decay * np.sum(self.W_r ** 2)
        l2_loss += 0.5 * weight_decay * np.sum(self.W_h ** 2)
        l2_loss += 0.5 * weight_decay * np.sum(self.W_y ** 2)
        return l2_loss