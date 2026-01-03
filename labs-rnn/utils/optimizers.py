"""
优化器模块
实现多种优化算法：SGD、Adam、RMSProp
"""
import numpy as np


class Optimizer:
    """优化器基类"""
    
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def _get_params_and_grads(self, model):
        """获取模型参数和梯度
        
        Args:
            model: 模型实例
            
        Returns:
            params_and_grads: [(name, param, grad), ...]
        """
        params_and_grads = []
        
        # Mamba模型参数
        mamba_params = [
            ('W_in', 'dW_in'), ('b_in', 'db_in'),
            ('W_gate', 'dW_gate'), ('b_gate', 'db_gate'),
            ('B', 'dB'), ('C', 'dC'),
            ('W_out', 'dW_out'), ('b_out', 'db_out')
        ]
        
        # MinGRU模型参数
        mingru_params = [
            ('W_z', 'dW_z'), ('U_z', 'dU_z'), ('b_z', 'db_z'),
            ('W_h', 'dW_h'), ('U_h', 'dU_h'), ('b_h', 'db_h'),
            ('W_out', 'dW_out'), ('b_out', 'db_out')
        ]
        
        # 尝试获取参数
        for param_name, grad_name in mamba_params + mingru_params:
            if hasattr(model, param_name):
                param = getattr(model, param_name)
                # 梯度可能在cache中或直接作为属性
                if hasattr(model, 'cache') and grad_name in model.cache:
                    grad = model.cache[grad_name]
                elif hasattr(model, grad_name):
                    grad = getattr(model, grad_name)
                else:
                    continue
                params_and_grads.append((param_name, param, grad))
        
        return params_and_grads
    
    def step(self, model, weight_decay=0.0):
        """执行一步优化"""
        raise NotImplementedError
    
    def reset(self):
        """重置优化器状态"""
        pass


class SGD(Optimizer):
    """随机梯度下降优化器
    
    支持动量(momentum)和Nesterov加速
    
    Args:
        lr: 学习率
        momentum: 动量系数，默认0
        nesterov: 是否使用Nesterov动量，默认False
    """
    
    def __init__(self, lr=0.01, momentum=0.0, nesterov=False):
        super().__init__(lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = {}
    
    def step(self, model, weight_decay=0.0):
        """执行一步SGD优化
        
        Args:
            model: 模型实例
            weight_decay: L2正则化系数
        """
        params_and_grads = self._get_params_and_grads(model)
        
        for name, param, grad in params_and_grads:
            # 添加L2正则化梯度（偏置项不加正则化）
            if 'b_' not in name and 'b' != name:
                grad = grad + weight_decay * param
            
            # 初始化速度
            if name not in self.velocity:
                self.velocity[name] = np.zeros_like(param)
            
            # 更新速度
            v = self.velocity[name]
            v = self.momentum * v - self.lr * grad
            self.velocity[name] = v
            
            # 更新参数
            if self.nesterov:
                param += self.momentum * v - self.lr * grad
            else:
                param += v
            
            # 写回参数
            setattr(model, name, param)
    
    def reset(self):
        """重置速度"""
        self.velocity = {}


class Adam(Optimizer):
    """Adam优化器
    
    自适应学习率优化算法，结合了动量和RMSProp的优点
    
    Args:
        lr: 学习率，默认0.001
        beta1: 一阶矩估计的指数衰减率，默认0.9
        beta2: 二阶矩估计的指数衰减率，默认0.999
        epsilon: 数值稳定性常数，默认1e-8
    """
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = 0   # 时间步
    
    def step(self, model, weight_decay=0.0):
        """执行一步Adam优化
        
        Args:
            model: 模型实例
            weight_decay: L2正则化系数（AdamW风格）
        """
        self.t += 1
        params_and_grads = self._get_params_and_grads(model)
        
        for name, param, grad in params_and_grads:
            # 添加L2正则化梯度（偏置项不加正则化）
            if 'b_' not in name and 'b' != name:
                grad = grad + weight_decay * param
            
            # 初始化一阶和二阶矩估计
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
            
            # 更新一阶矩估计（动量）
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            
            # 更新二阶矩估计（自适应学习率）
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            # 偏差校正
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # 更新参数
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            # 写回参数
            setattr(model, name, param)
    
    def reset(self):
        """重置优化器状态"""
        self.m = {}
        self.v = {}
        self.t = 0


class RMSProp(Optimizer):
    """RMSProp优化器
    
    使用梯度平方的移动平均来调整学习率
    
    Args:
        lr: 学习率，默认0.001
        alpha: 平滑常数（衰减率），默认0.99
        epsilon: 数值稳定性常数，默认1e-8
        momentum: 动量系数，默认0
    """
    
    def __init__(self, lr=0.001, alpha=0.99, epsilon=1e-8, momentum=0.0):
        super().__init__(lr)
        self.alpha = alpha
        self.epsilon = epsilon
        self.momentum = momentum
        self.v = {}  # 梯度平方的移动平均
        self.buffer = {}  # 动量缓冲
    
    def step(self, model, weight_decay=0.0):
        """执行一步RMSProp优化
        
        Args:
            model: 模型实例
            weight_decay: L2正则化系数
        """
        params_and_grads = self._get_params_and_grads(model)
        
        for name, param, grad in params_and_grads:
            # 添加L2正则化梯度（偏置项不加正则化）
            if 'b_' not in name and 'b' != name:
                grad = grad + weight_decay * param
            
            # 初始化移动平均
            if name not in self.v:
                self.v[name] = np.zeros_like(param)
                if self.momentum > 0:
                    self.buffer[name] = np.zeros_like(param)
            
            # 更新梯度平方的移动平均
            self.v[name] = self.alpha * self.v[name] + (1 - self.alpha) * (grad ** 2)
            
            # 计算更新量
            if self.momentum > 0:
                self.buffer[name] = self.momentum * self.buffer[name] + \
                                    grad / (np.sqrt(self.v[name]) + self.epsilon)
                param -= self.lr * self.buffer[name]
            else:
                param -= self.lr * grad / (np.sqrt(self.v[name]) + self.epsilon)
            
            # 写回参数
            setattr(model, name, param)
    
    def reset(self):
        """重置优化器状态"""
        self.v = {}
        self.buffer = {}


class AdaGrad(Optimizer):
    """AdaGrad优化器
    
    自适应学习率，对稀疏梯度效果好
    
    Args:
        lr: 学习率，默认0.01
        epsilon: 数值稳定性常数，默认1e-8
    """
    
    def __init__(self, lr=0.01, epsilon=1e-8):
        super().__init__(lr)
        self.epsilon = epsilon
        self.G = {}  # 梯度平方累积
    
    def step(self, model, weight_decay=0.0):
        """执行一步AdaGrad优化
        
        Args:
            model: 模型实例
            weight_decay: L2正则化系数
        """
        params_and_grads = self._get_params_and_grads(model)
        
        for name, param, grad in params_and_grads:
            # 添加L2正则化梯度（偏置项不加正则化）
            if 'b_' not in name and 'b' != name:
                grad = grad + weight_decay * param
            
            # 初始化累积
            if name not in self.G:
                self.G[name] = np.zeros_like(param)
            
            # 累积梯度平方
            self.G[name] += grad ** 2
            
            # 更新参数
            param -= self.lr * grad / (np.sqrt(self.G[name]) + self.epsilon)
            
            # 写回参数
            setattr(model, name, param)
    
    def reset(self):
        """重置优化器状态"""
        self.G = {}


def get_optimizer(name, **kwargs):
    """获取优化器实例
    
    Args:
        name: 优化器名称 ('sgd', 'adam', 'rmsprop', 'adagrad')
        **kwargs: 优化器参数
        
    Returns:
        optimizer: 优化器实例
    """
    optimizers = {
        'sgd': SGD,
        'adam': Adam,
        'rmsprop': RMSProp,
        'adagrad': AdaGrad
    }
    
    name = name.lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")
    
    return optimizers[name](**kwargs)

