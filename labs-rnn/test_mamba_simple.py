import numpy as np
from models.mamba import Mamba

# 简单测试Mamba模型的前向传播功能
def test_mamba_simple():
    print("Testing Mamba model with simple inputs...")
    
    # 模型参数
    input_size = 1
    hidden_size = 64
    output_size = 1
    state_size = 32
    kernel_size = 4
    seq_len = 10
    batch_size = 2
    
    # 初始化模型
    model = Mamba(input_size, hidden_size, output_size, state_size, kernel_size)
    
    # 创建随机输入数据
    x = np.random.randn(seq_len, input_size, batch_size)
    s_prev = np.zeros((state_size, batch_size))
    
    print(f"Input shape: {x.shape}")
    print(f"Initial state shape: {s_prev.shape}")
    
    # 前向传播
    y_pred, s = model.forward(x, s_prev)
    
    print(f"Output shape: {y_pred.shape}")
    print(f"Final state shape: {s.shape}")
    
    # 打印一些输出值
    print("\nSample outputs:")
    for t in range(3):
        print(f"Time step {t}: {y_pred[t, 0, :]}")
    
    # 测试反向传播
    print("\nTesting backward pass...")
    dy = np.random.randn(seq_len, output_size, batch_size)
    ds_next = np.zeros((state_size, batch_size))
    
    dx, ds_prev_grad = model.backward(dy, ds_next)
    
    print(f"Input gradient shape: {dx.shape}")
    print(f"State gradient shape: {ds_prev_grad.shape}")
    
    # 测试参数更新
    print("\nTesting parameter update...")
    model.update()
    print("Parameter update completed.")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_mamba_simple()