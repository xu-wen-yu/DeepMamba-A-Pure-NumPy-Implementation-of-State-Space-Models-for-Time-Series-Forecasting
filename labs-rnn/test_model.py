import numpy as np
import sys
import os
from utils.data_loader import DataLoader
from models.min_gru import MinGRU
from models.mamba import Mamba

# 确保可以导入utils和models模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def train_model(model_type='min_gru', seq_len=30, batch_size=32, epochs=5, learning_rate=0.01, hidden_size=128):
    """训练指定模型
    Args:
        model_type: 模型类型，可选'min_gru'或'mamba'
        seq_len: 序列长度
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        hidden_size: 隐藏层大小
    """
    print(f"Training {model_type} model...")
    print(f"Hyperparameters: seq_len={seq_len}, batch_size={batch_size}, epochs={epochs}, lr={learning_rate}, hidden_size={hidden_size}")
    
    # 1. 数据加载与预处理
    data_loader = DataLoader(data_dir='data')
    
    # 加载股票数据
    data = data_loader.load_yahoo_stock(ticker='AAPL', start_date='2010-01-01', end_date='2023-12-31')
    
    # 预处理数据
    feature_data, normalized_data, scaler = data_loader.preprocess_stock_data(data, feature='Close')
    
    # 创建批次
    batches = data_loader.create_stock_batches(normalized_data, seq_len, batch_size)
    print(f"Created {len(batches)} batches")
    
    # 2. 模型初始化
    input_size = 1
    output_size = 1
    
    if model_type == 'min_gru':
        model = MinGRU(input_size, hidden_size, output_size, learning_rate)
    elif model_type == 'mamba':
        model = Mamba(input_size, hidden_size, output_size, state_size=64, kernel_size=4, learning_rate=learning_rate)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    
    # 3. 训练循环
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (x_batch, y_batch) in enumerate(batches):
            # 初始化隐藏状态或状态
            if model_type == 'min_gru':
                h_prev = np.zeros((hidden_size, batch_size))
                y_pred, h = model.forward(x_batch, h_prev)
                # 仅使用最后一个时间步的预测结果
                y_pred_last = y_pred[-1, :, :]
                
                # 计算损失（MSE）
                loss = np.mean((y_pred_last - y_batch) ** 2)
                epoch_loss += loss
                
                # 反向传播
                dy = np.zeros_like(y_pred)
                dy[-1, :, :] = 2 * (y_pred_last - y_batch) / batch_size
                dh_next = np.zeros((hidden_size, batch_size))
                dx, dh_prev = model.backward(dy, dh_next)
            
            elif model_type == 'mamba':
                s_prev = np.zeros((model.state_size, batch_size))
                y_pred, s = model.forward(x_batch, s_prev)
                # 仅使用最后一个时间步的预测结果
                y_pred_last = y_pred[-1, :, :]
                
                # 计算损失（MSE）
                loss = np.mean((y_pred_last - y_batch) ** 2)
                epoch_loss += loss
                
                # 反向传播
                dy = np.zeros_like(y_pred)
                dy[-1, :, :] = 2 * (y_pred_last - y_batch) / batch_size
                ds_next = np.zeros((model.state_size, batch_size))
                dx, ds_prev = model.backward(dy, ds_next)
            
            # 参数更新
            model.update()
        
        # 打印每轮损失
        avg_epoch_loss = epoch_loss / len(batches)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
    
    print(f"Training {model_type} completed!")
    return model, scaler, feature_data, normalized_data

import argparse

if __name__ == "__main__":
    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description='Train and test sequence models for stock prediction')
    parser.add_argument('--model_type', type=str, default='min_gru', choices=['min_gru', 'mamba'], help='Model type to train')
    parser.add_argument('--seq_len', type=int, default=30, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size')
    
    args = parser.parse_args()
    
    # 训练模型
    train_model(
        model_type=args.model_type,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size
    )