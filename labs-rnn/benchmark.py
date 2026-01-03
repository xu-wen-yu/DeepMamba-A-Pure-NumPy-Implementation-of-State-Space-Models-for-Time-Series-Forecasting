import numpy as np
import matplotlib.pyplot as plt
import time
from utils.data_loader import DataLoader
from models.min_gru import MinGRU
from models.mamba import Mamba

# 性能指标计算函数
def calculate_metrics(y_true, y_pred):
    """计算性能指标
    Args:
        y_true: 真实值
        y_pred: 预测值
    Returns:
        metrics: 包含MSE、RMSE、MAE、R²的字典
    """
    # 确保输入是一维数组
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

# 模型训练与评估函数
def train_and_evaluate(model_type, seq_len, batch_size, epochs, learning_rate, hidden_size, data_loader, normalized_data):
    """训练并评估模型
    Args:
        model_type: 模型类型
        seq_len: 序列长度
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        hidden_size: 隐藏层大小
        data_loader: 数据加载器
        normalized_data: 归一化后的数据
    Returns:
        history: 训练历史（损失和时间）
        metrics: 性能指标
    """
    # 创建批次
    batches = data_loader.create_stock_batches(normalized_data, seq_len, batch_size)
    
    # 初始化模型
    input_size = 1
    output_size = 1
    
    if model_type == 'min_gru':
        model = MinGRU(input_size, hidden_size, output_size, learning_rate)
    elif model_type == 'mamba':
        model = Mamba(input_size, hidden_size, output_size, state_size=64, kernel_size=4, learning_rate=learning_rate)
    
    # 训练历史
    history = {
        'train_loss': [],
        'epoch_time': []
    }
    
    # 训练循环
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0.0
        
        for x_batch, y_batch in batches:
            if model_type == 'min_gru':
                h_prev = np.zeros((hidden_size, batch_size))
                y_pred, h = model.forward(x_batch, h_prev)
                y_pred_last = y_pred[-1, :, :]
                
                # 损失计算
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
                y_pred_last = y_pred[-1, :, :]
                
                # 损失计算
                loss = np.mean((y_pred_last - y_batch) ** 2)
                epoch_loss += loss
                
                # 反向传播
                dy = np.zeros_like(y_pred)
                dy[-1, :, :] = 2 * (y_pred_last - y_batch) / batch_size
                ds_next = np.zeros((model.state_size, batch_size))
                dx, ds_prev = model.backward(dy, ds_next)
            
            # 参数更新
            model.update()
        
        # 计算平均损失和时间
        avg_loss = epoch_loss / len(batches)
        epoch_time = time.time() - start_time
        
        history['train_loss'].append(avg_loss)
        history['epoch_time'].append(epoch_time)
        
        print(f"{model_type} epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}, time={epoch_time:.2f}s")
    
    # 验证模型
    print(f"Validating {model_type} model...")
    
    # 使用最后一个批次作为验证
    x_val, y_val = batches[-1]
    
    if model_type == 'min_gru':
        h_prev = np.zeros((hidden_size, batch_size))
        y_pred, _ = model.forward(x_val, h_prev)
    elif model_type == 'mamba':
        s_prev = np.zeros((model.state_size, batch_size))
        y_pred, _ = model.forward(x_val, s_prev)
    
    y_pred_last = y_pred[-1, :, :]
    metrics = calculate_metrics(y_val, y_pred_last)
    
    return history, metrics

# 主函数
def main():
    # 超参数
    seq_len = 30
    batch_size = 32
    epochs = 5
    learning_rate = 0.01
    hidden_size = 128
    
    print("Stock Time Series Prediction Benchmark: MinGRU vs Mamba")
    print(f"Hyperparameters: seq_len={seq_len}, batch_size={batch_size}, epochs={epochs}, lr={learning_rate}, hidden_size={hidden_size}")
    
    # 数据加载与预处理
    data_loader = DataLoader(data_dir='data')
    
    # 加载股票数据
    data = data_loader.load_yahoo_stock(ticker='AAPL', start_date='2010-01-01', end_date='2023-12-31')
    
    # 预处理数据
    feature_data, normalized_data, scaler = data_loader.preprocess_stock_data(data, feature='Close')
    
    # 训练并评估MinGRU
    print("\n" + "="*50)
    print("Training and evaluating MinGRU...")
    mingru_history, mingru_metrics = train_and_evaluate(
        'min_gru', seq_len, batch_size, epochs, learning_rate, hidden_size, data_loader, normalized_data
    )
    
    # 训练并评估Mamba
    print("\n" + "="*50)
    print("Training and evaluating Mamba...")
    mamba_history, mamba_metrics = train_and_evaluate(
        'mamba', seq_len, batch_size, epochs, learning_rate, hidden_size, data_loader, normalized_data
    )
    
    # 结果汇总
    print("\n" + "="*60)
    print("PERFORMANCE METRICS COMPARISON")
    print("="*60)
    print(f"{'Metric':<10} {'MinGRU':<15} {'Mamba':<15}")
    print("-"*60)
    for metric in ['MSE', 'RMSE', 'MAE', 'R²']:
        mingru_val = mingru_metrics[metric]
        mamba_val = mamba_metrics[metric]
        print(f"{metric:<10} {mingru_val:<15.6f} {mamba_val:<15.6f}")
    
    print("\n" + "="*60)
    print("TRAINING HISTORY")
    print("="*60)
    print(f"{'Epoch':<6} {'MinGRU Loss':<15} {'Mamba Loss':<15} {'MinGRU Time(s)':<18} {'Mamba Time(s)':<18}")
    print("-"*60)
    for i in range(epochs):
        mingru_loss = mingru_history['train_loss'][i]
        mamba_loss = mamba_history['train_loss'][i]
        mingru_time = mingru_history['epoch_time'][i]
        mamba_time = mamba_history['epoch_time'][i]
        print(f"{i+1:<6} {mingru_loss:<15.6f} {mamba_loss:<15.6f} {mingru_time:<18.2f} {mamba_time:<18.2f}")
    
    # 可视化
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 绘制训练损失对比图
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs+1), mingru_history['train_loss'], label='MinGRU', marker='o')
    plt.plot(range(1, epochs+1), mamba_history['train_loss'], label='Mamba', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss (MSE)')
    plt.title('MinGRU vs Mamba: Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('mamba_vs_mingru.png', dpi=300)
    print("Visualization saved as 'mamba_vs_mingru.png'")
    
    # 绘制训练时间对比图
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(epochs) - 0.2, mingru_history['epoch_time'], width=0.4, label='MinGRU')
    plt.bar(np.arange(epochs) + 0.2, mamba_history['epoch_time'], width=0.4, label='Mamba')
    plt.xlabel('Epoch')
    plt.ylabel('Training Time (seconds)')
    plt.title('MinGRU vs Mamba: Training Time Comparison')
    plt.xticks(np.arange(epochs), [f'Epoch {i+1}' for i in range(epochs)])
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('training_time_comparison.png', dpi=300)
    print("Training time visualization saved as 'training_time_comparison.png'")
    
    # 关键分析
    print("\n" + "="*60)
    print("KEY ANALYSIS")
    print("="*60)
    
    # 收敛速度分析（首2个epoch损失下降幅度）
    mingru_drop = mingru_history['train_loss'][0] - mingru_history['train_loss'][1]
    mamba_drop = mamba_history['train_loss'][0] - mamba_history['train_loss'][1]
    print(f"Convergence Speed (Epoch 1-2 loss drop): MinGRU={mingru_drop:.6f}, Mamba={mamba_drop:.6f}")
    
    # 最终损失比较
    mingru_final = mingru_history['train_loss'][-1]
    mamba_final = mamba_history['train_loss'][-1]
    print(f"Final Training Loss: MinGRU={mingru_final:.6f}, Mamba={mamba_final:.6f}")
    
    # 训练效率比较
    mingru_total_time = sum(mingru_history['epoch_time'])
    mamba_total_time = sum(mamba_history['epoch_time'])
    print(f"Total Training Time: MinGRU={mingru_total_time:.2f}s, Mamba={mamba_total_time:.2f}s")
    
    # 泛化能力比较
    print(f"Generalization (R²): MinGRU={mingru_metrics['R²']:.6f}, Mamba={mamba_metrics['R²']:.6f}")
    
    print("\nBenchmark completed!")

if __name__ == "__main__":
    main()