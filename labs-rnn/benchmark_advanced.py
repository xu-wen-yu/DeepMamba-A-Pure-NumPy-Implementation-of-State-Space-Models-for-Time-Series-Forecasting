"""
增强版基准测试脚本
展示优化器、正则化、多特征数据、深层模型等新功能
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from utils.data_loader import DataLoader
from utils.optimizers import Adam, RMSProp, SGD, get_optimizer
from models.min_gru import MinGRU
from models.mamba import Mamba
from models.deep_mamba import DeepMamba, BidirectionalMamba

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'stix'  # 使用STIX字体渲染数学符号


# ==================== 性能指标计算 ====================

def calculate_metrics(y_true, y_pred):
    """计算性能指标"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}


# ==================== 训练函数 ====================

def train_model(model, batches, epochs, optimizer=None, weight_decay=0.0, 
                model_type='mamba', hidden_size=128, verbose=True):
    """通用模型训练函数
    
    Args:
        model: 模型实例
        batches: 训练批次
        epochs: 训练轮数
        optimizer: 优化器实例（如果为None则使用模型内置的update方法）
        weight_decay: L2正则化系数
        model_type: 模型类型
        hidden_size: 隐藏层大小
        verbose: 是否打印训练信息
        
    Returns:
        history: 训练历史
    """
    history = {'train_loss': [], 'epoch_time': []}
    batch_size = batches[0][0].shape[-1]
    
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0.0
        
        for x_batch, y_batch in batches:
            # 前向传播
            if model_type == 'min_gru':
                h_prev = np.zeros((hidden_size, batch_size))
                y_pred, _ = model.forward(x_batch, h_prev)
                y_pred_last = y_pred[-1, :, :]
                
                # 损失计算
                loss = np.mean((y_pred_last - y_batch) ** 2)
                if weight_decay > 0:
                    loss += model.get_l2_loss(weight_decay)
                epoch_loss += loss
                
                # 反向传播
                dy = np.zeros_like(y_pred)
                dy[-1, :, :] = 2 * (y_pred_last - y_batch) / batch_size
                dh_next = np.zeros((hidden_size, batch_size))
                model.backward(dy, dh_next)
                
            elif model_type == 'mamba':
                s_prev = np.zeros((model.state_size, batch_size))
                y_pred, _ = model.forward(x_batch, s_prev)
                y_pred_last = y_pred[-1, :, :]
                
                loss = np.mean((y_pred_last - y_batch) ** 2)
                if weight_decay > 0:
                    loss += model.get_l2_loss(weight_decay)
                epoch_loss += loss
                
                dy = np.zeros_like(y_pred)
                dy[-1, :, :] = 2 * (y_pred_last - y_batch) / batch_size
                ds_next = np.zeros((model.state_size, batch_size))
                model.backward(dy, ds_next)
                
            elif model_type == 'deep_mamba':
                y_pred, _ = model.forward(x_batch, training=True)
                y_pred_last = y_pred[-1, :, :]
                
                loss = np.mean((y_pred_last - y_batch) ** 2)
                if weight_decay > 0:
                    loss += model.get_l2_loss(weight_decay)
                epoch_loss += loss
                
                dy = np.zeros_like(y_pred)
                dy[-1, :, :] = 2 * (y_pred_last - y_batch) / batch_size
                model.backward(dy)
            
            # 参数更新
            if optimizer is not None:
                optimizer.step(model, weight_decay)
            else:
                model.update(weight_decay=weight_decay)
        
        avg_loss = epoch_loss / len(batches)
        epoch_time = time.time() - start_time
        
        history['train_loss'].append(avg_loss)
        history['epoch_time'].append(epoch_time)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}, time={epoch_time:.2f}s")
    
    return history


def evaluate_model(model, batches, model_type='mamba', hidden_size=128):
    """评估模型"""
    batch_size = batches[0][0].shape[-1]
    all_y_true = []
    all_y_pred = []
    
    for x_batch, y_batch in batches:
        if model_type == 'min_gru':
            h_prev = np.zeros((hidden_size, batch_size))
            y_pred, _ = model.forward(x_batch, h_prev)
        elif model_type == 'mamba':
            s_prev = np.zeros((model.state_size, batch_size))
            y_pred, _ = model.forward(x_batch, s_prev)
        elif model_type == 'deep_mamba':
            y_pred, _ = model.forward(x_batch, training=False)
        
        y_pred_last = y_pred[-1, :, :]
        all_y_true.append(y_batch)
        all_y_pred.append(y_pred_last)
    
    y_true = np.concatenate(all_y_true, axis=-1)
    y_pred = np.concatenate(all_y_pred, axis=-1)
    
    return calculate_metrics(y_true, y_pred)


# ==================== 实验1：优化器对比 ====================

def experiment_optimizers():
    """对比不同优化器的性能"""
    print("\n" + "="*70)
    print("实验1：优化器对比 (SGD vs Adam vs RMSProp)")
    print("="*70)
    
    # 数据准备
    data_loader = DataLoader(data_dir='data')
    data = data_loader.load_yahoo_stock(ticker='AAPL')
    _, normalized_data, _ = data_loader.preprocess_stock_data(data, feature='Close')
    
    seq_len, batch_size, epochs = 30, 32, 10
    hidden_size = 64
    
    batches = data_loader.create_stock_batches(normalized_data, seq_len, batch_size)
    train_batches, test_batches = data_loader.train_test_split(batches, train_ratio=0.8)
    
    optimizers_config = [
        ('SGD', SGD(lr=0.01, momentum=0.9)),
        ('Adam', Adam(lr=0.001)),
        ('RMSProp', RMSProp(lr=0.001))
    ]
    
    results = {}
    
    for opt_name, optimizer in optimizers_config:
        print(f"\n训练 Mamba + {opt_name}...")
        
        model = Mamba(1, hidden_size, 1, state_size=32, learning_rate=0.01)
        optimizer.reset()
        
        history = train_model(
            model, train_batches, epochs, 
            optimizer=optimizer, weight_decay=0.0001,
            model_type='mamba', hidden_size=hidden_size
        )
        
        metrics = evaluate_model(model, test_batches, model_type='mamba', hidden_size=hidden_size)
        results[opt_name] = {'history': history, 'metrics': metrics}
        
        print(f"  测试集 - MSE: {metrics['MSE']:.6f}, R²: {metrics['R²']:.4f}")
    
    # 绘制对比图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for opt_name in results:
        plt.plot(results[opt_name]['history']['train_loss'], label=opt_name, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('优化器对比：训练损失')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    opt_names = list(results.keys())
    r2_scores = [results[name]['metrics']['R²'] for name in opt_names]
    plt.bar(opt_names, r2_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.ylabel('$R^2$ Score')
    plt.title('优化器对比：测试集$R^2$')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=300)
    print("\n图表已保存: optimizer_comparison.png")
    
    return results


# ==================== 实验2：正则化效果 ====================

def experiment_regularization():
    """对比不同正则化强度的效果"""
    print("\n" + "="*70)
    print("实验2：L2正则化效果对比")
    print("="*70)
    
    data_loader = DataLoader(data_dir='data')
    data = data_loader.load_yahoo_stock(ticker='AAPL')
    _, normalized_data, _ = data_loader.preprocess_stock_data(data, feature='Close')
    
    seq_len, batch_size, epochs = 30, 32, 15
    hidden_size = 64
    
    batches = data_loader.create_stock_batches(normalized_data, seq_len, batch_size)
    train_batches, test_batches = data_loader.train_test_split(batches, train_ratio=0.8)
    
    weight_decays = [0.0, 0.0001, 0.001, 0.01]
    results = {}
    
    for wd in weight_decays:
        print(f"\n训练 Mamba (weight_decay={wd})...")
        
        model = Mamba(1, hidden_size, 1, state_size=32, learning_rate=0.01)
        optimizer = Adam(lr=0.001)
        
        history = train_model(
            model, train_batches, epochs,
            optimizer=optimizer, weight_decay=wd,
            model_type='mamba', hidden_size=hidden_size
        )
        
        metrics = evaluate_model(model, test_batches, model_type='mamba', hidden_size=hidden_size)
        results[wd] = {'history': history, 'metrics': metrics}
        
        print(f"  测试集 - MSE: {metrics['MSE']:.6f}, R²: {metrics['R²']:.4f}")
    
    # 绘制对比图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for wd in results:
        plt.plot(results[wd]['history']['train_loss'], label=f'λ={wd}', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('L2正则化：训练损失')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    wds = [str(wd) for wd in weight_decays]
    r2_scores = [results[wd]['metrics']['R²'] for wd in weight_decays]
    plt.bar(wds, r2_scores, color='#9b59b6')
    plt.xlabel('Weight Decay (λ)')
    plt.ylabel('$R^2$ Score')
    plt.title('L2正则化：测试集$R^2$')
    
    plt.tight_layout()
    plt.savefig('regularization_comparison.png', dpi=300)
    print("\n图表已保存: regularization_comparison.png")
    
    return results


# ==================== 实验3：深层模型对比 ====================

def experiment_deep_models():
    """对比不同深度的模型"""
    print("\n" + "="*70)
    print("实验3：模型深度对比 (1层 vs 2层 vs 3层)")
    print("="*70)
    
    data_loader = DataLoader(data_dir='data')
    data = data_loader.load_yahoo_stock(ticker='AAPL')
    _, normalized_data, _ = data_loader.preprocess_stock_data(data, feature='Close')
    
    seq_len, batch_size, epochs = 30, 32, 10
    hidden_size = 64
    
    batches = data_loader.create_stock_batches(normalized_data, seq_len, batch_size)
    train_batches, test_batches = data_loader.train_test_split(batches, train_ratio=0.8)
    
    layer_configs = [1, 2, 3]
    results = {}
    
    for num_layers in layer_configs:
        print(f"\n训练 DeepMamba ({num_layers}层)...")
        
        model = DeepMamba(
            input_size=1, hidden_size=hidden_size, output_size=1,
            num_layers=num_layers, state_size=32,
            use_residual=True, use_layer_norm=True,
            learning_rate=0.001
        )
        
        print(f"  参数数量: {model.get_num_parameters():,}")
        
        optimizer = Adam(lr=0.001)
        
        history = train_model(
            model, train_batches, epochs,
            optimizer=optimizer, weight_decay=0.0001,
            model_type='deep_mamba', hidden_size=hidden_size
        )
        
        metrics = evaluate_model(model, test_batches, model_type='deep_mamba', hidden_size=hidden_size)
        results[num_layers] = {
            'history': history, 
            'metrics': metrics,
            'num_params': model.get_num_parameters()
        }
        
        print(f"  测试集 - MSE: {metrics['MSE']:.6f}, R²: {metrics['R²']:.4f}")
    
    # 绘制对比图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for nl in results:
        plt.plot(results[nl]['history']['train_loss'], label=f'{nl}层', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('模型深度：训练损失')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    layers = [str(nl) for nl in layer_configs]
    r2_scores = [results[nl]['metrics']['R²'] for nl in layer_configs]
    plt.bar(layers, r2_scores, color='#1abc9c')
    plt.xlabel('层数')
    plt.ylabel('$R^2$ Score')
    plt.title('模型深度：测试集$R^2$')
    
    plt.subplot(1, 3, 3)
    params = [results[nl]['num_params'] for nl in layer_configs]
    plt.bar(layers, params, color='#f39c12')
    plt.xlabel('层数')
    plt.ylabel('参数数量')
    plt.title('模型深度：参数数量')
    
    plt.tight_layout()
    plt.savefig('deep_model_comparison.png', dpi=300)
    print("\n图表已保存: deep_model_comparison.png")
    
    return results


# ==================== 实验4：多特征数据 ====================

def experiment_multifeature():
    """使用多特征数据进行训练"""
    print("\n" + "="*70)
    print("实验4：多特征股票数据预测")
    print("="*70)
    
    data_loader = DataLoader(data_dir='data')
    
    # 加载多特征数据
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    result = data_loader.load_yahoo_stock_multifeature(
        ticker='AAPL', features=features
    )
    
    print(f"特征: {result['features']}")
    print(f"数据形状: {result['data'].shape}")
    
    seq_len, batch_size, epochs = 30, 32, 10
    n_features = len(features)
    hidden_size = 64
    
    # 创建多特征批次（预测Close价格，索引3）
    batches = data_loader.create_multifeature_batches(
        result['data'], seq_len, batch_size, target_idx=3
    )
    train_batches, test_batches = data_loader.train_test_split(batches, train_ratio=0.8)
    
    print(f"\n训练 DeepMamba (多特征输入)...")
    
    model = DeepMamba(
        input_size=n_features, hidden_size=hidden_size, output_size=1,
        num_layers=2, state_size=32,
        use_residual=True, use_layer_norm=True,
        learning_rate=0.001
    )
    
    optimizer = Adam(lr=0.001)
    
    history = train_model(
        model, train_batches, epochs,
        optimizer=optimizer, weight_decay=0.0001,
        model_type='deep_mamba', hidden_size=hidden_size
    )
    
    metrics = evaluate_model(model, test_batches, model_type='deep_mamba', hidden_size=hidden_size)
    
    print(f"\n测试集结果:")
    print(f"  MSE: {metrics['MSE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAE: {metrics['MAE']:.6f}")
    print(f"  R²: {metrics['R²']:.4f}")
    
    return {'history': history, 'metrics': metrics}


# ==================== 实验5：不同数据集 ====================

def experiment_datasets():
    """在不同数据集上测试模型"""
    print("\n" + "="*70)
    print("实验5：不同数据集对比")
    print("="*70)
    
    data_loader = DataLoader(data_dir='data')
    
    datasets = {
        '股票数据': lambda: data_loader.load_yahoo_stock_multifeature('AAPL'),
        '天气数据': lambda: data_loader.load_weather_data('Beijing'),
        '合成正弦波': lambda: data_loader.load_synthetic_sine(n_samples=5000, n_features=3)
    }
    
    seq_len, batch_size, epochs = 30, 32, 10
    hidden_size = 64
    
    results = {}
    
    for name, load_func in datasets.items():
        print(f"\n加载 {name}...")
        data_result = load_func()
        
        n_features = len(data_result['features'])
        print(f"  特征数: {n_features}, 样本数: {len(data_result['data'])}")
        
        batches = data_loader.create_multifeature_batches(
            data_result['data'], seq_len, batch_size, target_idx=0
        )
        
        if len(batches) < 5:
            print(f"  数据量不足，跳过...")
            continue
        
        train_batches, test_batches = data_loader.train_test_split(batches, train_ratio=0.8)
        
        print(f"  训练 DeepMamba...")
        
        model = DeepMamba(
            input_size=n_features, hidden_size=hidden_size, output_size=1,
            num_layers=2, state_size=32,
            learning_rate=0.001
        )
        
        optimizer = Adam(lr=0.001)
        
        history = train_model(
            model, train_batches, epochs,
            optimizer=optimizer, weight_decay=0.0001,
            model_type='deep_mamba', hidden_size=hidden_size,
            verbose=False
        )
        
        metrics = evaluate_model(model, test_batches, model_type='deep_mamba', hidden_size=hidden_size)
        results[name] = {'history': history, 'metrics': metrics}
        
        print(f"  测试集 - MSE: {metrics['MSE']:.6f}, R²: {metrics['R²']:.4f}")
    
    # 绘制对比图
    if results:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        for name in results:
            plt.plot(results[name]['history']['train_loss'], label=name, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('不同数据集：训练损失')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        names = list(results.keys())
        r2_scores = [results[name]['metrics']['R²'] for name in names]
        colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(names)]
        plt.bar(names, r2_scores, color=colors)
        plt.ylabel('$R^2$ Score')
        plt.title('不同数据集：测试集$R^2$')
        plt.xticks(rotation=15)
        
        plt.tight_layout()
        plt.savefig('dataset_comparison.png', dpi=300)
        print("\n图表已保存: dataset_comparison.png")
    
    return results


# ==================== 主函数 ====================

def main():
    print("="*70)
    print("增强版基准测试：Mamba模型优化实验")
    print("="*70)
    
    # 运行所有实验
    print("\n开始运行实验...")
    
    # 实验1：优化器对比
    opt_results = experiment_optimizers()
    
    # 实验2：正则化效果
    reg_results = experiment_regularization()
    
    # 实验3：深层模型
    deep_results = experiment_deep_models()
    
    # 实验4：多特征数据
    multi_results = experiment_multifeature()
    
    # 实验5：不同数据集
    dataset_results = experiment_datasets()
    
    # 总结
    print("\n" + "="*70)
    print("实验总结")
    print("="*70)
    
    print("\n1. 优化器对比:")
    for name, res in opt_results.items():
        print(f"   {name}: R²={res['metrics']['R²']:.4f}")
    
    print("\n2. 正则化效果:")
    for wd, res in reg_results.items():
        print(f"   λ={wd}: R²={res['metrics']['R²']:.4f}")
    
    print("\n3. 模型深度:")
    for nl, res in deep_results.items():
        print(f"   {nl}层: R²={res['metrics']['R²']:.4f}, 参数={res['num_params']:,}")
    
    print("\n4. 多特征预测:")
    print(f"   R²={multi_results['metrics']['R²']:.4f}")
    
    print("\n5. 数据集对比:")
    for name, res in dataset_results.items():
        print(f"   {name}: R²={res['metrics']['R²']:.4f}")
    
    print("\n所有实验完成！")


if __name__ == "__main__":
    main()

