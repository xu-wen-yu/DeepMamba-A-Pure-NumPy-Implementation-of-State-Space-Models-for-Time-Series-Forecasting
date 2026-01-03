import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_yahoo_stock(self, ticker='AAPL', start_date='2010-01-01', end_date='2023-12-31'):
        """加载雅虎股票数据并保存到本地"""
        try:
            # 尝试下载数据
            data = yf.download(ticker, start=start_date, end=end_date)
            
            # 检查数据是否为空
            if data.empty:
                print(f"Warning: No data downloaded for {ticker}. Using simulated data.")
                return self._create_simulated_data()
            
            # 保存原始数据
            data_path = os.path.join(self.data_dir, f'{ticker}_raw.csv')
            data.to_csv(data_path)
            
            return data
        except Exception as e:
            print(f"Error downloading data: {e}. Using simulated data.")
            return self._create_simulated_data()
    
    def load_yahoo_stock_multifeature(self, ticker='AAPL', start_date='2010-01-01', 
                                       end_date='2023-12-31', features=None):
        """加载雅虎股票数据，支持多特征
        
        Args:
            ticker: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            features: 特征列表，默认['Open', 'High', 'Low', 'Close', 'Volume']
            
        Returns:
            dict: 包含数据、缩放器、原始数据和特征列表
        """
        if features is None:
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        try:
            # 下载数据
            df = yf.download(ticker, start=start_date, end=end_date)
            
            if df.empty:
                print(f"Warning: No data downloaded for {ticker}. Using simulated data.")
                df = self._create_simulated_data()
            
            # 使用多个特征
            data = df[features].values  # (n_samples, n_features)
            
            # 数据归一化
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            # 保存数据
            data_path = os.path.join(self.data_dir, f'{ticker}_stock_data.csv')
            df.to_csv(data_path)
            
            return {
                'data': scaled_data,
                'scaler': scaler,
                'original_data': data,
                'features': features,
                'df': df
            }
        except Exception as e:
            print(f"Error downloading data: {e}. Using simulated data.")
            df = self._create_simulated_data()
            data = df[features].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            return {
                'data': scaled_data,
                'scaler': scaler,
                'original_data': data,
                'features': features,
                'df': df
            }
    
    def _create_simulated_data(self):
        """创建模拟股票数据"""
        # 创建日期范围
        dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='B')
        n_samples = len(dates)
        
        # 创建模拟的Close价格数据（使用随机游走+趋势）
        np.random.seed(42)
        trend = np.linspace(0, 100, n_samples)
        random_walk = np.cumsum(np.random.normal(0, 1, n_samples))
        close_prices = 50 + trend + random_walk
        
        # 创建其他价格字段（基于Close价格）
        high_prices = close_prices * (1 + np.random.uniform(0, 0.02, n_samples))
        low_prices = close_prices * (1 - np.random.uniform(0, 0.02, n_samples))
        open_prices = close_prices * (1 + np.random.uniform(-0.01, 0.01, n_samples))
        adj_close = close_prices * (1 + np.random.uniform(-0.005, 0.005, n_samples))
        volume = np.random.randint(1000000, 10000000, n_samples)
        
        # 创建DataFrame
        data = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Adj Close': adj_close,
            'Volume': volume
        }, index=dates)
        
        return data
    
    def preprocess_stock_data(self, data, feature='Close'):
        """预处理股票数据：仅保留指定特征，归一化"""
        # 仅保留指定特征
        feature_data = data[[feature]].values
        
        # 归一化到[0,1]
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(feature_data)
        
        # 保存归一化后的数据
        normalized_path = os.path.join(self.data_dir, f'{feature}_normalized.csv')
        np.savetxt(normalized_path, normalized_data, delimiter=',')
        
        return feature_data, normalized_data, scaler
    
    def create_stock_batches(self, normalized_data, seq_len, batch_size):
        """创建股票数据批次
        Args:
            normalized_data: 归一化后的数据，形状(n_samples, 1)
            seq_len: 序列长度
            batch_size: 批次大小
        Returns:
            batches: 批次列表，每个批次为(x, y)，x形状为(seq_len, input_size, batch_size)，y形状为(1, batch_size)
        """
        input_size = 1
        n_samples = len(normalized_data) - seq_len
        batches = []
        
        # 计算可生成的完整批次数量
        n_batches = n_samples // batch_size
        
        # 遍历生成批次
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            x_batch = []
            y_batch = []
            
            for j in range(end_idx - start_idx):
                sample_idx = start_idx + j
                x_sample = normalized_data[sample_idx:sample_idx+seq_len].reshape(seq_len, input_size)
                y_sample = normalized_data[sample_idx+seq_len].reshape(1, input_size)
                
                x_batch.append(x_sample)
                y_batch.append(y_sample)
            
            # 转换为正确的形状：(seq_len, input_size, batch_size)
            x_batch = np.transpose(np.array(x_batch), (1, 2, 0))
            y_batch = np.transpose(np.array(y_batch), (1, 2, 0))[:, 0, :]  # 形状为(1, batch_size)
            
            batches.append((x_batch, y_batch))
        
        return batches
    
    def create_multifeature_batches(self, normalized_data, seq_len, batch_size, 
                                     target_idx=3, predict_all=False):
        """创建多特征数据批次
        
        Args:
            normalized_data: 归一化后的数据，形状(n_samples, n_features)
            seq_len: 序列长度
            batch_size: 批次大小
            target_idx: 目标特征索引（默认3，即Close价格）
            predict_all: 是否预测所有特征，默认False只预测target_idx
            
        Returns:
            batches: 批次列表，每个批次为(x, y)
                     x形状为(seq_len, n_features, batch_size)
                     y形状为(output_size, batch_size)
        """
        n_samples, n_features = normalized_data.shape
        n_samples = n_samples - seq_len
        batches = []
        
        # 计算可生成的完整批次数量
        n_batches = n_samples // batch_size
        
        # 遍历生成批次
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            x_batch = []
            y_batch = []
            
            for j in range(end_idx - start_idx):
                sample_idx = start_idx + j
                x_sample = normalized_data[sample_idx:sample_idx+seq_len]  # (seq_len, n_features)
                
                if predict_all:
                    y_sample = normalized_data[sample_idx+seq_len]  # (n_features,)
                else:
                    y_sample = normalized_data[sample_idx+seq_len, target_idx:target_idx+1]  # (1,)
                
                x_batch.append(x_sample)
                y_batch.append(y_sample)
            
            # 转换为正确的形状：(seq_len, n_features, batch_size)
            x_batch = np.transpose(np.array(x_batch), (1, 2, 0))
            y_batch = np.array(y_batch).T  # (output_size, batch_size)
            
            batches.append((x_batch, y_batch))
        
        return batches
    
    def train_test_split(self, batches, train_ratio=0.8):
        """划分训练集和测试集
        
        Args:
            batches: 批次列表
            train_ratio: 训练集比例
            
        Returns:
            train_batches: 训练批次
            test_batches: 测试批次
        """
        n_train = int(len(batches) * train_ratio)
        train_batches = batches[:n_train]
        test_batches = batches[n_train:]
        return train_batches, test_batches
    
    # ==================== 天气数据集 ====================
    
    def load_weather_data(self, city='Beijing', start_date='2010-01-01', 
                          end_date='2023-12-31'):
        """加载天气数据（模拟数据）
        
        Args:
            city: 城市名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            dict: 包含数据、缩放器、原始数据和特征列表
        """
        # 创建日期范围
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_samples = len(dates)
        
        np.random.seed(42)
        
        # 模拟天气数据
        # 温度：有季节性变化
        day_of_year = np.array([d.dayofyear for d in dates])
        temperature = 15 + 15 * np.sin(2 * np.pi * day_of_year / 365) + \
                      np.random.normal(0, 3, n_samples)
        
        # 湿度：与温度负相关
        humidity = 60 - 0.5 * temperature + np.random.normal(0, 10, n_samples)
        humidity = np.clip(humidity, 20, 100)
        
        # 气压：随机波动
        pressure = 1013 + np.random.normal(0, 10, n_samples)
        
        # 风速：随机
        wind_speed = np.abs(np.random.normal(5, 3, n_samples))
        
        # 降水量：与湿度相关
        precipitation = np.maximum(0, (humidity - 50) * 0.1 + np.random.exponential(2, n_samples))
        
        # 创建DataFrame
        df = pd.DataFrame({
            'Temperature': temperature,
            'Humidity': humidity,
            'Pressure': pressure,
            'WindSpeed': wind_speed,
            'Precipitation': precipitation
        }, index=dates)
        
        features = ['Temperature', 'Humidity', 'Pressure', 'WindSpeed', 'Precipitation']
        data = df[features].values
        
        # 数据归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # 保存数据
        data_path = os.path.join(self.data_dir, f'{city}_weather_data.csv')
        df.to_csv(data_path)
        
        print(f"Weather data for {city} loaded: {n_samples} samples, {len(features)} features")
        
        return {
            'data': scaled_data,
            'scaler': scaler,
            'original_data': data,
            'features': features,
            'df': df
        }
    
    # ==================== 交通流量数据集 ====================
    
    def load_traffic_data(self, city='Beijing', start_date='2020-01-01', 
                          end_date='2023-12-31'):
        """加载交通流量数据（模拟数据）
        
        Args:
            city: 城市名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            dict: 包含数据、缩放器、原始数据和特征列表
        """
        # 创建小时级别的日期范围
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        n_samples = len(dates)
        
        np.random.seed(42)
        
        # 模拟交通流量数据
        hour_of_day = np.array([d.hour for d in dates])
        day_of_week = np.array([d.dayofweek for d in dates])
        
        # 基础流量：早晚高峰
        base_flow = 500 + 300 * np.exp(-((hour_of_day - 8) ** 2) / 8) + \
                    250 * np.exp(-((hour_of_day - 18) ** 2) / 8)
        
        # 周末流量较低
        weekend_factor = np.where(day_of_week >= 5, 0.6, 1.0)
        
        # 主干道流量
        main_road_flow = base_flow * weekend_factor + np.random.normal(0, 50, n_samples)
        main_road_flow = np.maximum(0, main_road_flow)
        
        # 次干道流量
        secondary_road_flow = main_road_flow * 0.6 + np.random.normal(0, 30, n_samples)
        secondary_road_flow = np.maximum(0, secondary_road_flow)
        
        # 平均速度：与流量负相关
        avg_speed = 60 - 0.03 * main_road_flow + np.random.normal(0, 5, n_samples)
        avg_speed = np.clip(avg_speed, 10, 80)
        
        # 拥堵指数
        congestion_index = (1000 - main_road_flow) / 1000 * 10
        congestion_index = np.clip(congestion_index + np.random.normal(0, 0.5, n_samples), 0, 10)
        
        # 事故数量（泊松分布）
        accidents = np.random.poisson(0.1, n_samples)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'MainRoadFlow': main_road_flow,
            'SecondaryRoadFlow': secondary_road_flow,
            'AvgSpeed': avg_speed,
            'CongestionIndex': congestion_index,
            'Accidents': accidents,
            'HourOfDay': hour_of_day,
            'DayOfWeek': day_of_week
        }, index=dates)
        
        features = ['MainRoadFlow', 'SecondaryRoadFlow', 'AvgSpeed', 
                    'CongestionIndex', 'HourOfDay', 'DayOfWeek']
        data = df[features].values
        
        # 数据归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # 保存数据
        data_path = os.path.join(self.data_dir, f'{city}_traffic_data.csv')
        df.to_csv(data_path)
        
        print(f"Traffic data for {city} loaded: {n_samples} samples, {len(features)} features")
        
        return {
            'data': scaled_data,
            'scaler': scaler,
            'original_data': data,
            'features': features,
            'df': df
        }
    
    # ==================== 电力消耗数据集 ====================
    
    def load_electricity_data(self, start_date='2018-01-01', end_date='2023-12-31'):
        """加载电力消耗数据（模拟数据）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            dict: 包含数据、缩放器、原始数据和特征列表
        """
        # 创建小时级别的日期范围
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        n_samples = len(dates)
        
        np.random.seed(42)
        
        hour_of_day = np.array([d.hour for d in dates])
        day_of_year = np.array([d.dayofyear for d in dates])
        day_of_week = np.array([d.dayofweek for d in dates])
        
        # 基础负荷
        base_load = 1000
        
        # 日内变化（白天用电多）
        daily_pattern = 200 * np.sin(np.pi * hour_of_day / 12 - np.pi/4)
        
        # 季节性变化（夏冬用电多）
        seasonal_pattern = 300 * np.cos(2 * np.pi * day_of_year / 365)
        
        # 周末用电少
        weekend_factor = np.where(day_of_week >= 5, -100, 0)
        
        # 总负荷
        total_load = base_load + daily_pattern + seasonal_pattern + weekend_factor + \
                     np.random.normal(0, 50, n_samples)
        total_load = np.maximum(0, total_load)
        
        # 工业用电
        industrial_load = total_load * 0.4 + np.random.normal(0, 20, n_samples)
        
        # 居民用电
        residential_load = total_load * 0.35 + np.random.normal(0, 15, n_samples)
        
        # 商业用电
        commercial_load = total_load * 0.25 + np.random.normal(0, 10, n_samples)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'TotalLoad': total_load,
            'IndustrialLoad': industrial_load,
            'ResidentialLoad': residential_load,
            'CommercialLoad': commercial_load,
            'HourOfDay': hour_of_day,
            'DayOfWeek': day_of_week
        }, index=dates)
        
        features = ['TotalLoad', 'IndustrialLoad', 'ResidentialLoad', 
                    'CommercialLoad', 'HourOfDay', 'DayOfWeek']
        data = df[features].values
        
        # 数据归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # 保存数据
        data_path = os.path.join(self.data_dir, 'electricity_data.csv')
        df.to_csv(data_path)
        
        print(f"Electricity data loaded: {n_samples} samples, {len(features)} features")
        
        return {
            'data': scaled_data,
            'scaler': scaler,
            'original_data': data,
            'features': features,
            'df': df
        }
    
    # ==================== 正弦波合成数据集 ====================
    
    def load_synthetic_sine(self, n_samples=10000, n_features=3, noise_level=0.1):
        """生成正弦波合成数据集（用于测试模型）
        
        Args:
            n_samples: 样本数量
            n_features: 特征数量
            noise_level: 噪声水平
            
        Returns:
            dict: 包含数据、缩放器、原始数据和特征列表
        """
        np.random.seed(42)
        
        t = np.linspace(0, 100, n_samples)
        
        # 生成多个不同频率的正弦波
        data = np.zeros((n_samples, n_features))
        features = []
        
        for i in range(n_features):
            freq = 0.1 * (i + 1)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = 1 + 0.5 * i
            
            data[:, i] = amplitude * np.sin(2 * np.pi * freq * t + phase) + \
                         noise_level * np.random.randn(n_samples)
            features.append(f'Sine_{i+1}')
        
        # 数据归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # 创建DataFrame
        df = pd.DataFrame(data, columns=features)
        
        print(f"Synthetic sine data generated: {n_samples} samples, {n_features} features")
        
        return {
            'data': scaled_data,
            'scaler': scaler,
            'original_data': data,
            'features': features,
            'df': df
        }
    
    # ==================== 技术指标计算 ====================
    
    def add_technical_indicators(self, df):
        """为股票数据添加技术指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            df: 添加了技术指标的DataFrame
        """
        df = df.copy()
        
        # 移动平均线
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # 指数移动平均线
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        
        # 波动率
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # 价格变化率
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        
        # 删除NaN值
        df = df.dropna()
        
        return df
