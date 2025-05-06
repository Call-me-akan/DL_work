import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 数据加载和预处理
class DatasetLoader:
    def __init__(self, data_dir, img_size=(64, 64)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.classes = ['cat', 'dog']

    def augment_image(self, img_array):
        augmented_images = []
        # 原始图像
        augmented_images.append(img_array)
        
        # 水平翻转
        augmented_images.append(np.fliplr(img_array))
        
        # 随机旋转 (±15度)
        angle = np.random.uniform(-15, 15)
        rotated = np.array(Image.fromarray((img_array * 255).astype(np.uint8)).rotate(angle)) / 255.0
        augmented_images.append(rotated)
        
        # 随机亮度和对比度调整
        brightness = np.random.uniform(0.8, 1.2)
        contrast = np.random.uniform(0.8, 1.2)
        adjusted = img_array * brightness
        adjusted = (adjusted - 0.5) * contrast + 0.5
        adjusted = np.clip(adjusted, 0, 1)
        augmented_images.append(adjusted)
        
        return augmented_images

    def load_data(self):
        X, y = [], []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            for image_idx, img_name in enumerate(os.listdir(class_dir)):
                if image_idx > 1000:
                    break
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(self.img_size)
                    img_array = np.array(img) / 255.0
                    
                    # 应用数据增强
                    augmented_images = self.augment_image(img_array)
                    for aug_img in augmented_images:
                        X.append(aug_img)
                        y.append(class_idx)
                except:
                    print(f"Error loading {img_path}")

        X = np.array(X).transpose(0, 3, 1, 2)  # 转为NCHW格式
        y = np.array(y)
        return train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)


# 高效卷积层实现
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He初始化
        scale = np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)

        # 缓存用于反向传播
        self.cache = None
        self.dW = None
        self.db = None

    def im2col(self, x):
        N, C, H, W = x.shape
        K = self.kernel_size
        P = self.padding
        stride = self.stride

        # 添加padding
        x_padded = np.pad(x, [(0, 0), (0, 0), (P, P), (P, P)], mode='constant')

        # 计算输出尺寸
        H_out = (H + 2 * P - K) // stride + 1
        W_out = (W + 2 * P - K) // stride + 1

        # 提取图像块
        cols = np.zeros((N, C, K, K, H_out, W_out))
        for i in range(K):
            for j in range(K):
                cols[:, :, i, j, :, :] = x_padded[:, :, i:i + H_out * stride:stride, j:j + W_out * stride:stride]

        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)
        return cols

    def forward(self, x):
        N, C, H, W = x.shape
        K = self.kernel_size
        P = self.padding

        # im2col转换
        cols = self.im2col(x)
        W_reshaped = self.W.reshape(self.out_channels, -1).T

        # 矩阵乘法实现卷积
        out = np.dot(cols, W_reshaped) + self.b
        H_out = (H + 2 * P - K) // self.stride + 1
        W_out = (W + 2 * P - K) // self.stride + 1
        out = out.reshape(N, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)

        # 保存中间结果用于反向传播
        self.cache = (x, cols)
        return out

    def backward(self, dout):
        # print(f"Conv2D backward: dW shape={self.dW.shape}, db shape={self.db.shape}")
        x, cols = self.cache
        N, C, H, W = x.shape
        K = self.kernel_size

        # 转换梯度形状
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # 计算dW和db
        self.dW = np.dot(cols.T, dout_reshaped).reshape(self.W.shape)
        self.db = np.sum(dout_reshaped, axis=0)

        # 计算输入梯度
        W_reshaped = self.W.reshape(self.out_channels, -1)
        dcols = np.dot(dout_reshaped, W_reshaped)

        # col2im转换
        dx = self.col2im(dcols, x.shape)
        return dx

    def col2im(self, dcols, x_shape):
        N, C, H, W = x_shape
        K = self.kernel_size
        P = self.padding
        stride = self.stride

        H_padded = H + 2 * P
        W_padded = W + 2 * P
        dx_padded = np.zeros((N, C, H_padded, W_padded))

        cols_reshaped = dcols.reshape(N, (H_padded - K) // stride + 1, (W_padded - K) // stride + 1, C, K, K)
        cols_reshaped = cols_reshaped.transpose(0, 3, 4, 5, 1, 2)

        for i in range(K):
            for j in range(K):
                dx_padded[:, :, i:i + H_padded - K + 1:stride, j:j + W_padded - K + 1:stride] += cols_reshaped[:, :, i,
                                                                                                 j, :, :]

        # 移除padding
        if P > 0:
            dx = dx_padded[:, :, P:-P, P:-P]
        else:
            dx = dx_padded
        return dx

    def parameters(self):
        return [self.W, self.b]


# 批量归一化层
class BatchNorm:
    def __init__(self, channels, momentum=0.9):
        self.gamma = np.ones(channels)
        self.beta = np.zeros(channels)
        self.dgamma = None  # 新增：保存梯度
        self.dbeta = None
        self.momentum = momentum
        self.eps = 1e-5

        # 运行时统计量
        self.running_mean = np.zeros(channels)
        self.running_var = np.ones(channels)

        # 缓存
        self.cache = None
        self.training = True

    def forward(self, x):
        if self.training:
            mu = x.mean(axis=(0, 2, 3), keepdims=True)
            var = x.var(axis=(0, 2, 3), keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()
        else:
            mu = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)

        x_hat = (x - mu) / np.sqrt(var + self.eps)
        out = self.gamma.reshape(1, -1, 1, 1) * x_hat + self.beta.reshape(1, -1, 1, 1)

        self.cache = (x, mu, var, x_hat)
        return out

    def backward(self, dout):
        x, mu, var, x_hat = self.cache
        N, C, H, W = x.shape

        # 计算梯度
        self.dgamma = np.sum(dout * x_hat, axis=(0, 2, 3))  # 保存到类属性
        self.dbeta = np.sum(dout, axis=(0, 2, 3))

        dx_hat = dout * self.gamma.reshape(1, -1, 1, 1)
        dvar = np.sum(dx_hat * (x - mu) * (-0.5) * (var + self.eps) ** (-1.5), axis=(0, 2, 3), keepdims=True)
        dmu = np.sum(dx_hat * (-1 / np.sqrt(var + self.eps)), axis=(0, 2, 3), keepdims=True) + dvar * np.mean(
            -2 * (x - mu), axis=(0, 2, 3), keepdims=True)
        dx = dx_hat / np.sqrt(var + self.eps) + dvar * 2 * (x - mu) / (N * H * W) + dmu / (N * H * W)

        return dx  # 仅返回输入梯度

    def parameters(self):
        return [self.gamma, self.beta]


# ReLU激活函数
class ReLU:
    def forward(self, x):
        # 保存原始输入的维度
        self.original_shape = x.shape
        # 将输入展平为2D以兼容后续全连接层
        x_flatten = x.reshape(x.shape[0], -1)
        self.mask = (x_flatten <= 0)
        return np.maximum(0, x_flatten).reshape(self.original_shape)  # 恢复原形状

    def backward(self, grad):
        # 确保梯度形状与mask一致
        grad_flatten = grad.reshape(self.mask.shape)
        grad_flatten[self.mask] = 0
        return grad_flatten.reshape(self.original_shape)


# 残差块
class ResidualBlock:
    def __init__(self, in_channels, out_channels, stride=1):
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride, 1)
        self.bn1 = BatchNorm(out_channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(out_channels, out_channels, 3, 1, 1)
        self.bn2 = BatchNorm(out_channels)
        self.relu2 = ReLU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = Sequential(
                Conv2D(in_channels, out_channels, 1, stride),
                BatchNorm(out_channels)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut:
            residual = self.shortcut.forward(residual)

        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        out += residual
        out = self.relu2.forward(out)
        return out

    def backward(self, dout):
        # 反向传播ReLU2
        dout = self.relu2.backward(dout)

        # 主路径反向传播
        dout_main = self.bn2.backward(dout)  # bn2返回dx，非元组
        dout_main = self.conv2.backward(dout_main)
        dout_main = self.relu1.backward(dout_main)
        dout_main = self.bn1.backward(dout_main)
        dout_main = self.conv1.backward(dout_main)

        # 短路路径反向传播
        if self.shortcut:
            d_residual = self.shortcut.backward(dout)
        else:
            d_residual = dout

        # 合并梯度
        dx = dout_main + d_residual
        return dx  # 返回单一梯度

    def parameters(self):
        params = []
        # 主路径参数
        params += self.conv1.parameters()
        params += self.bn1.parameters()
        params += self.conv2.parameters()
        params += self.bn2.parameters()
        # 短路路径参数
        if self.shortcut is not None:
            params += self.shortcut.parameters()
        return params


# 全局平均池化
class GlobalAvgPool:
    def forward(self, x):
        # print(f"GlobalAvgPool input shape: {x.shape}")
        self.input_shape = x.shape
        return np.mean(x, axis=(2, 3))

    def backward(self, dout):
        # 将2D梯度扩展回4D形状（N, C, H, W）
        N, C = dout.shape
        H, W = self.input_shape[2], self.input_shape[3]
        dx = dout.reshape(N, C, 1, 1) / (H * W)  # 平均操作的梯度
        dx = np.repeat(dx, H, axis=2)  # 沿H维度复制
        dx = np.repeat(dx, W, axis=3)  # 沿W维度复制
        return dx


# 全连接层
class Dense:
    def __init__(self, in_dim, out_dim):
        scale = np.sqrt(2. / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)
        self.cache = None
        self.dW = None
        self.db = None

    def forward(self, x):
        # print(f"Dense input shape: {x.shape}")
        self.cache = x
        # print(f"Dense forward: input shape {x.shape}")
        output = np.dot(x, self.W) + self.b
        # print(f"Dense forward: output shape {output.shape}")
        return output

    def backward(self, dout):
        x = self.cache
        # print(f"Dense backward: dout shape {dout.shape}")
        self.dW = np.dot(x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        # print(f"Dense backward: dW shape {self.dW.shape}, db shape {self.db.shape}, dx shape {dx.shape}")
        return dx

    def parameters(self):
        return [self.W, self.b]


# 交叉熵损失
class CrossEntropyLoss:
    def __init__(self):
        self.cache = None

    def forward(self, x, y):
        m = y.shape[0]
        shifted = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        loss = -np.log(probs[range(m), y]).mean()
        self.cache = (probs, y)
        return loss

    def backward(self):
        probs, y = self.cache
        m = y.shape[0]
        grad = probs.copy()
        grad[range(m), y] -= 1
        # print(f"Loss backward: grad shape {grad.shape}")
        return grad / m


# ResNet模型
class ResNet:
    def __init__(self):
        self.layers = [
            Conv2D(3, 64, 7, stride=2, padding=3),
            BatchNorm(64),
            ReLU(),
            MaxPool2D(3, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            GlobalAvgPool(),
            Dense(256, 2)
        ]
        self.params = []
        # 递归收集所有子层参数
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                self.params += layer.parameters()

    def save_model(self, path):
        """保存模型参数到指定路径"""
        model_state = {
            'params': [p.copy() for p in self.params],
            'timestamp': np.datetime64('now')
        }
        np.save(path, model_state)
        print(f"模型已保存到: {path}")

    def load_model(self, path):
        """从指定路径加载模型参数"""
        try:
            model_state = np.load(path, allow_pickle=True).item()
            for i, p in enumerate(self.params):
                p[:] = model_state['params'][i]
            print(f"模型已从 {path} 加载")
            print(f"模型保存时间: {model_state['timestamp']}")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            raise

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        grads = []

        def collect_grads(layer, dout, p_layer=False):
            if hasattr(layer, 'backward') and p_layer==False:
                dout = layer.backward(dout)

            if isinstance(layer, (Conv2D, Dense)):
                grads.extend(reversed([layer.dW, layer.db]))
            elif isinstance(layer, BatchNorm):
                grads.extend(reversed([layer.dgamma, layer.dbeta]))
            elif isinstance(layer, (ResidualBlock, Sequential)):
                # 递归处理 ResidualBlock 或 Sequential 内部的层
                sub_layers = layer.layers if isinstance(layer, Sequential) else [
                    layer.conv1, layer.bn1, layer.relu1, layer.conv2, layer.bn2, layer.relu2
                ]
                if layer.shortcut:
                    sub_layers.extend(layer.shortcut.layers)

                for sub_layer in reversed(sub_layers):
                    dout = collect_grads(sub_layer, dout, True)

            return dout

        for layer in reversed(self.layers):
            dout = collect_grads(layer, dout)

        # 确保参数与梯度顺序严格一致
        return grads[::-1]

# 最大池化层
class MaxPool2D:
    def __init__(self, kernel_size, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        K = self.kernel_size
        stride = self.stride

        # 计算输出尺寸
        H_out = (H - K) // stride + 1
        W_out = (W - K) // stride + 1

        # 提取池化区域
        out = np.zeros((N, C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + K
                w_start = j * stride
                w_end = w_start + K

                x_slice = x[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.max(x_slice, axis=(2, 3))

        self.cache = (x, out)
        return out

    def backward(self, dout):
        x, out = self.cache
        N, C, H, W = x.shape
        K = self.kernel_size
        stride = self.stride

        dx = np.zeros_like(x)
        for i in range(dout.shape[2]):
            for j in range(dout.shape[3]):
                h_start = i * stride
                h_end = h_start + K
                w_start = j * stride
                w_end = w_start + K

                dout_slice = dout[:, :, i, j][:, :, np.newaxis, np.newaxis]
                x_slice = x[:, :, h_start:h_end, w_start:w_end]
                mask = (x_slice == np.max(x_slice, axis=(2, 3), keepdims=True))
                dx[:, :, h_start:h_end, w_start:w_end] += dout_slice * mask
        return dx


# 序列模型
class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params += layer.parameters()
        return params


# 训练器
class Trainer:
    def __init__(self, model, lr=0.001, momentum=0.9, weight_decay=1e-4):
        self.model = model
        self.lr = lr
        self.initial_lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p) for p in model.params]
        self.epoch = 0

    def step(self, grads):
        assert len(grads) == len(self.model.params), f"参数{len(grads)}与梯度数量{len(self.model.params)}不匹配"
        
        # 应用权重衰减
        for i, param in enumerate(self.model.params):
            grads[i] += self.weight_decay * param
            
        # 更新学习率
        self.lr = self.initial_lr * (0.1 ** (self.epoch // 30))  # 每30个epoch降低学习率
        
        for i, (param, grad) in enumerate(zip(self.model.params, grads)):
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
            param += self.velocities[i]

    def update_epoch(self):
        self.epoch += 1


# 训练流程
def train(data_dir='D:\DL_work\PetImages', epochs=100, batch_size=32, save_path='model.npy'):
    # 加载数据
    loader = DatasetLoader(data_dir)
    X_train, X_val, y_train, y_val = loader.load_data()

    # 初始化模型
    model = ResNet()
    criterion = CrossEntropyLoss()
    trainer = Trainer(model, lr=0.01)  # 使用更大的初始学习率

    best_acc = 0
    patience = 10  # 早停耐心值
    no_improve_epochs = 0
    
    # 记录训练历史
    train_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.training = True  # 启用训练模式
        epoch_losses = []
        
        # 打乱训练数据
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            # 前向传播
            output = model.forward(batch_X)
            loss = criterion.forward(output, batch_y)
            epoch_losses.append(loss)

            # 反向传播
            grad = criterion.backward()
            grads = model.backward(grad)

            # 参数更新
            trainer.step(grads)

        # 计算平均训练损失
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.training = False  # 启用评估模式
        val_preds = []
        val_losses = []
        
        for i in range(0, len(X_val), batch_size):
            batch_X = X_val[i:i + batch_size]
            batch_y = y_val[i:i + batch_size]
            
            output = model.forward(batch_X)
            val_loss = criterion.forward(output, batch_y)
            val_losses.append(val_loss)
            val_preds.extend(np.argmax(output, axis=1))

        acc = accuracy_score(y_val, val_preds)
        val_accs.append(acc)
        
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {np.mean(val_losses):.4f} | "
              f"Val Acc: {acc:.4f}")

        # 早停检查
        if acc > best_acc:
            best_acc = acc
            no_improve_epochs = 0
            # 保存最佳模型
            model.save_model(save_path)
            print(f"保存新的最佳模型，准确率: {acc:.4f}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        trainer.update_epoch()

    return model, train_losses, val_accs

def predict(model, image_path, img_size=(64, 64)):
    """使用模型进行预测"""
    try:
        # 加载和预处理图像
        img = Image.open(image_path).convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # 转换为CHW格式
        img_array = np.expand_dims(img_array, axis=0)  # 添加batch维度
        
        # 预测
        model.training = False
        output = model.forward(img_array)
        probs = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
        pred_class = np.argmax(probs, axis=1)[0]
        confidence = probs[0][pred_class]
        
        return pred_class, confidence
    except Exception as e:
        print(f"预测时出错: {str(e)}")
        return None, None

if __name__ == "__main__":
    # 训练模型
    model, train_losses, val_accs = train(save_path='best_model.npy')
    
    # 加载模型示例
    model = ResNet()
    model.load_model('best_model.npy')
    
    # 预测示例
    pred_class, confidence = predict(model, r'D:/DL_work/PetImages/Dog/560.jpg')
    if pred_class is not None:
        class_name = '猫' if pred_class == 0 else '狗'
        print(f"预测结果: {class_name}, 置信度: {confidence:.4f}")