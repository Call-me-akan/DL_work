
# Cats vs Dogs Classification with ResNet  

## 项目描述  
基于残差网络（ResNet）的猫狗图像二分类项目，使用Kaggle公开数据集实现图像分类任务，验证深层神经网络在中小规模数据集上的有效性。通过数据增强、超参数调优和轻量化模型设计，实现高效的二分类性能。  


## 数据集  
### 数据集来源  
- **数据集链接**：[Microsoft Cats vs Dogs Dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset)  
- **数据集说明**：包含猫和狗的RGB图像，适用于二分类任务。原始数据集包含约25,000张训练图像和12,500张测试图像，本项目使用前1000张/类别作为训练数据（共2000张原始图像，通过数据增强扩展至8000张）。  


## 代码结构  
```  
├── Whole.py          # all 
 
└── README.md           # 项目说明文件  
```  


## 如何运行  
### 1. 环境要求  
- Python 3.8+  
- 依赖库：  
  ```bash  
  pip install numpy pillow matplotlib scikit-learn  
  ```  

### 2. 数据集下载  
1. 访问Kaggle数据集链接：[https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset)  
2. 点击“Download”获取数据集，解压后放入项目目录下的`PetImages`文件夹（目录结构：`PetImages/Cat/`, `PetImages/Dog/`）。  

### 3. 运行代码  
```bash  
# 训练模型  
python train.py  

# 预测示例（使用best_model.pkl）  
# 在predict函数中输入图像路径，输出类别（0=猫，1=狗）及置信度  
```  


## 核心技术  
1. **轻量化ResNet设计**：  
   - 自定义残差块（Residual Block），通过短路连接解决梯度消失问题。  
   - 全局平均池化替代全连接层，减少参数量，适配CPU训练。  
2. **数据增强**：  
   - 水平翻转、随机旋转（±15°）、亮度/对比度调整，提升数据多样性。  
3. **训练优化**：  
   - 动量SGD优化器，学习率衰减策略，早停法防止过拟合。  


## 结果  
- **测试集准确率**：93.2%  
- **评估指标**：精确率93.5%，召回率92.8%，F1分数93.1%（详见训练日志及图表）。  
- **训练曲线**：包含损失曲线和准确率曲线，展示训练集/验证集性能变化（文件：`loss_curve.png`, `accuracy_curve.png`）。  


## 贡献  
欢迎提交Issue反馈问题或建议，如需改进模型或扩展功能，可提交Pull Request。  

  
**作者**：李家辉  
**日期**：2025年5月  
**联系**：Call_me_akan@outlook.com

  
![GitHub Stars](https://img.shields.io/github/stars/your-username/your-repo?style=flat-square)  
![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)  
