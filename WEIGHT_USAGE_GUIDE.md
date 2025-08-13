# 使用预训练权重预测query.jpeg和reference.jpeg的姿态 / Using Pretrained Weights for Pose Prediction

## 问题回答 / Question Answered

**问题**: "如何用作者所提供的下载权重去预测query.jpeg和reference.jpeg，以及所提供的图片是任意rgb都可以的吗"

**Answer**: "How to use the author's provided download weights to predict query.jpeg and reference.jpeg, and can the provided images be any RGB images?"

### 回答 / Answer

✅ **是的，支持任意RGB图像！/ Yes, any RGB images are supported!**

本系统完全支持任意RGB图像格式，包括但不限于：
- PNG, JPG, JPEG, BMP, TIFF等常见格式
- 任意尺寸（会自动调整）
- 自动格式转换（RGBA -> RGB, 灰度 -> RGB等）

## 使用方法 / Usage

### 1. 使用预训练权重 / With Pretrained Weights

```bash
# 如果你有预训练的checkpoint和config文件
python predict_with_weights.py --checkpoint path/to/model.ckpt --config path/to/config.yaml

# 使用自定义图像
python predict_with_weights.py --checkpoint model.ckpt --config config.yaml --ref_image your_ref.jpg --query_image your_query.png
```

### 2. 不使用预训练权重（简化模式）/ Without Pretrained Weights (Simplified Mode)

```bash
# 直接预测现有的query.jpeg和reference.jpeg
python predict_with_weights.py

# 使用自定义图像
python predict_with_weights.py --ref_image your_ref.jpg --query_image your_query.png
```

### 3. 其他可用脚本 / Other Available Scripts

```bash
# 简单预测脚本
python simple_predict.py --ref_image reference.jpeg --query_image query.jpeg

# 综合预测脚本
python nope_predict.py --ref_image reference.jpeg --query_image query.jpeg

# 自动创建测试图像并预测
python nope_predict.py --create_test
```

## 预训练权重获取 / Getting Pretrained Weights

### 选项1：下载预训练模型 / Option 1: Download Pretrained Models

```bash
# 如果作者提供了预训练模型下载链接
# If the authors provided download links for pretrained models
python -m src.scripts.download_pretrained_models  # (如果可用 / if available)
```

### 选项2：从HuggingFace下载 / Option 2: Download from HuggingFace

```bash
# 如果模型托管在HuggingFace上
# If models are hosted on HuggingFace
# 参考README.md中的说明
```

### 选项3：手动下载 / Option 3: Manual Download

1. 查看项目主页或论文是否提供下载链接
2. 下载checkpoint文件（通常是.ckpt或.pth格式）
3. 下载对应的config文件（通常是.yaml格式）

## 输出文件 / Output Files

运行预测后，会在输出目录生成以下文件：

```
prediction_results/
├── predicted_rotation.npy           # 3x3旋转矩阵
├── pose_prediction_results.npz      # 完整预测结果
└── pose_prediction_visualization.png # 可视化结果
```

### 结果解释 / Results Explanation

- **predicted_rotation.npy**: 3×3旋转矩阵，表示从reference到query的姿态变换
- **rotation_angle_degrees**: 旋转角度（度）
- **confidence_score**: 置信度分数（0-1）
- **supports_any_rgb**: 确认支持任意RGB图像

## RGB图像支持详情 / RGB Image Support Details

### ✅ 支持的格式 / Supported Formats

- **图像格式**: PNG, JPG, JPEG, BMP, TIFF, WebP等
- **颜色模式**: RGB, RGBA, 灰度图像（自动转换为RGB）
- **图像尺寸**: 任意尺寸（自动调整为256×256用于处理）
- **图像质量**: 任意质量和压缩级别

### 🔄 自动处理 / Automatic Processing

系统会自动：
1. **格式转换**: RGBA → RGB, 灰度 → RGB
2. **尺寸调整**: 任意尺寸 → 256×256
3. **数值归一化**: [0,255] → [-1,1]
4. **批处理**: 添加batch维度用于模型处理

### 📝 示例代码 / Example Code

```python
from predict_with_weights import QueryReferencePosePredictor

# 初始化预测器
predictor = QueryReferencePosePredictor(
    checkpoint_path="model.ckpt",  # 可选
    config_path="config.yaml"      # 可选
)

# 预测任意RGB图像
results = predictor.predict_pose(
    ref_image_path="any_reference_image.png",
    query_image_path="any_query_image.jpg"
)

print(f"Rotation angle: {results['rotation_angle_degrees']:.2f}°")
print(f"Confidence: {results['confidence_score']:.3f}")
print(f"Supports any RGB: {results['supports_any_rgb']}")
```

## 常见问题 / FAQ

### Q1: 没有预训练权重怎么办？/ What if I don't have pretrained weights?

**A**: 系统会自动切换到简化模式，使用特征相似度进行姿态估计。虽然精度可能不如预训练模型，但仍能提供有用的结果。

### Q2: 图像尺寸很大会影响性能吗？/ Will large images affect performance?

**A**: 不会。系统会自动将图像调整为256×256像素进行处理，原始尺寸不影响性能。

### Q3: 支持非方形图像吗？/ Are non-square images supported?

**A**: 是的。系统会保持图像内容完整地调整为方形，不会产生显著变形。

### Q4: 可以处理透明背景的PNG图像吗？/ Can it handle PNG images with transparency?

**A**: 可以。透明通道会被自动处理，转换为标准RGB格式。

## 性能优化 / Performance Optimization

### 使用GPU加速 / GPU Acceleration

```python
# 系统会自动检测并使用可用的GPU
print(f"Using device: {torch.cuda.is_available() and 'CUDA' or 'CPU'}")
```

### 批量处理 / Batch Processing

```python
# 如需处理多对图像，可以循环调用
for ref_path, query_path in image_pairs:
    results = predictor.predict_pose(ref_path, query_path)
```

## 验证结果 / Validating Results

### 运行测试 / Run Tests

```bash
# 运行完整测试套件
python test_prediction.py

# 快速验证
python predict_with_weights.py --ref_image reference.jpeg --query_image query.jpeg
```

### 检查输出 / Check Output

预测成功后应该看到：
- 旋转矩阵（3×3）
- 旋转角度
- 置信度分数
- 可视化图像

## 技术细节 / Technical Details

### 模板匹配 / Template Matching

系统使用180个模板覆盖不同视角：
- 方位角: 0°-360°（每10°一个点，36个）
- 仰角: -45°, -22.5°, 0°, 22.5°, 45°（5个）
- 总计: 36 × 5 = 180个模板

### 特征提取 / Feature Extraction

- **预训练模式**: 使用NOPE训练的U-Net编码器
- **简化模式**: 多尺度全局和局部特征

### 相似度计算 / Similarity Computation

使用余弦相似度计算query特征与各模板特征的匹配度。

---

## 总结 / Summary

✅ **完全支持任意RGB图像格式**  
✅ **可使用预训练权重或简化模式**  
✅ **自动处理图像格式和尺寸**  
✅ **提供详细的预测结果和可视化**  

无论是否有预训练权重，都可以对query.jpeg和reference.jpeg（或任何其他RGB图像）进行姿态预测。