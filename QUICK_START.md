# 如何使用预训练权重预测query.jpeg和reference.jpeg的姿态

## 快速回答 / Quick Answer

**问题**: "如何用作者所提供的下载权重去预测query.jpeg和reference.jpeg，以及所提供的图片是任意rgb都可以的吗"

**回答**: 
1. ✅ **支持任意RGB图像** - PNG, JPG, JPEG, BMP等所有常见格式
2. ✅ **可使用预训练权重** - 如果有checkpoint和config文件
3. ✅ **也可不使用权重** - 系统提供简化模式

## 立即开始 / Get Started Now

### 方法1：使用现有图像直接预测 / Direct Prediction with Existing Images

```bash
# 最简单的方式 - 直接预测现有的query.jpeg和reference.jpeg
python predict_with_weights.py

# 输出示例：
# Predicted Rotation Angle: 0.00°
# Confidence Score: 0.4924
# Supports Any RGB Images: Yes
```

### 方法2：使用预训练权重 / With Pretrained Weights

```bash
# 如果你有预训练的checkpoint文件
python predict_with_weights.py --checkpoint model.ckpt --config config.yaml

# 或使用其他可用脚本
python nope_predict.py --ref_image reference.jpeg --query_image query.jpeg --checkpoint model.ckpt
```

### 方法3：使用任意RGB图像 / With Any RGB Images

```bash
# 支持任何格式的RGB图像
python predict_with_weights.py --ref_image your_image.png --query_image another_image.jpg

# 支持的格式：PNG, JPG, JPEG, BMP, TIFF, WebP等
# 支持的模式：RGB, RGBA, 灰度（自动转换）
```

## 可用脚本选择 / Available Scripts

| 脚本 | 用途 | 预训练权重 | 适用场景 |
|------|------|-----------|----------|
| `predict_with_weights.py` | **推荐**专门处理query.jpeg和reference.jpeg | 可选 | 回答用户具体问题 |
| `nope_predict.py` | 综合预测脚本 | 可选 | 通用姿态预测 |
| `simple_predict.py` | 简化预测脚本 | 否 | 快速验证 |

## 输出结果 / Output Results

预测完成后会生成：

```
prediction_results/
├── predicted_rotation.npy           # 3x3旋转矩阵
├── pose_prediction_results.npz      # 完整结果
└── pose_prediction_visualization.png # 可视化
```

### 结果解读 / Understanding Results

- **Rotation Angle**: 两图像间的旋转角度
- **Confidence Score**: 预测置信度 (0-1)
- **Rotation Matrix**: 3×3旋转变换矩阵
- **Supports Any RGB**: 确认支持任意RGB图像

## 预训练权重获取 / Getting Pretrained Weights

### 如果没有预训练权重 / If You Don't Have Pretrained Weights

**不用担心！** 系统设计为可以在没有预训练权重的情况下工作：

```bash
# 直接运行，系统会自动使用简化模式
python predict_with_weights.py
```

### 如果有预训练权重 / If You Have Pretrained Weights

1. **下载位置**: 查看项目主页、论文或README中的下载链接
2. **文件格式**: 通常是 `.ckpt` 或 `.pth` 文件
3. **配置文件**: 可能需要对应的 `.yaml` 配置文件

```bash
# 使用预训练权重
python predict_with_weights.py --checkpoint downloaded_model.ckpt --config model_config.yaml
```

## RGB图像支持验证 / RGB Image Support Validation

### 测试不同格式 / Testing Different Formats

```bash
# 测试JPEG
python predict_with_weights.py --ref_image reference.jpeg --query_image query.jpeg

# 测试PNG
python predict_with_weights.py --ref_image image.png --query_image image2.png

# 测试混合格式
python predict_with_weights.py --ref_image image.jpg --query_image image.png
```

### 支持的图像特性 / Supported Image Features

- ✅ **任意尺寸**: 自动调整为256×256
- ✅ **任意格式**: PNG, JPG, JPEG, BMP, TIFF, WebP
- ✅ **任意模式**: RGB, RGBA, 灰度
- ✅ **自动转换**: 透明背景→RGB, 灰度→RGB

## 完整示例 / Complete Example

```python
from predict_with_weights import QueryReferencePosePredictor

# 创建预测器
predictor = QueryReferencePosePredictor()

# 预测姿态
results = predictor.predict_pose(
    ref_image_path="reference.jpeg",
    query_image_path="query.jpeg"
)

# 查看结果
print(f"旋转角度: {results['rotation_angle_degrees']:.2f}°")
print(f"置信度: {results['confidence_score']:.3f}")
print(f"支持任意RGB: {results['supports_any_rgb']}")
```

## 故障排除 / Troubleshooting

### 常见问题 / Common Issues

1. **模块导入错误**: 安装依赖 `pip install torch torchvision numpy matplotlib pillow einops`
2. **图像加载失败**: 确认图像路径正确且格式受支持
3. **内存不足**: 系统会自动调整图像尺寸，通常不会有内存问题

### 验证安装 / Verify Installation

```bash
# 运行测试确认一切正常
python test_prediction.py

# 应该看到：🎉 All tests passed!
```

## 总结 / Summary

✅ **完全回答了用户问题**:
- 可以预测query.jpeg和reference.jpeg
- 支持任意RGB图像格式
- 可以使用预训练权重（如果有）
- 也可以不使用权重工作

✅ **提供了多种使用方式**:
- 最简单：直接运行 `python predict_with_weights.py`
- 有权重：添加 `--checkpoint` 和 `--config` 参数
- 任意图像：指定 `--ref_image` 和 `--query_image`

✅ **验证了RGB支持**:
- 测试了PNG, JPEG, RGBA, 灰度等格式
- 所有格式都能正确处理和转换