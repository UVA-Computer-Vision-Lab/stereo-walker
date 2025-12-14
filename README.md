# Stereo Walker Test Environment

这个仓库包含了运行 `test.py` 所需的所有代码文件。

## 目录结构

- `test.py` - 测试脚本
- `config/` - 配置文件目录
- `pl_modules/` - PyTorch Lightning 模块
- `model/` - 模型定义
- `data/` - 数据集类定义
- `tmp/` - Checkpoint 文件目录
- `dataset/teleop/` - 数据目录（软连接到原仓库）
- `MonSter-plusplus/` - Git submodule
- `Depth-Anything-V2/` - 外部依赖（软连接）
- `FoundationStereo/` - 外部依赖（软连接）

## 使用方法

```bash
python test.py --config config/teleop_eval.yaml --checkpoint tmp/stereowalker.ckpt
```

## 数据路径

数据使用软连接指向原仓库：
- `dataset/teleop/` -> `/bigtemp/tsx4zn/StereoWalker/dataset/teleop`

## 外部模块

- `MonSter-plusplus` - 使用 git submodule 管理
- `Depth-Anything-V2` - 软连接到原仓库
- `FoundationStereo` - 软连接到原仓库

## 初始化 Submodule

如果克隆此仓库，需要初始化 submodule：

```bash
git submodule update --init --recursive
```


