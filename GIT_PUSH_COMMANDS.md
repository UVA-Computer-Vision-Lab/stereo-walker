# Git Push 命令

## 当前仓库信息
- 远程仓库: `https://github.com/UVA-Computer-Vision-Lab/stereo-walker.git`
- Submodules:
  - `MonSter-plusplus`: `https://github.com/Junda24/MonSter-plusplus.git`
  - `Depth-Anything-V2`: `https://github.com/Smirkkkk/Depth-Anything-V2.git`

## 推送步骤

### 1. 添加所有更改到暂存区
```bash
cd /bigtemp/tsx4zn/stereo-walker
git add .
```

### 2. 提交更改
```bash
git commit -m "Add stereo-walker test environment with submodules"
```

### 3. 推送主仓库
```bash
# 如果这是第一次推送，需要设置上游分支
git push -u origin main
# 或者如果分支名是 master
git push -u origin master

# 如果已经设置过上游分支，直接推送
git push
```

### 4. 推送 submodules（如果需要）
```bash
# 推送 MonSter-plusplus submodule
cd MonSter-plusplus
git push origin main  # 或 master，取决于该仓库的分支名
cd ..

# 推送 Depth-Anything-V2 submodule
cd Depth-Anything-V2
git push origin main  # 或 master，取决于该仓库的分支名
cd ..
```

## 完整推送命令（一键执行）

```bash
cd /bigtemp/tsx4zn/stereo-walker

# 添加并提交所有更改
git add .
git commit -m "Add stereo-walker test environment with submodules"

# 推送主仓库（根据实际分支名调整）
git push -u origin main
# 或
git push -u origin master
```

## 注意事项

1. **首次推送**: 如果远程仓库还没有创建，需要先在 GitHub 上创建仓库
2. **分支名**: 根据实际情况使用 `main` 或 `master`
3. **Submodules**: Submodules 的更改需要在其各自的仓库中提交和推送
4. **权限**: 确保您有推送权限到远程仓库

## 克隆此仓库后初始化 submodules

其他人克隆仓库后，需要运行：
```bash
git submodule update --init --recursive
```


