# Deprecated Environment Wrapper

本目录仅为旧导入和脚本路径保留兼容包装，不再包含环境探测实现。

新代码使用：

```python
from traning.lib.environment import collect_environment_report
```

GPU 检查的新入口是：

```bash
bash src/traning/lib/environment/check_gpu.sh
```
