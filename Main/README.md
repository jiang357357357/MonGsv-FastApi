# FastApi Main

主运行目录。

当前提供的统一入口：

- `run_gateway.py`

启动方式：

```bash
python FastApi/Main/run_gateway.py
```

说明：

- 实际服务实现仍在 `FastApi/Base`
- `Main` 只负责主运行入口聚合
- 后续如果增加 `run_api.py`、`run_worker.py`、`run_dev.py`，也统一放在这里
