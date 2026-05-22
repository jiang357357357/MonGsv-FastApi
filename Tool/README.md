# FastApi Tool

用于 `Code/FastApi` 主网关的运维脚本。

当前提供：

- `status_gateway.ps1`
  - 查看 `.monconfig` 里的网关端口
  - 查看该端口是否被占用
  - 输出占用进程信息

- `stop_gateway.ps1`
  - 按 `.monconfig` 端口停止当前网关进程
  - 可用 `-Port` 指定端口
  - 可用 `-Force` 跳过确认

- `cleanup_gateway.ps1`
  - 停止网关进程
  - 清理 `.monconfig` 指定的 `TEMP_DIR`
  - 可用 `-KeepTemp` 仅停止进程不清理临时目录
  - 可用 `-Force` 跳过确认

## 用法

```powershell
pwsh ./Code/FastApi/Tool/status_gateway.ps1
pwsh ./Code/FastApi/Tool/stop_gateway.ps1
pwsh ./Code/FastApi/Tool/cleanup_gateway.ps1
```

Windows PowerShell 也可直接运行：

```powershell
.\Code\FastApi\Tool\status_gateway.ps1
.\Code\FastApi\Tool\stop_gateway.ps1
.\Code\FastApi\Tool\cleanup_gateway.ps1
```
