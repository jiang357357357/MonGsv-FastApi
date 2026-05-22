#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS SoVITS训练服务启动脚本

启动SoVITS训练的FastAPI服务
"""

import argparse
import uvicorn
from Code.FastApi.Base.Training.sovits_training.server import app


def main():
    parser = argparse.ArgumentParser(description="启动SoVITS训练API服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8006, help="服务器端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--reload", action="store_true", help="启用自动重载")
    parser.add_argument("--log-level", default="info", help="日志级别")
    
    args = parser.parse_args()
    
    print(f"启动SoVITS训练API服务...")
    print(f"地址: http://{args.host}:{args.port}")
    print(f"文档: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()