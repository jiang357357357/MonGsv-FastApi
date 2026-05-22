#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR语音识别服务启动脚本

专门用于启动asr_recognition模块的服务
"""

import uvicorn
from .server import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )