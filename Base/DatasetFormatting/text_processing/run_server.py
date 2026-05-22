"""
GPT-SoVITS 文本处理服务启动脚本

启动文本处理的FastAPI服务器
"""

import os
import sys
import argparse
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GPT-SoVITS 文本处理服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8002, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="启用自动重载")
    parser.add_argument("--log-level", default="info", help="日志级别")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    
    args = parser.parse_args()
    
    print(f"🚀 启动GPT-SoVITS文本处理服务...")
    print(f"📍 地址: http://{args.host}:{args.port}")
    print(f"📖 API文档: http://{args.host}:{args.port}/docs")
    print(f"🔧 配置: reload={args.reload}, log_level={args.log_level}")
    
    try:
        import uvicorn
        
        uvicorn.run(
            "server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            workers=args.workers if not args.reload else 1
        )
        
    except ImportError:
        print("❌ 缺少uvicorn依赖，请安装: pip install uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 服务启动失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()