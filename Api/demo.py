#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS API客户端演示脚本

展示如何使用API客户端进行完整的语音合成工作流
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from Code.FastApi.Api import (
    SyncGPTSoVITSClient, GPTSoVITSClient,
    get_request_builder, get_default_config,
    AudioUtils, FileUtils, ProgressTracker,
    create_temp_audio_file, format_duration
)


class GPTSoVITSDemo:
    """GPT-SoVITS API演示类"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.config = get_default_config()
        self.config.set("base_url", base_url)
        
        # 创建客户端和构建器
        self.sync_client = SyncGPTSoVITSClient(base_url=base_url)
        self.async_client = GPTSoVITSClient(base_url=base_url)
        self.builder = get_request_builder(self.config)
        
        # 临时文件列表
        self.temp_files = []
    
    def cleanup(self):
        """清理临时文件"""
        FileUtils.cleanup_temp_files(self.temp_files)
        print("✓ 临时文件清理完成")
    
    def demo_health_check(self):
        """演示健康检查"""
        print("=" * 60)
        print("🏥 健康检查演示")
        print("=" * 60)
        
        try:
            result = self.sync_client.health_check()
            print(f"✓ 服务健康状态: {result}")
            return True
        except Exception as e:
            print(f"✗ 健康检查失败: {e}")
            return False
    
    def demo_audio_processing(self):
        """演示音频处理功能"""
        print("\n" + "=" * 60)
        print("🎵 音频处理演示")
        print("=" * 60)
        
        try:
            # 创建测试音频
            print("📝 创建测试音频文件...")
            audio_file = create_temp_audio_file(duration=10.0, sample_rate=22050)
            self.temp_files.append(audio_file)
            
            # 获取音频信息
            info = AudioUtils.get_audio_info(audio_file)
            print(f"✓ 音频信息: 时长={info.get('duration', 0):.1f}秒, "
                  f"采样率={info.get('sample_rate', 0)}Hz")
            
            # 音频切分
            print("\n🔪 执行音频切分...")
            output_dir = tempfile.mkdtemp()
            slice_request = self.builder.audio_slice_request(
                input_path=audio_file,
                output_dir=output_dir,
                threshold=-30.0,
                min_length=2000
            )
            
            slice_response = self.sync_client.audio_slice(slice_request)
            print(f"✓ 音频切分完成:")
            print(f"  - 输出目录: {slice_response.output_dir}")
            print(f"  - 处理文件数: {len(slice_response.processed_files)}")
            print(f"  - 输出文件数: {len(slice_response.output_files)}")
            
            return True
            
        except Exception as e:
            print(f"✗ 音频处理失败: {e}")
            return False
    
    async def demo_async_workflow(self):
        """演示异步工作流"""
        print("\n" + "=" * 60)
        print("⚡ 异步工作流演示")
        print("=" * 60)
        
        try:
            async with self.async_client as client:
                # 并发执行多个健康检查
                print("🔄 执行并发健康检查...")
                tasks = [client.health_check() for _ in range(3)]
                results = await asyncio.gather(*tasks)
                print(f"✓ 并发请求完成，结果数量: {len(results)}")
                
                # 异步推理演示
                print("\n🎯 异步推理演示...")
                
                # 创建参考音频
                ref_audio = create_temp_audio_file(duration=3.0)
                self.temp_files.append(ref_audio)
                
                inference_request = self.builder.inference_request(
                    text="你好，这是异步推理测试。",
                    ref_audio_path=ref_audio,
                    prompt_text="参考音频文本",
                    text_language="zh",
                    return_base64=True
                )
                
                response = await client.inference(inference_request)
                print(f"✓ 异步推理完成:")
                print(f"  - 音频数据长度: {len(response.audio_data) if response.audio_data else 0}")
                print(f"  - 处理时间: {response.processing_time:.2f}秒")
                
                return True
                
        except Exception as e:
            print(f"✗ 异步工作流失败: {e}")
            return False
    
    def demo_complete_pipeline(self):
        """演示完整数据处理管道"""
        print("\n" + "=" * 60)
        print("🏭 完整数据处理管道演示")
        print("=" * 60)
        
        tracker = ProgressTracker(total_steps=4)
        
        try:
            # 步骤1: 准备测试数据
            tracker.update(1, "准备测试数据...")
            
            # 创建测试音频目录
            input_dir = tempfile.mkdtemp()
            for i in range(3):
                audio_file = create_temp_audio_file(duration=5.0)
                import shutil
                shutil.copy(audio_file, os.path.join(input_dir, f"test_{i}.wav"))
                self.temp_files.append(audio_file)
            
            print(f"✓ 创建了3个测试音频文件")
            
            # 步骤2: ASR识别
            tracker.update(2, "执行ASR语音识别...")
            
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.list').name
            self.temp_files.append(output_file)
            
            asr_request = self.builder.asr_request(
                input_path=input_dir,
                output_file=output_file,
                model_type="funasr",
                language="zh"
            )
            
            asr_response = self.sync_client.asr_recognize(asr_request)
            print(f"✓ ASR识别完成，处理了 {len(asr_response.processed_files)} 个文件")
            
            # 步骤3: 文本特征提取
            tracker.update(3, "提取文本特征...")
            
            text_output_dir = tempfile.mkdtemp()
            text_request = self.builder.text_processing_request(
                list_file=asr_response.output_file,
                output_dir=text_output_dir,
                language="zh"
            )
            
            text_response = self.sync_client.text_processing(text_request)
            print(f"✓ 文本特征提取完成")
            
            # 步骤4: 音频特征提取
            tracker.update(4, "提取音频特征...")
            
            audio_output_dir = tempfile.mkdtemp()
            audio_request = self.builder.audio_features_request(
                list_file=asr_response.output_file,
                output_dir=audio_output_dir,
                version="v2Pro"
            )
            
            audio_response = self.sync_client.audio_features(audio_request)
            print(f"✓ 音频特征提取完成")
            
            tracker.finish("完整数据处理管道执行完成！")
            return True
            
        except Exception as e:
            print(f"✗ 完整管道执行失败: {e}")
            return False
    
    def demo_training_workflow(self):
        """演示训练工作流"""
        print("\n" + "=" * 60)
        print("🎓 训练工作流演示")
        print("=" * 60)
        
        try:
            # SoVITS训练演示
            print("🚀 启动SoVITS训练...")
            
            exp_root = tempfile.mkdtemp()
            sovits_request = self.builder.sovits_training_request(
                exp_name="demo_sovits",
                exp_root=exp_root,
                version="v2Pro",
                batch_size=16,
                total_epoch=2,  # 演示用，设置较小的轮数
                learning_rate=0.0001
            )
            
            sovits_response = self.sync_client.start_sovits_training(sovits_request)
            print(f"✓ SoVITS训练启动成功:")
            print(f"  - 任务ID: {sovits_response.job_id}")
            print(f"  - 实验名称: {sovits_response.exp_name}")
            print(f"  - 状态: {sovits_response.status}")
            
            # 检查训练状态
            if sovits_response.job_id:
                print("\n📊 检查训练状态...")
                status = self.sync_client.get_training_status(sovits_response.job_id)
                print(f"✓ 当前训练状态: {status.status}")
                if status.progress:
                    print(f"  - 进度: {status.progress:.1f}%")
                if status.current_epoch:
                    print(f"  - 当前轮次: {status.current_epoch}/{status.total_epochs}")
            
            return True
            
        except Exception as e:
            print(f"✗ 训练工作流失败: {e}")
            return False
    
    def demo_inference_modes(self):
        """演示不同推理模式"""
        print("\n" + "=" * 60)
        print("🎤 推理模式演示")
        print("=" * 60)
        
        try:
            # 创建参考音频
            ref_audio = create_temp_audio_file(duration=3.0)
            self.temp_files.append(ref_audio)
            
            # 模式1: 文件路径推理
            print("📁 模式1: 使用音频文件路径推理...")
            
            inference_request1 = self.builder.inference_request(
                text="这是使用文件路径的推理测试。",
                ref_audio_path=ref_audio,
                prompt_text="参考音频的文本内容",
                text_language="zh",
                prompt_language="zh",
                return_base64=False
            )
            
            response1 = self.sync_client.inference(inference_request1)
            print(f"✓ 文件路径推理完成，输出路径: {response1.audio_path}")
            
            # 模式2: Base64推理
            print("\n🔢 模式2: 使用Base64音频推理...")
            
            base64_audio = AudioUtils.encode_audio_file(ref_audio)
            inference_request2 = self.builder.inference_request(
                text="这是使用Base64音频的推理测试。",
                ref_audio_base64=base64_audio,
                prompt_text="参考音频的文本内容",
                return_base64=True
            )
            
            async def _async_inference():
                async with self.async_client as client:
                    return await client.inference_with_base64_audio(inference_request2)
            
            response2 = asyncio.run(_async_inference())
            print(f"✓ Base64推理完成，音频数据长度: {len(response2.audio_data)}")
            
            # 保存Base64音频到文件
            if response2.audio_data:
                output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
                self.temp_files.append(output_file)
                AudioUtils.decode_audio_base64(response2.audio_data, output_file)
                print(f"✓ Base64音频已保存到: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"✗ 推理模式演示失败: {e}")
            return False
    
    def demo_batch_processing(self):
        """演示批量处理"""
        print("\n" + "=" * 60)
        print("📦 批量处理演示")
        print("=" * 60)
        
        try:
            from Code.FastApi.Api.models import BatchProject, BatchRequest
            
            # 创建多个测试项目
            projects = []
            for i in range(2):
                # 为每个项目创建测试音频目录
                project_input = tempfile.mkdtemp()
                project_output = tempfile.mkdtemp()
                
                # 创建测试音频文件
                for j in range(2):
                    audio_file = create_temp_audio_file(duration=3.0)
                    import shutil
                    shutil.copy(audio_file, os.path.join(project_input, f"audio_{j}.wav"))
                    self.temp_files.append(audio_file)
                
                project = BatchProject(
                    name=f"demo_project_{i}",
                    input_dir=project_input,
                    output_dir=project_output,
                    language="zh",
                    version="v2Pro"
                )
                projects.append(project)
            
            # 创建批量请求
            batch_request = BatchRequest(
                projects=projects,
                max_concurrent=2
            )
            
            print(f"🚀 启动批量处理，项目数量: {len(projects)}")
            batch_response = self.sync_client.batch_process(batch_request)
            
            print(f"✓ 批量处理启动成功:")
            print(f"  - 批次ID: {batch_response.batch_id}")
            print(f"  - 总项目数: {batch_response.total_projects}")
            
            return True
            
        except Exception as e:
            print(f"✗ 批量处理演示失败: {e}")
            return False
    
    def run_all_demos(self):
        """运行所有演示"""
        print("🎬 GPT-SoVITS API客户端完整演示")
        print("=" * 80)
        
        demos = [
            ("健康检查", self.demo_health_check),
            ("音频处理", self.demo_audio_processing),
            ("异步工作流", lambda: asyncio.run(self.demo_async_workflow())),
            ("完整数据管道", self.demo_complete_pipeline),
            ("训练工作流", self.demo_training_workflow),
            ("推理模式", self.demo_inference_modes),
            ("批量处理", self.demo_batch_processing),
        ]
        
        results = {}
        passed = 0
        
        for demo_name, demo_func in demos:
            print(f"\n🎯 开始演示: {demo_name}")
            try:
                result = demo_func()
                results[demo_name] = result
                if result:
                    passed += 1
                    print(f"✅ {demo_name} 演示成功")
                else:
                    print(f"❌ {demo_name} 演示失败")
            except Exception as e:
                print(f"💥 {demo_name} 演示异常: {e}")
                results[demo_name] = False
        
        # 清理临时文件
        self.cleanup()
        
        # 输出总结
        print("\n" + "=" * 80)
        print("📊 演示结果总结")
        print("=" * 80)
        
        for demo_name, result in results.items():
            status = "✅ 成功" if result else "❌ 失败"
            print(f"{demo_name:<20} {status}")
        
        print(f"\n🎉 总计: {passed}/{len(demos)} 个演示成功")
        print(f"📈 成功率: {passed/len(demos)*100:.1f}%")
        
        return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-SoVITS API客户端演示")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="API服务地址 (默认: http://localhost:8000)")
    parser.add_argument("--demo", choices=[
        "health", "audio", "async", "pipeline", 
        "training", "inference", "batch", "all"
    ], default="all", help="要运行的演示类型")
    
    args = parser.parse_args()
    
    # 创建演示实例
    demo = GPTSoVITSDemo(base_url=args.url)
    
    print(f"🌐 连接到服务: {args.url}")
    
    # 根据参数运行对应演示
    if args.demo == "all":
        demo.run_all_demos()
    elif args.demo == "health":
        demo.demo_health_check()
    elif args.demo == "audio":
        demo.demo_audio_processing()
    elif args.demo == "async":
        asyncio.run(demo.demo_async_workflow())
    elif args.demo == "pipeline":
        demo.demo_complete_pipeline()
    elif args.demo == "training":
        demo.demo_training_workflow()
    elif args.demo == "inference":
        demo.demo_inference_modes()
    elif args.demo == "batch":
        demo.demo_batch_processing()
    
    demo.cleanup()


if __name__ == "__main__":
    main()