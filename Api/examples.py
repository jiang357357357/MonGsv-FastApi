#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS API使用示例

展示如何使用API客户端进行各种操作
"""

import asyncio
import tempfile
import os
from pathlib import Path

from .client import GPTSoVITSClient, SyncGPTSoVITSClient
from .models import *


# ==================== 异步示例 ====================

async def example_health_check():
    """健康检查示例"""
    print("=== 健康检查示例 ===")
    
    async with GPTSoVITSClient() as client:
        try:
            health = await client.health_check()
            print(f"服务状态: {health}")
            
            services = await client.get_services_status()
            print(f"各服务状态: {services}")
            
        except Exception as e:
            print(f"健康检查失败: {e}")


async def example_audio_slice():
    """音频切分示例"""
    print("\n=== 音频切分示例 ===")
    
    async with GPTSoVITSClient() as client:
        try:
            # 创建请求
            config = AudioSliceConfig(
                threshold=-30.0,
                min_length=3000,
                min_interval=200
            )
            
            request = AudioSliceRequest(
                input_path="/path/to/input/audio.wav",
                output_dir="/path/to/output/sliced",
                config=config,
                timeout=600
            )
            
            # 执行切分
            response = await client.audio_slice(request)
            
            if response.success:
                print(f"切分成功!")
                print(f"输出目录: {response.output_dir}")
                print(f"处理文件数: {len(response.processed_files)}")
                print(f"输出文件数: {len(response.output_files)}")
                print(f"处理时间: {response.processing_time}s")
            else:
                print(f"切分失败: {response.message}")
                
        except Exception as e:
            print(f"音频切分异常: {e}")


async def example_asr_recognition():
    """ASR识别示例"""
    print("\n=== ASR识别示例 ===")
    
    async with GPTSoVITSClient() as client:
        try:
            # 创建请求
            config = ASRConfig(
                model_type="funasr",
                language="zh",
                precision="float32",
                batch_size=1
            )
            
            request = ASRRequest(
                input_path="/path/to/sliced/audio",
                output_file="/path/to/output/annotations.list",
                config=config,
                timeout=1800
            )
            
            # 执行识别
            response = await client.asr_recognize(request)
            
            if response.success:
                print(f"识别成功!")
                print(f"输出文件: {response.output_file}")
                print(f"识别结果数: {len(response.recognition_results)}")
                print(f"总时长: {response.total_duration}s")
                print(f"处理时间: {response.processing_time}s")
            else:
                print(f"识别失败: {response.message}")
                
        except Exception as e:
            print(f"ASR识别异常: {e}")


async def example_complete_workflow():
    """完整工作流示例"""
    print("\n=== 完整工作流示例 ===")
    
    async with GPTSoVITSClient() as client:
        try:
            # 创建请求
            config = WorkflowConfig(
                language="zh",
                version="v2Pro",
                skip_existing=True,
                parallel_processing=True
            )
            
            request = WorkflowRequest(
                project_name="my_voice_project",
                input_audio_dir="/path/to/raw/audio",
                output_dir="/path/to/project/output",
                config=config,
                timeout=3600
            )
            
            # 执行工作流
            response = await client.complete_workflow(request)
            
            if response.success:
                print(f"工作流启动成功!")
                print(f"项目名称: {response.project_name}")
                print(f"工作流ID: {response.workflow_id}")
                print(f"当前步骤: {response.current_step}")
                print(f"总体进度: {response.overall_progress}%")
                
                # 显示步骤状态
                for step in response.steps:
                    print(f"  步骤 {step.step_name}: {step.status}")
            else:
                print(f"工作流失败: {response.message}")
                
        except Exception as e:
            print(f"工作流异常: {e}")


async def example_training():
    """训练示例"""
    print("\n=== 训练示例 ===")
    
    async with GPTSoVITSClient() as client:
        try:
            # GPT训练
            gpt_config = GPTTrainingConfig(
                batch_size=8,
                total_epoch=15,
                learning_rate=0.01,
                save_every_epoch=5,
                gpu_numbers="0"
            )
            
            gpt_request = GPTTrainingRequest(
                exp_name="my_voice_project",
                exp_root="/path/to/experiments",
                config=gpt_config,
                timeout=7200
            )
            
            gpt_response = await client.start_gpt_training(gpt_request)
            
            if gpt_response.success:
                print(f"GPT训练启动成功!")
                print(f"任务ID: {gpt_response.job_id}")
                print(f"日志文件: {gpt_response.log_file}")
                
                # 等待训练完成
                print("等待GPT训练完成...")
                final_status = await client.wait_for_training_completion(
                    gpt_response.job_id, check_interval=60, max_wait_time=7200
                )
                
                if final_status.status == "completed":
                    print("GPT训练完成!")
                    
                    # 开始SoVITS训练
                    sovits_config = SoVITSTrainingConfig(
                        version="v2Pro",
                        batch_size=32,
                        total_epoch=8,
                        learning_rate=0.0001,
                        save_every_epoch=4,
                        gpu_numbers="0"
                    )
                    
                    sovits_request = SoVITSTrainingRequest(
                        exp_name="my_voice_project",
                        exp_root="/path/to/experiments",
                        config=sovits_config,
                        timeout=3600
                    )
                    
                    sovits_response = await client.start_sovits_training(sovits_request)
                    
                    if sovits_response.success:
                        print(f"SoVITS训练启动成功!")
                        print(f"任务ID: {sovits_response.job_id}")
                        
                        # 等待训练完成
                        print("等待SoVITS训练完成...")
                        final_status = await client.wait_for_training_completion(
                            sovits_response.job_id, check_interval=60, max_wait_time=3600
                        )
                        
                        if final_status.status == "completed":
                            print("SoVITS训练完成! 可以开始推理了。")
                        else:
                            print(f"SoVITS训练失败: {final_status.error_message}")
                    else:
                        print(f"SoVITS训练启动失败: {sovits_response.message}")
                else:
                    print(f"GPT训练失败: {final_status.error_message}")
            else:
                print(f"GPT训练启动失败: {gpt_response.message}")
                
        except Exception as e:
            print(f"训练异常: {e}")


async def example_inference():
    """推理示例"""
    print("\n=== 推理示例 ===")
    
    async with GPTSoVITSClient() as client:
        try:
            # 创建推理配置
            config = InferenceConfig(
                top_k=20,
                top_p=0.6,
                temperature=0.6,
                how_to_cut="不切",
                speed=1.0
            )
            
            # 使用文件路径的推理
            request = InferenceRequest(
                text="你好，欢迎使用GPT-SoVITS语音合成系统！这是一个测试文本。",
                text_language="zh",
                ref_audio_path="/path/to/reference.wav",
                prompt_text="参考音频的文本内容",
                prompt_language="zh",
                config=config,
                output_format="wav",
                return_base64=True,
                timeout=300
            )
            
            response = await client.inference(request)
            
            if response.success:
                print(f"推理成功!")
                print(f"音频时长: {response.duration}s")
                print(f"采样率: {response.sample_rate}Hz")
                print(f"处理时间: {response.processing_time}s")
                print(f"文本片段数: {len(response.text_segments) if response.text_segments else 0}")
                
                if response.audio_data:
                    # 保存Base64音频
                    output_path = "/path/to/output/generated.wav"
                    client.save_base64_audio(response.audio_data, output_path)
                    print(f"音频已保存到: {output_path}")
                elif response.audio_path:
                    print(f"音频文件: {response.audio_path}")
            else:
                print(f"推理失败: {response.message}")
                
        except Exception as e:
            print(f"推理异常: {e}")


async def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    async with GPTSoVITSClient() as client:
        try:
            # 创建批量项目
            projects = [
                BatchProject(
                    name="voice_1",
                    input_dir="/path/to/voice1/audio",
                    output_dir="/path/to/voice1/output",
                    language="zh",
                    version="v2Pro"
                ),
                BatchProject(
                    name="voice_2", 
                    input_dir="/path/to/voice2/audio",
                    output_dir="/path/to/voice2/output",
                    language="en",
                    version="v2Pro"
                ),
                BatchProject(
                    name="voice_3",
                    input_dir="/path/to/voice3/audio", 
                    output_dir="/path/to/voice3/output",
                    language="ja",
                    version="v2Pro"
                )
            ]
            
            request = BatchRequest(
                projects=projects,
                max_concurrent=2,
                timeout=7200
            )
            
            response = await client.batch_process(request)
            
            if response.success:
                print(f"批量处理启动成功!")
                print(f"批次ID: {response.batch_id}")
                print(f"总项目数: {response.total_projects}")
                print(f"已完成: {response.completed_projects}")
                print(f"失败数: {response.failed_projects}")
                
                # 显示各项目结果
                for result in response.project_results:
                    print(f"  项目 {result.get('project')}: {result.get('status', 'unknown')}")
            else:
                print(f"批量处理失败: {response.message}")
                
        except Exception as e:
            print(f"批量处理异常: {e}")


# ==================== 同步示例 ====================

def sync_example_basic_usage():
    """同步客户端基础使用示例"""
    print("\n=== 同步客户端示例 ===")
    
    client = SyncGPTSoVITSClient()
    
    try:
        # 健康检查
        health = client.health_check()
        print(f"服务状态: {health}")
        
        # 音频切分
        slice_config = AudioSliceConfig(threshold=-30.0)
        slice_request = AudioSliceRequest(
            input_path="/path/to/input.wav",
            output_dir="/path/to/output",
            config=slice_config
        )
        
        slice_response = client.audio_slice(slice_request)
        print(f"切分结果: {slice_response.success}")
        
        # 推理
        inference_config = InferenceConfig(temperature=0.7)
        inference_request = InferenceRequest(
            text="这是一个测试文本",
            text_language="zh",
            ref_audio_path="/path/to/reference.wav",
            config=inference_config,
            return_base64=True
        )
        
        inference_response = client.inference(inference_request)
        print(f"推理结果: {inference_response.success}")
        
    except Exception as e:
        print(f"同步客户端异常: {e}")


# ==================== 实用工具示例 ====================

async def example_audio_utilities():
    """音频工具示例"""
    print("\n=== 音频工具示例 ===")
    
    client = GPTSoVITSClient()
    
    # Base64编码音频文件
    audio_file = "/path/to/audio.wav"
    if os.path.exists(audio_file):
        base64_data = client.encode_audio_file(audio_file)
        print(f"音频文件已编码为Base64，长度: {len(base64_data)}")
        
        # 使用Base64进行推理
        config = InferenceConfig()
        request = InferenceRequest(
            text="使用Base64音频进行推理测试",
            text_language="zh",
            ref_audio_base64=base64_data,
            config=config,
            return_base64=True
        )
        
        async with client:
            response = await client.inference_with_base64_audio(request)
            
            if response.success and response.audio_data:
                # 保存生成的音频
                output_file = "/path/to/generated.wav"
                client.save_base64_audio(response.audio_data, output_file)
                print(f"生成的音频已保存到: {output_file}")


# ==================== 主函数 ====================

async def run_all_examples():
    """运行所有异步示例"""
    print("🚀 GPT-SoVITS API客户端使用示例")
    print("=" * 50)
    
    # 运行各个示例
    await example_health_check()
    await example_audio_slice()
    await example_asr_recognition()
    await example_complete_workflow()
    await example_training()
    await example_inference()
    await example_batch_processing()
    await example_audio_utilities()
    
    # 运行同步示例
    sync_example_basic_usage()
    
    print("\n" + "=" * 50)
    print("✅ 所有示例运行完成!")


def run_sync_examples():
    """运行同步示例"""
    print("🔄 运行同步示例...")
    sync_example_basic_usage()


if __name__ == "__main__":
    # 运行异步示例
    asyncio.run(run_all_examples())
    
    # 或者只运行同步示例
    # run_sync_examples()