#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS API测试模块

提供API功能的全面测试
"""

import asyncio
import os
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any

from .client import GPTSoVITSClient, SyncGPTSoVITSClient
from .models import *
from .utils import *
from .exceptions import *


class TestGPTSoVITSAPI:
    """GPT-SoVITS API测试类"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = SyncGPTSoVITSClient(base_url=base_url)
        self.async_client = GPTSoVITSClient(base_url=base_url)
        self.temp_files = []
    
    def cleanup(self):
        """清理测试文件"""
        FileUtils.cleanup_temp_files(self.temp_files)
        self.temp_files.clear()
    
    def create_test_audio(self, duration: float = 3.0) -> str:
        """创建测试音频文件"""
        audio_file = create_temp_audio_file(duration)
        self.temp_files.append(audio_file)
        return audio_file
    
    def create_test_list_file(self, audio_files: list) -> str:
        """创建测试标注文件"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.list', encoding='utf-8')
        
        for i, audio_file in enumerate(audio_files):
            # 格式: audio_path|speaker|language|text
            temp_file.write(f"{audio_file}|speaker1|zh|这是第{i+1}个测试音频。\n")
        
        temp_file.close()
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    # ==================== 基础测试 ====================
    
    def test_health_check(self):
        """测试健康检查"""
        print("测试健康检查...")
        try:
            result = self.client.health_check()
            print(f"✓ 健康检查成功: {result}")
            return True
        except Exception as e:
            print(f"✗ 健康检查失败: {e}")
            return False
    
    def test_services_status(self):
        """测试服务状态"""
        print("测试服务状态...")
        try:
            async def _test():
                async with self.async_client as client:
                    return await client.get_services_status()
            
            result = asyncio.run(_test())
            print(f"✓ 服务状态获取成功: {result}")
            return True
        except Exception as e:
            print(f"✗ 服务状态获取失败: {e}")
            return False
    
    # ==================== 数据准备测试 ====================
    
    def test_audio_slice(self):
        """测试音频切分"""
        print("测试音频切分...")
        try:
            # 创建测试音频
            audio_file = self.create_test_audio(10.0)  # 10秒音频
            output_dir = tempfile.mkdtemp()
            
            # 构建请求
            builder = get_request_builder()
            request = builder.audio_slice_request(
                input_path=audio_file,
                output_dir=output_dir,
                threshold=-30.0,
                min_length=2000
            )
            
            # 执行切分
            response = self.client.audio_slice(request)
            
            print(f"✓ 音频切分成功:")
            print(f"  输出目录: {response.output_dir}")
            print(f"  处理文件数: {len(response.processed_files)}")
            print(f"  输出文件数: {len(response.output_files)}")
            
            return True
        except Exception as e:
            print(f"✗ 音频切分失败: {e}")
            return False
    
    def test_asr_recognize(self):
        """测试ASR识别"""
        print("测试ASR识别...")
        try:
            # 创建测试音频目录
            audio_dir = tempfile.mkdtemp()
            for i in range(3):
                audio_file = self.create_test_audio(3.0)
                # 复制到测试目录
                import shutil
                shutil.copy(audio_file, os.path.join(audio_dir, f"test_{i}.wav"))
            
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.list').name
            self.temp_files.append(output_file)
            
            # 构建请求
            builder = get_request_builder()
            request = builder.asr_request(
                input_path=audio_dir,
                output_file=output_file,
                model_type="funasr",
                language="zh"
            )
            
            # 执行识别
            response = self.client.asr_recognize(request)
            
            print(f"✓ ASR识别成功:")
            print(f"  输出文件: {response.output_file}")
            print(f"  处理文件数: {len(response.processed_files)}")
            print(f"  识别结果数: {len(response.recognition_results)}")
            
            return True
        except Exception as e:
            print(f"✗ ASR识别失败: {e}")
            return False
    
    # ==================== 数据格式化测试 ====================
    
    def test_text_processing(self):
        """测试文本处理"""
        print("测试文本处理...")
        try:
            # 创建测试标注文件
            audio_files = [self.create_test_audio() for _ in range(3)]
            list_file = self.create_test_list_file(audio_files)
            output_dir = tempfile.mkdtemp()
            
            # 构建请求
            builder = get_request_builder()
            request = builder.text_processing_request(
                list_file=list_file,
                output_dir=output_dir,
                language="zh"
            )
            
            # 执行处理
            response = self.client.text_processing(request)
            
            print(f"✓ 文本处理成功:")
            print(f"  输出目录: {response.output_dir}")
            print(f"  文本文件: {response.text_file}")
            print(f"  BERT目录: {response.bert_dir}")
            
            return True
        except Exception as e:
            print(f"✗ 文本处理失败: {e}")
            return False
    
    def test_audio_features(self):
        """测试音频特征提取"""
        print("测试音频特征提取...")
        try:
            # 创建测试标注文件
            audio_files = [self.create_test_audio() for _ in range(2)]
            list_file = self.create_test_list_file(audio_files)
            output_dir = tempfile.mkdtemp()
            
            # 构建请求
            builder = get_request_builder()
            request = builder.audio_features_request(
                list_file=list_file,
                output_dir=output_dir,
                version="v2Pro"
            )
            
            # 执行提取
            response = self.client.audio_features(request)
            
            print(f"✓ 音频特征提取成功:")
            print(f"  输出目录: {response.output_dir}")
            print(f"  CNHubert目录: {response.cnhubert_dir}")
            print(f"  WAV32K目录: {response.wav32k_dir}")
            
            return True
        except Exception as e:
            print(f"✗ 音频特征提取失败: {e}")
            return False
    
    def test_semantic_encoding(self):
        """测试语义编码"""
        print("测试语义编码...")
        try:
            # 创建测试标注文件
            audio_files = [self.create_test_audio() for _ in range(2)]
            list_file = self.create_test_list_file(audio_files)
            output_dir = tempfile.mkdtemp()
            
            # 构建请求
            builder = get_request_builder()
            request = builder.semantic_encoding_request(
                list_file=list_file,
                output_dir=output_dir,
                version="v2Pro"
            )
            
            # 执行编码
            response = self.client.semantic_encoding(request)
            
            print(f"✓ 语义编码成功:")
            print(f"  输出目录: {response.output_dir}")
            print(f"  语义文件: {response.semantic_file}")
            
            return True
        except Exception as e:
            print(f"✗ 语义编码失败: {e}")
            return False
    
    # ==================== 训练测试 ====================
    
    def test_gpt_training(self):
        """测试GPT训练"""
        print("测试GPT训练...")
        try:
            # 构建请求
            builder = get_request_builder()
            request = builder.gpt_training_request(
                exp_name="test_gpt",
                exp_root=tempfile.mkdtemp(),
                batch_size=4,
                total_epoch=2,
                learning_rate=0.01
            )
            
            # 开始训练
            response = self.client.start_gpt_training(request)
            
            print(f"✓ GPT训练启动成功:")
            print(f"  任务ID: {response.job_id}")
            print(f"  实验名称: {response.exp_name}")
            print(f"  状态: {response.status}")
            
            # 检查训练状态
            if response.job_id:
                status = self.client.get_training_status(response.job_id)
                print(f"  训练状态: {status.status}")
            
            return True
        except Exception as e:
            print(f"✗ GPT训练失败: {e}")
            return False
    
    def test_sovits_training(self):
        """测试SoVITS训练"""
        print("测试SoVITS训练...")
        try:
            # 构建请求
            builder = get_request_builder()
            request = builder.sovits_training_request(
                exp_name="test_sovits",
                exp_root=tempfile.mkdtemp(),
                version="v2Pro",
                batch_size=16,
                total_epoch=2
            )
            
            # 开始训练
            response = self.client.start_sovits_training(request)
            
            print(f"✓ SoVITS训练启动成功:")
            print(f"  任务ID: {response.job_id}")
            print(f"  实验名称: {response.exp_name}")
            print(f"  状态: {response.status}")
            
            return True
        except Exception as e:
            print(f"✗ SoVITS训练失败: {e}")
            return False
    
    # ==================== 推理测试 ====================
    
    def test_inference(self):
        """测试推理"""
        print("测试推理...")
        try:
            # 创建参考音频
            ref_audio = self.create_test_audio(5.0)
            
            # 构建请求
            builder = get_request_builder()
            request = builder.inference_request(
                text="你好，这是一个测试语音合成。",
                ref_audio_path=ref_audio,
                prompt_text="参考音频的文本内容",
                text_language="zh",
                prompt_language="zh",
                return_base64=True
            )
            
            # 执行推理
            response = self.client.inference(request)
            
            print(f"✓ 推理成功:")
            print(f"  音频数据长度: {len(response.audio_data) if response.audio_data else 0}")
            print(f"  音频路径: {response.audio_path}")
            print(f"  采样率: {response.sample_rate}")
            print(f"  时长: {response.duration}")
            
            return True
        except Exception as e:
            print(f"✗ 推理失败: {e}")
            return False
    
    def test_inference_with_base64(self):
        """测试Base64音频推理"""
        print("测试Base64音频推理...")
        try:
            # 创建参考音频并编码
            ref_audio = self.create_test_audio(3.0)
            ref_audio_base64 = AudioUtils.encode_audio_file(ref_audio)
            
            # 构建请求
            builder = get_request_builder()
            request = builder.inference_request(
                text="这是使用Base64音频的测试。",
                ref_audio_base64=ref_audio_base64,
                prompt_text="参考音频文本",
                return_base64=True
            )
            
            # 执行推理
            async def _test():
                async with self.async_client as client:
                    return await client.inference_with_base64_audio(request)
            
            response = asyncio.run(_test())
            
            print(f"✓ Base64音频推理成功:")
            print(f"  音频数据长度: {len(response.audio_data) if response.audio_data else 0}")
            
            return True
        except Exception as e:
            print(f"✗ Base64音频推理失败: {e}")
            return False
    
    # ==================== 工作流测试 ====================
    
    def test_complete_workflow(self):
        """测试完整工作流"""
        print("测试完整工作流...")
        try:
            # 创建测试音频目录
            input_dir = tempfile.mkdtemp()
            for i in range(3):
                audio_file = self.create_test_audio(5.0)
                import shutil
                shutil.copy(audio_file, os.path.join(input_dir, f"audio_{i}.wav"))
            
            output_dir = tempfile.mkdtemp()
            
            # 构建请求
            builder = get_request_builder()
            request = builder.workflow_request(
                project_name="test_workflow",
                input_audio_dir=input_dir,
                output_dir=output_dir,
                language="zh",
                version="v2Pro"
            )
            
            # 执行工作流
            response = self.client.complete_workflow(request)
            
            print(f"✓ 完整工作流启动成功:")
            print(f"  项目名称: {response.project_name}")
            print(f"  工作流ID: {response.workflow_id}")
            print(f"  当前步骤: {response.current_step}")
            print(f"  总体进度: {response.overall_progress}")
            
            return True
        except Exception as e:
            print(f"✗ 完整工作流失败: {e}")
            return False
    
    # ==================== 工具测试 ====================
    
    def test_audio_utils(self):
        """测试音频工具"""
        print("测试音频工具...")
        try:
            # 创建测试音频
            audio_file = self.create_test_audio(3.0)
            
            # 测试编码解码
            base64_data = AudioUtils.encode_audio_file(audio_file)
            print(f"✓ 音频编码成功，长度: {len(base64_data)}")
            
            # 解码到新文件
            decoded_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
            self.temp_files.append(decoded_file)
            AudioUtils.decode_audio_base64(base64_data, decoded_file)
            print(f"✓ 音频解码成功: {decoded_file}")
            
            # 获取音频信息
            info = AudioUtils.get_audio_info(audio_file)
            print(f"✓ 音频信息: {info}")
            
            # 验证音频文件
            AudioUtils.validate_audio_file(audio_file, min_duration=1.0, max_duration=10.0)
            print(f"✓ 音频验证通过")
            
            return True
        except Exception as e:
            print(f"✗ 音频工具测试失败: {e}")
            return False
    
    def test_file_utils(self):
        """测试文件工具"""
        print("测试文件工具...")
        try:
            # 创建测试文件
            test_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
            test_file.write("测试内容")
            test_file.close()
            self.temp_files.append(test_file.name)
            
            # 测试文件哈希
            file_hash = FileUtils.get_file_hash(test_file.name)
            print(f"✓ 文件哈希: {file_hash}")
            
            # 测试目录创建
            test_dir = tempfile.mkdtemp()
            FileUtils.ensure_dir(os.path.join(test_dir, "subdir"))
            print(f"✓ 目录创建成功")
            
            # 测试文件查找
            files = FileUtils.find_files(test_dir, "*.tmp")
            print(f"✓ 文件查找: {len(files)} 个文件")
            
            return True
        except Exception as e:
            print(f"✗ 文件工具测试失败: {e}")
            return False
    
    # ==================== 综合测试 ====================
    
    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试"""
        print("=" * 60)
        print("开始GPT-SoVITS API全面测试")
        print("=" * 60)
        
        test_methods = [
            ("健康检查", self.test_health_check),
            ("服务状态", self.test_services_status),
            ("音频切分", self.test_audio_slice),
            ("ASR识别", self.test_asr_recognize),
            ("文本处理", self.test_text_processing),
            ("音频特征", self.test_audio_features),
            ("语义编码", self.test_semantic_encoding),
            ("GPT训练", self.test_gpt_training),
            ("SoVITS训练", self.test_sovits_training),
            ("推理", self.test_inference),
            ("Base64推理", self.test_inference_with_base64),
            ("完整工作流", self.test_complete_workflow),
            ("音频工具", self.test_audio_utils),
            ("文件工具", self.test_file_utils),
        ]
        
        results = {}
        passed = 0
        total = len(test_methods)
        
        for test_name, test_method in test_methods:
            print(f"\n[{passed + 1}/{total}] {test_name}")
            print("-" * 40)
            
            try:
                result = test_method()
                results[test_name] = result
                if result:
                    passed += 1
            except Exception as e:
                print(f"✗ 测试异常: {e}")
                results[test_name] = False
        
        # 清理测试文件
        self.cleanup()
        
        # 输出测试结果
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        
        for test_name, result in results.items():
            status = "✓ 通过" if result else "✗ 失败"
            print(f"{test_name:<20} {status}")
        
        print(f"\n总计: {passed}/{total} 个测试通过")
        print(f"成功率: {passed/total*100:.1f}%")
        
        return results


# ==================== 性能测试 ====================

class PerformanceTest:
    """性能测试类"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = SyncGPTSoVITSClient(base_url=base_url)
    
    def test_concurrent_requests(self, num_requests: int = 10):
        """测试并发请求"""
        print(f"测试 {num_requests} 个并发健康检查请求...")
        
        import concurrent.futures
        import time
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(self.client.health_check) for _ in range(num_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✓ {num_requests} 个并发请求完成")
        print(f"  总耗时: {duration:.2f}秒")
        print(f"  平均响应时间: {duration/num_requests:.3f}秒")
        print(f"  QPS: {num_requests/duration:.1f}")
        
        return results
    
    def test_large_audio_processing(self):
        """测试大音频文件处理"""
        print("测试大音频文件处理...")
        
        # 创建较大的测试音频（30秒）
        large_audio = create_temp_audio_file(30.0)
        
        try:
            start_time = time.time()
            
            # 测试音频切分
            output_dir = tempfile.mkdtemp()
            builder = get_request_builder()
            request = builder.audio_slice_request(
                input_path=large_audio,
                output_dir=output_dir
            )
            
            response = self.client.audio_slice(request)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✓ 大音频文件处理完成")
            print(f"  处理时间: {duration:.2f}秒")
            print(f"  输出文件数: {len(response.output_files)}")
            
        finally:
            os.unlink(large_audio)


# ==================== 主测试函数 ====================

def run_basic_tests(base_url: str = "http://localhost:8000"):
    """运行基础测试"""
    tester = TestGPTSoVITSAPI(base_url)
    return tester.run_all_tests()


def run_performance_tests(base_url: str = "http://localhost:8000"):
    """运行性能测试"""
    tester = PerformanceTest(base_url)
    
    print("=" * 60)
    print("性能测试")
    print("=" * 60)
    
    # 并发测试
    tester.test_concurrent_requests(10)
    
    # 大文件测试
    tester.test_large_audio_processing()


def run_all_tests(base_url: str = "http://localhost:8000"):
    """运行所有测试"""
    print("开始完整测试套件...")
    
    # 基础功能测试
    basic_results = run_basic_tests(base_url)
    
    # 性能测试
    run_performance_tests(base_url)
    
    return basic_results


if __name__ == "__main__":
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print(f"测试目标: {base_url}")
    results = run_all_tests(base_url)
    
    # 根据测试结果设置退出码
    failed_tests = [name for name, result in results.items() if not result]
    if failed_tests:
        print(f"\n失败的测试: {', '.join(failed_tests)}")
        sys.exit(1)
    else:
        print("\n所有测试通过！")
        sys.exit(0)