#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR语音识别模块使用示例

展示各种使用方式和最佳实践
"""

import asyncio
import os
import tempfile
from pathlib import Path

from . import ASRRecognitionService, ASRConfig, ASRRequest
from .utils import ASRUtils, create_test_audio_list


async def example_basic_usage():
    """基础使用示例"""
    print("=== 基础ASR识别示例 ===")
    
    # 创建测试音频
    with tempfile.TemporaryDirectory() as temp_dir:
        input_audio = os.path.join(temp_dir, "test.wav")
        output_dir = os.path.join(temp_dir, "output")
        
        # 这里应该是真实的音频文件
        # 为了演示，我们创建一个占位文件
        Path(input_audio).touch()
        
        # 初始化API
        api = ASRRecognitionService()
        
        # 使用默认配置
        request = ASRRequest(
            input_path=input_audio,
            output_dir=output_dir
        )
        
        print(f"使用默认配置进行识别...")
        print(f"输入文件: {input_audio}")
        print(f"输出目录: {output_dir}")
        
        # 注意：这里会失败因为是占位文件，实际使用时请提供真实音频
        # result = await api.recognize_audio(request)
        # print(f"识别结果: {result.success}")


async def example_custom_config():
    """自定义配置示例"""
    print("\n=== 自定义配置示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        input_audio = os.path.join(temp_dir, "chinese_audio.wav")
        output_dir = os.path.join(temp_dir, "output")
        
        Path(input_audio).touch()
        
        # 自定义中文识别配置
        config = ASRConfig(
            model_type="funasr",
            model_size="large",
            language="zh",
            precision="float32",
            vad_filter=True,
            beam_size=5
        )
        
        api = ASRRecognitionService()
        request = ASRRequest(
            input_path=input_audio,
            output_dir=output_dir,
            config=config
        )
        
        print(f"使用自定义配置:")
        print(f"  模型类型: {config.model_type}")
        print(f"  语言: {config.language}")
        print(f"  精度: {config.precision}")


async def example_smart_config():
    """智能配置示例"""
    print("\n=== 智能配置示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 模拟不同语言的音频文件
        test_files = [
            "chinese_speech_zh.wav",
            "english_audio_en.wav", 
            "japanese_voice_ja.wav",
            "cantonese_yue.wav"
        ]
        
        api = ASRRecognitionService()
        
        for filename in test_files:
            file_path = os.path.join(temp_dir, filename)
            Path(file_path).touch()
            
            # 智能配置建议
            smart_config = ASRUtils.suggest_config_for_file(file_path)
            
            print(f"文件: {filename}")
            print(f"  检测语言: {ASRUtils.detect_language_from_filename(filename)}")
            print(f"  建议配置: {smart_config.model_type}/{smart_config.language}")
            
            # 也可以手动指定语言获取建议
            manual_config = api.suggest_config(smart_config.language)
            print(f"  API建议: {manual_config.model_type}/{manual_config.language}")


async def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "audio_files")
        output_dir = os.path.join(temp_dir, "recognition_output")
        os.makedirs(input_dir)
        
        # 创建多个测试音频文件
        test_files = [
            "interview_01.wav",
            "interview_02.wav", 
            "meeting_record.wav",
            "phone_call.wav"
        ]
        
        for filename in test_files:
            Path(os.path.join(input_dir, filename)).touch()
        
        # 批量分析
        print("分析音频目录...")
        batch_info = ASRUtils.batch_analyze_directory(input_dir)
        
        print(f"发现 {batch_info['total_files']} 个音频文件")
        print(f"语言分布: {batch_info['language_distribution']}")
        print(f"主要语言: {batch_info['summary']['main_language']}")
        
        # 获取批量处理建议配置
        batch_config = batch_info['summary']['suggested_config']
        print(f"建议配置: {batch_config.model_type}/{batch_config.language}")
        
        # 执行批量识别
        api = ASRRecognitionService()
        print("开始批量识别...")
        
        # results = await api.batch_recognize(input_dir, output_dir, batch_config)
        # success_count = sum(1 for r in results if r.success)
        # print(f"批量识别完成: {success_count}/{len(results)} 成功")


def example_result_analysis():
    """结果分析示例"""
    print("\n=== 识别结果分析示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建示例识别结果文件
        list_file = os.path.join(temp_dir, "recognition_result.list")
        
        sample_results = [
            "/path/to/audio1.wav|speaker1|ZH|你好，这是一段测试音频。",
            "/path/to/audio2.wav|speaker1|ZH|今天天气很好，适合出去走走。",
            "/path/to/audio3.wav|speaker1|EN|Hello, this is a test audio file.",
            "/path/to/audio4.wav|speaker1|ZH|",  # 空识别结果
        ]
        
        with open(list_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sample_results))
        
        # 解析识别结果
        print("解析识别结果...")
        results = ASRUtils.parse_recognition_result(list_file)
        
        print(f"识别条目数: {len(results)}")
        for result in results:
            print(f"  {result['line_number']}: [{result['language']}] {result['text'][:50]}...")
        
        # 验证识别质量
        print("\n验证识别质量...")
        validation = ASRUtils.validate_recognition_result(list_file)
        
        print(f"验证结果: {'通过' if validation['valid'] else '失败'}")
        print(f"总条目数: {validation['total_entries']}")
        print(f"空文本数: {validation['empty_text_count']}")
        print(f"质量评分: {validation['quality_score']:.1f}/100")
        print(f"语言分布: {validation['language_distribution']}")


async def example_multi_language():
    """多语言识别示例"""
    print("\n=== 多语言识别示例 ===")
    
    # 支持的语言配置
    language_configs = {
        "中文": ASRConfig(model_type="funasr", language="zh", precision="float32"),
        "粤语": ASRConfig(model_type="funasr", language="yue", precision="float32"),
        "英文": ASRConfig(model_type="faster_whisper", language="en", precision="float16"),
        "日文": ASRConfig(model_type="faster_whisper", language="ja", precision="float16"),
        "韩文": ASRConfig(model_type="faster_whisper", language="ko", precision="float16"),
        "自动检测": ASRConfig(model_type="faster_whisper", language="auto", precision="float16")
    }
    
    print("支持的语言配置:")
    for lang_name, config in language_configs.items():
        print(f"  {lang_name}: {config.model_type} ({config.precision})")
    
    # 获取API支持的模型信息
    api = ASRRecognitionService()
    supported_models = api.get_supported_models()
    
    print(f"\nAPI支持的模型:")
    for model_type, info in supported_models.items():
        print(f"  {model_type}:")
        print(f"    语言: {info['languages']}")
        print(f"    大小: {info['sizes']}")
        print(f"    精度: {info['precisions']}")


def example_performance_optimization():
    """性能优化示例"""
    print("\n=== 性能优化示例 ===")
    
    # 不同场景的优化配置
    optimization_configs = {
        "快速识别": ASRConfig(
            model_type="faster_whisper",
            model_size="medium",
            precision="int8",
            beam_size=1
        ),
        "平衡模式": ASRConfig(
            model_type="faster_whisper", 
            model_size="large-v3",
            precision="float16",
            beam_size=3
        ),
        "高精度": ASRConfig(
            model_type="funasr",
            model_size="large",
            precision="float32",
            beam_size=5
        )
    }
    
    print("性能优化配置:")
    for mode, config in optimization_configs.items():
        print(f"  {mode}:")
        print(f"    模型: {config.model_type}/{config.model_size}")
        print(f"    精度: {config.precision}")
        print(f"    束搜索: {config.beam_size}")
    
    # 处理时间估算示例
    with tempfile.TemporaryDirectory() as temp_dir:
        test_audio = os.path.join(temp_dir, "test.wav")
        Path(test_audio).touch()
        
        # 模拟音频信息
        mock_info = {
            'duration': 60.0,  # 1分钟音频
            'sample_rate': 16000,
            'channels': 1
        }
        
        print(f"\n处理时间估算 (60秒音频):")
        for mode, config in optimization_configs.items():
            estimated_time = ASRUtils.estimate_processing_time(mock_info, config)
            print(f"  {mode}: 约 {estimated_time:.1f} 秒")


async def example_error_handling():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    api = ASRRecognitionService()
    
    # 测试各种错误情况
    error_cases = [
        {
            "name": "文件不存在",
            "request": ASRRequest(
                input_path="/nonexistent/file.wav",
                output_dir="/tmp/output"
            )
        },
        {
            "name": "不支持的配置",
            "request": ASRRequest(
                input_path="/tmp/test.wav",
                output_dir="/tmp/output",
                config=ASRConfig(
                    model_type="unsupported_model",
                    language="unsupported_lang"
                )
            )
        }
    ]
    
    for case in error_cases:
        print(f"测试: {case['name']}")
        try:
            result = await api.recognize_audio(case['request'])
            print(f"  结果: {result.success}")
            if not result.success:
                print(f"  错误: {result.message}")
        except Exception as e:
            print(f"  异常: {e}")


async def run_all_examples():
    """运行所有示例"""
    print("🎤 ASR语音识别模块使用示例")
    print("=" * 50)
    
    try:
        await example_basic_usage()
        await example_custom_config()
        await example_smart_config()
        await example_batch_processing()
        example_result_analysis()
        await example_multi_language()
        example_performance_optimization()
        await example_error_handling()
        
        print("\n✅ 所有示例运行完成！")
        print("\n💡 使用提示:")
        print("1. 实际使用时请提供真实的音频文件")
        print("2. 根据音频语言选择合适的模型配置")
        print("3. 大批量处理时建议使用批量接口")
        print("4. 注意检查识别结果的质量")
        
    except Exception as e:
        print(f"\n❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_examples())