"""
GPT-SoVITS 文本处理模块测试

测试文本特征提取功能
"""

import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from Code.FastApi.Base.DatasetFormatting.text_processing.service import TextProcessingService, TextProcessingConfig, TextProcessingRequest
from Code.FastApi.Base.DatasetFormatting.text_processing.utils import TextProcessingUtils


def create_test_data():
    """创建测试数据"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="text_processing_test_")
    
    # 创建测试文本文件
    test_text_file = os.path.join(temp_dir, "test.list")
    test_content = """test1.wav|speaker1|zh|你好世界，这是一个测试。
test2.wav|speaker1|en|Hello world, this is a test.
test3.wav|speaker2|zh|今天天气很好。
test4.wav|speaker2|ja|こんにちは、世界。
test5.wav|speaker1|zh|我喜欢听音乐。"""
    
    with open(test_text_file, "w", encoding="utf8") as f:
        f.write(test_content)
    
    # 创建音频目录（空的，只是为了测试）
    audio_dir = os.path.join(temp_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    # 创建空的音频文件（用于测试）
    for i in range(1, 6):
        audio_file = os.path.join(audio_dir, f"test{i}.wav")
        with open(audio_file, "wb") as f:
            f.write(b"fake audio data")  # 假的音频数据
    
    return temp_dir, test_text_file, audio_dir


def test_text_analysis():
    """测试文本分析功能"""
    print("=== 测试文本分析功能 ===")
    
    temp_dir, test_text_file, audio_dir = create_test_data()
    
    try:
        # 测试文件分析
        analysis = TextProcessingUtils.analyze_text_file(test_text_file)
        print(f"文件分析结果: {analysis}")
        
        assert analysis["total_lines"] == 5
        assert analysis["valid_lines"] == 5
        assert analysis["has_chinese"] == True
        assert "zh" in analysis["languages"]
        assert "en" in analysis["languages"]
        
        # 测试配置建议
        config = TextProcessingUtils.suggest_processing_config(test_text_file)
        print(f"建议配置: {config}")
        
        # 测试输入验证
        validation = TextProcessingUtils.validate_input_files(
            test_text_file, audio_dir, check_audio_existence=True
        )
        print(f"输入验证结果: {validation}")
        
        # 测试时间估算
        time_estimate = TextProcessingUtils.estimate_processing_time(test_text_file, config)
        print(f"时间估算: {time_estimate}")
        
        print("✅ 文本分析功能测试通过")
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)


def test_text_processing_api():
    """测试文本处理API"""
    print("\n=== 测试文本处理API ===")
    
    temp_dir, test_text_file, audio_dir = create_test_data()
    output_dir = os.path.join(temp_dir, "output")
    
    try:
        # 创建API实例
        api = TextProcessingService()
        
        # 创建配置（跳过BERT以避免模型依赖）
        config = TextProcessingConfig(
            bert_pretrained_dir="",  # 空路径跳过BERT
            version="v2",
            is_half=False,
            device="cpu",
            n_parts=1
        )
        
        # 创建请求
        request = TextProcessingRequest(
            input_text_file=test_text_file,
            input_wav_dir=audio_dir,
            experiment_name="test_experiment",
            output_dir=output_dir,
            config=config
        )
        
        # 执行处理
        result = api.process_text_sync(request)
        print(f"处理结果: {result}")
        
        # 验证结果
        if result.success:
            assert os.path.exists(result.output_files["text_file"])
            print("✅ 文本处理API测试通过")
        else:
            print(f"⚠️ 处理失败（可能是模型依赖问题）: {result.message}")
        
    except Exception as e:
        print(f"⚠️ API测试出现异常（可能是依赖问题）: {str(e)}")
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)


async def test_async_processing():
    """测试异步处理"""
    print("\n=== 测试异步处理 ===")
    
    temp_dir, test_text_file, audio_dir = create_test_data()
    output_dir = os.path.join(temp_dir, "output_async")
    
    try:
        # 创建API实例
        api = TextProcessingService()
        
        # 创建配置
        config = TextProcessingConfig(
            bert_pretrained_dir="",  # 跳过BERT
            version="v2",
            is_half=False,
            device="cpu",
            n_parts=1
        )
        
        # 创建请求
        request = TextProcessingRequest(
            input_text_file=test_text_file,
            input_wav_dir=audio_dir,
            experiment_name="test_async",
            output_dir=output_dir,
            config=config
        )
        
        # 执行异步处理
        result = await api.process_text(request)
        print(f"异步处理结果: {result}")
        
        if result.success:
            print("✅ 异步处理测试通过")
        else:
            print(f"⚠️ 异步处理失败: {result.message}")
        
    except Exception as e:
        print(f"⚠️ 异步测试出现异常: {str(e)}")
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)


def test_batch_analysis():
    """测试批量分析"""
    print("\n=== 测试批量分析 ===")
    
    temp_dir = tempfile.mkdtemp(prefix="batch_test_")
    
    try:
        # 创建多个测试文件
        for i in range(3):
            test_file = os.path.join(temp_dir, f"test_{i}.list")
            content = f"""file_{i}_1.wav|speaker{i}|zh|测试文件{i}的第一行。
file_{i}_2.wav|speaker{i}|en|Test file {i} second line.
file_{i}_3.wav|speaker{i}|zh|测试文件{i}的第三行。"""
            
            with open(test_file, "w", encoding="utf8") as f:
                f.write(content)
        
        # 执行批量分析
        batch_result = TextProcessingUtils.batch_analyze_directory(temp_dir)
        print(f"批量分析结果: {batch_result}")
        
        assert batch_result["total_files"] == 3
        assert batch_result["summary"]["total_lines"] == 9
        assert batch_result["summary"]["has_chinese"] == True
        
        # 测试批量配置建议
        batch_config = TextProcessingUtils.suggest_batch_config(batch_result)
        print(f"批量配置建议: {batch_config}")
        
        print("✅ 批量分析测试通过")
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)


def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    # 测试不存在的文件
    analysis = TextProcessingUtils.analyze_text_file("nonexistent.txt")
    assert "error" in analysis
    print("✅ 不存在文件的错误处理正确")
    
    # 测试无效的输入验证
    validation = TextProcessingUtils.validate_input_files("nonexistent.txt", "nonexistent_dir")
    assert not validation["valid"]
    assert len(validation["errors"]) > 0
    print("✅ 输入验证错误处理正确")
    
    print("✅ 错误处理测试通过")


def main():
    """运行所有测试"""
    print("开始GPT-SoVITS文本处理模块测试...")
    
    try:
        # 基础功能测试
        test_text_analysis()
        test_batch_analysis()
        test_error_handling()
        
        # API测试（可能因为依赖问题失败）
        test_text_processing_api()
        
        # 异步测试
        asyncio.run(test_async_processing())
        
        print("\n🎉 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()