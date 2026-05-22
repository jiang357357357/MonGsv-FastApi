"""
GPT-SoVITS 文本处理模块使用示例

演示如何使用文本处理API进行文本特征提取
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from Code.FastApi.Base.DatasetFormatting.text_processing.service import TextProcessingService, TextProcessingConfig, TextProcessingRequest
from Code.FastApi.Base.DatasetFormatting.text_processing.utils import TextProcessingUtils


def create_example_data():
    """创建示例数据"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="text_processing_example_")
    
    # 创建示例标注文件
    example_list_file = os.path.join(temp_dir, "example.list")
    example_content = """audio_001.wav|speaker1|zh|你好，欢迎使用GPT-SoVITS文本处理模块。
audio_002.wav|speaker1|zh|这是一个用于提取文本特征的工具。
audio_003.wav|speaker2|en|Hello, this is an English text example.
audio_004.wav|speaker2|zh|今天天气很好，适合出去散步。
audio_005.wav|speaker1|zh|我们正在测试BERT特征提取功能。
audio_006.wav|speaker3|ja|こんにちは、世界。
audio_007.wav|speaker1|zh|文本处理是语音合成的重要步骤。
audio_008.wav|speaker2|zh|希望这个模块能够帮助到大家。"""
    
    with open(example_list_file, "w", encoding="utf8") as f:
        f.write(example_content)
    
    # 创建音频目录（模拟）
    audio_dir = os.path.join(temp_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    # 创建模拟音频文件
    for i in range(1, 9):
        audio_file = os.path.join(audio_dir, f"audio_{i:03d}.wav")
        with open(audio_file, "wb") as f:
            f.write(b"fake audio data for example")
    
    return temp_dir, example_list_file, audio_dir


async def example_basic_usage():
    """基础使用示例"""
    print("=== 基础文本处理示例 ===")
    
    # 创建示例数据
    temp_dir, list_file, audio_dir = create_example_data()
    output_dir = os.path.join(temp_dir, "output")
    
    try:
        # 创建API实例
        api = TextProcessingService()
        
        # 配置处理参数（跳过BERT以避免模型依赖）
        config = TextProcessingConfig(
            bert_pretrained_dir="",  # 空路径跳过BERT
            version="v2",
            is_half=False,
            device="cpu",
            n_parts=1
        )
        
        # 创建处理请求
        request = TextProcessingRequest(
            input_text_file=list_file,
            input_wav_dir=audio_dir,
            experiment_name="example_experiment",
            output_dir=output_dir,
            config=config
        )
        
        print(f"输入文件: {list_file}")
        print(f"音频目录: {audio_dir}")
        print(f"输出目录: {output_dir}")
        
        # 执行处理
        print("开始处理...")
        result = await api.process_text(request)
        
        # 显示结果
        if result.success:
            print(f"✅ 处理成功!")
            print(f"处理时间: {result.processing_time:.2f}秒")
            print(f"处理数量: {result.processed_count}")
            print(f"失败数量: {result.failed_count}")
            print(f"输出文件: {result.output_files}")
            
            # 显示输出文件内容
            if "text_file" in result.output_files and os.path.exists(result.output_files["text_file"]):
                print("\n输出文件内容预览:")
                with open(result.output_files["text_file"], "r", encoding="utf8") as f:
                    lines = f.readlines()[:3]  # 只显示前3行
                    for line in lines:
                        print(f"  {line.strip()}")
                    if len(lines) >= 3:
                        print("  ...")
        else:
            print(f"❌ 处理失败: {result.message}")
        
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)


def example_analysis_tools():
    """分析工具示例"""
    print("\n=== 文本分析工具示例 ===")
    
    # 创建示例数据
    temp_dir, list_file, audio_dir = create_example_data()
    
    try:
        # 1. 文件分析
        print("1. 分析文本文件:")
        analysis = TextProcessingUtils.analyze_text_file(list_file)
        print(f"   总行数: {analysis['total_lines']}")
        print(f"   有效行数: {analysis['valid_lines']}")
        print(f"   语言: {analysis['languages']}")
        print(f"   说话人: {analysis['speakers']}")
        print(f"   包含中文: {analysis['has_chinese']}")
        print(f"   平均文本长度: {analysis['avg_text_length']:.1f}")
        
        # 2. 配置建议
        print("\n2. 配置建议:")
        config = TextProcessingUtils.suggest_processing_config(list_file)
        print(f"   建议并行数: {config.n_parts}")
        print(f"   建议精度: {'半精度' if config.is_half else '全精度'}")
        print(f"   建议设备: {config.device}")
        
        # 3. 输入验证
        print("\n3. 输入验证:")
        validation = TextProcessingUtils.validate_input_files(
            list_file, audio_dir, check_audio_existence=True
        )
        print(f"   验证结果: {'通过' if validation['valid'] else '失败'}")
        if validation['errors']:
            print(f"   错误: {validation['errors']}")
        if validation['warnings']:
            print(f"   警告: {validation['warnings']}")
        
        # 4. 时间估算
        print("\n4. 处理时间估算:")
        time_estimate = TextProcessingUtils.estimate_processing_time(list_file, config)
        print(f"   预计总时间: {time_estimate['estimated_total_time']:.1f}秒")
        print(f"   每行处理时间: {time_estimate['estimated_time_per_line']:.3f}秒")
        print(f"   并行加速比: {time_estimate['parallel_speedup']:.1f}x")
        print(f"   包含BERT: {time_estimate['includes_bert']}")
        
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)


def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    # 创建多个示例文件
    temp_dir = tempfile.mkdtemp(prefix="batch_example_")
    
    try:
        # 创建多个标注文件
        for i in range(3):
            list_file = os.path.join(temp_dir, f"dataset_{i}.list")
            content = f"""file_{i}_001.wav|speaker{i}|zh|这是数据集{i}的第一个样本。
file_{i}_002.wav|speaker{i}|zh|这是数据集{i}的第二个样本。
file_{i}_003.wav|speaker{i}|en|This is sample {i+1} in English.
file_{i}_004.wav|speaker{i}|zh|数据集{i}包含多种语言的样本。"""
            
            with open(list_file, "w", encoding="utf8") as f:
                f.write(content)
        
        # 批量分析
        print("批量分析结果:")
        batch_analysis = TextProcessingUtils.batch_analyze_directory(temp_dir)
        print(f"   发现文件数: {batch_analysis['total_files']}")
        print(f"   总行数: {batch_analysis['summary']['total_lines']}")
        print(f"   总说话人数: {len(batch_analysis['summary']['total_speakers'])}")
        print(f"   总语言数: {len(batch_analysis['summary']['total_languages'])}")
        print(f"   包含中文: {batch_analysis['summary']['has_chinese']}")
        
        # 批量配置建议
        batch_config = TextProcessingUtils.suggest_batch_config(batch_analysis)
        print(f"\n批量处理建议:")
        print(f"   建议并行数: {batch_config.n_parts}")
        print(f"   建议版本: {batch_config.version}")
        
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)


async def example_advanced_config():
    """高级配置示例"""
    print("\n=== 高级配置示例 ===")
    
    # 创建示例数据
    temp_dir, list_file, audio_dir = create_example_data()
    output_dir = os.path.join(temp_dir, "output_advanced")
    
    try:
        # 创建API实例
        api = TextProcessingService()
        
        # 高级配置
        config = TextProcessingConfig(
            bert_pretrained_dir="",  # 跳过BERT
            version="v2",
            is_half=True,  # 使用半精度
            device="cpu",
            n_parts=2,  # 并行处理
            language_mapping={
                "ZH": "zh", "zh": "zh",
                "EN": "en", "en": "en", 
                "JA": "ja", "ja": "ja"
            }
        )
        
        # 创建处理请求
        request = TextProcessingRequest(
            input_text_file=list_file,
            input_wav_dir=audio_dir,
            experiment_name="advanced_example",
            output_dir=output_dir,
            config=config
        )
        
        print("使用高级配置处理...")
        print(f"并行数: {config.n_parts}")
        print(f"精度: {'半精度' if config.is_half else '全精度'}")
        
        # 执行处理
        result = await api.process_text(request)
        
        if result.success:
            print(f"✅ 高级配置处理成功!")
            print(f"处理详情: {result.details}")
        else:
            print(f"❌ 处理失败: {result.message}")
        
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)


async def main():
    """运行所有示例"""
    print("🎯 GPT-SoVITS 文本处理模块使用示例")
    print("=" * 50)
    
    try:
        # 基础使用示例
        await example_basic_usage()
        
        # 分析工具示例
        example_analysis_tools()
        
        # 批量处理示例
        example_batch_processing()
        
        # 高级配置示例
        await example_advanced_config()
        
        print("\n🎉 所有示例运行完成!")
        print("\n💡 使用提示:")
        print("1. 在实际使用中，请确保BERT模型路径正确")
        print("2. 根据数据量调整并行处理数")
        print("3. 中文文本会自动提取BERT特征")
        print("4. 使用智能配置工具获得最佳性能")
        
    except Exception as e:
        print(f"\n❌ 示例运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())