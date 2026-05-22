#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 音频处理器

提供音频预处理、后处理和格式转换功能
"""

import os
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from typing import Dict, Tuple, Optional, List, Union
import tempfile
import io
import base64


class AudioProcessor:
    """音频处理器"""
    
    def __init__(self, device: str = "auto"):
        """
        初始化音频处理器
        
        Args:
            device: 计算设备，auto为自动选择
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
    
    def load_audio(self, audio_path: str, target_sr: int = None) -> Tuple[np.ndarray, int]:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            target_sr: 目标采样率，None为保持原采样率
            
        Returns:
            Tuple[np.ndarray, int]: 音频数据和采样率
        """
        try:
            if target_sr:
                audio, sr = librosa.load(audio_path, sr=target_sr)
            else:
                audio, sr = librosa.load(audio_path, sr=None)
            
            return audio, sr
            
        except Exception as e:
            raise RuntimeError(f"加载音频文件失败: {e}")
    
    def save_audio(self, audio: np.ndarray, sr: int, output_path: str, 
                   format: str = "wav") -> str:
        """
        保存音频文件
        
        Args:
            audio: 音频数据
            sr: 采样率
            output_path: 输出路径
            format: 音频格式
            
        Returns:
            str: 保存的文件路径
        """
        try:
            # 确保音频数据在合理范围内
            audio = np.clip(audio, -1.0, 1.0)
            
            if format.lower() == "wav":
                sf.write(output_path, audio, sr, format='WAV')
            elif format.lower() == "mp3":
                # 使用torchaudio保存MP3（需要ffmpeg）
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                torchaudio.save(output_path, audio_tensor, sr, format="mp3")
            elif format.lower() == "ogg":
                sf.write(output_path, audio, sr, format='OGG')
            elif format.lower() == "flac":
                sf.write(output_path, audio, sr, format='FLAC')
            else:
                # 默认使用WAV格式
                sf.write(output_path, audio, sr, format='WAV')
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"保存音频文件失败: {e}")
    
    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        重采样音频
        
        Args:
            audio: 原始音频数据
            orig_sr: 原始采样率
            target_sr: 目标采样率
            
        Returns:
            np.ndarray: 重采样后的音频数据
        """
        if orig_sr == target_sr:
            return audio
        
        try:
            resampled_audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
            return resampled_audio
            
        except Exception as e:
            raise RuntimeError(f"音频重采样失败: {e}")
    
    def normalize_audio(self, audio: np.ndarray, target_level: float = 0.9) -> np.ndarray:
        """
        音频归一化
        
        Args:
            audio: 音频数据
            target_level: 目标电平
            
        Returns:
            np.ndarray: 归一化后的音频数据
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio * (target_level / max_val)
        return audio
    
    def trim_silence(self, audio: np.ndarray, sr: int, 
                    top_db: int = 20, frame_length: int = 2048, 
                    hop_length: int = 512) -> np.ndarray:
        """
        去除音频首尾静音
        
        Args:
            audio: 音频数据
            sr: 采样率
            top_db: 静音阈值(dB)
            frame_length: 帧长度
            hop_length: 跳跃长度
            
        Returns:
            np.ndarray: 去除静音后的音频数据
        """
        try:
            trimmed_audio, _ = librosa.effects.trim(
                audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length
            )
            return trimmed_audio
            
        except Exception as e:
            print(f"去除静音失败: {e}")
            return audio
    
    def add_padding(self, audio: np.ndarray, sr: int, 
                   start_padding: float = 0.1, end_padding: float = 0.1) -> np.ndarray:
        """
        在音频首尾添加静音填充
        
        Args:
            audio: 音频数据
            sr: 采样率
            start_padding: 开始填充时长(秒)
            end_padding: 结束填充时长(秒)
            
        Returns:
            np.ndarray: 添加填充后的音频数据
        """
        start_samples = int(start_padding * sr)
        end_samples = int(end_padding * sr)
        
        start_silence = np.zeros(start_samples, dtype=audio.dtype)
        end_silence = np.zeros(end_samples, dtype=audio.dtype)
        
        padded_audio = np.concatenate([start_silence, audio, end_silence])
        return padded_audio
    
    def apply_fade(self, audio: np.ndarray, sr: int, 
                  fade_in_duration: float = 0.01, fade_out_duration: float = 0.01) -> np.ndarray:
        """
        应用淡入淡出效果
        
        Args:
            audio: 音频数据
            sr: 采样率
            fade_in_duration: 淡入时长(秒)
            fade_out_duration: 淡出时长(秒)
            
        Returns:
            np.ndarray: 应用淡入淡出后的音频数据
        """
        fade_in_samples = int(fade_in_duration * sr)
        fade_out_samples = int(fade_out_duration * sr)
        
        audio_copy = audio.copy()
        
        # 淡入
        if fade_in_samples > 0 and len(audio_copy) > fade_in_samples:
            fade_in_curve = np.linspace(0, 1, fade_in_samples)
            audio_copy[:fade_in_samples] *= fade_in_curve
        
        # 淡出
        if fade_out_samples > 0 and len(audio_copy) > fade_out_samples:
            fade_out_curve = np.linspace(1, 0, fade_out_samples)
            audio_copy[-fade_out_samples:] *= fade_out_curve
        
        return audio_copy
    
    def concatenate_audios(self, audio_list: List[np.ndarray], 
                          pause_duration: float = 0.3, sr: int = 22050) -> np.ndarray:
        """
        拼接多个音频片段
        
        Args:
            audio_list: 音频片段列表
            pause_duration: 片段间停顿时长(秒)
            sr: 采样率
            
        Returns:
            np.ndarray: 拼接后的音频数据
        """
        if not audio_list:
            return np.array([])
        
        if len(audio_list) == 1:
            return audio_list[0]
        
        # 创建停顿
        pause_samples = int(pause_duration * sr)
        pause = np.zeros(pause_samples, dtype=audio_list[0].dtype)
        
        # 拼接音频
        result = audio_list[0]
        for audio in audio_list[1:]:
            result = np.concatenate([result, pause, audio])
        
        return result
    
    def change_speed(self, audio: np.ndarray, speed_factor: float) -> np.ndarray:
        """
        改变音频播放速度
        
        Args:
            audio: 音频数据
            speed_factor: 速度因子，>1为加速，<1为减速
            
        Returns:
            np.ndarray: 变速后的音频数据
        """
        if speed_factor == 1.0:
            return audio
        
        try:
            # 使用librosa的时间拉伸
            stretched_audio = librosa.effects.time_stretch(audio, rate=speed_factor)
            return stretched_audio
            
        except Exception as e:
            print(f"变速处理失败: {e}")
            return audio
    
    def apply_volume(self, audio: np.ndarray, volume_factor: float) -> np.ndarray:
        """
        调整音频音量
        
        Args:
            audio: 音频数据
            volume_factor: 音量因子，1.0为原音量
            
        Returns:
            np.ndarray: 调整音量后的音频数据
        """
        adjusted_audio = audio * volume_factor
        # 防止溢出
        adjusted_audio = np.clip(adjusted_audio, -1.0, 1.0)
        return adjusted_audio
    
    def detect_voice_activity(self, audio: np.ndarray, sr: int, 
                            frame_length: int = 2048, hop_length: int = 512,
                            energy_threshold: float = 0.01) -> List[Tuple[float, float]]:
        """
        检测语音活动区间
        
        Args:
            audio: 音频数据
            sr: 采样率
            frame_length: 帧长度
            hop_length: 跳跃长度
            energy_threshold: 能量阈值
            
        Returns:
            List[Tuple[float, float]]: 语音活动区间列表(开始时间, 结束时间)
        """
        # 计算短时能量
        frame_energy = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]
        
        # 检测语音活动
        voice_frames = frame_energy > energy_threshold
        
        # 转换为时间区间
        frame_times = librosa.frames_to_time(
            np.arange(len(voice_frames)), sr=sr, hop_length=hop_length
        )
        
        # 找到连续的语音区间
        voice_segments = []
        start_time = None
        
        for i, is_voice in enumerate(voice_frames):
            if is_voice and start_time is None:
                start_time = frame_times[i]
            elif not is_voice and start_time is not None:
                voice_segments.append((start_time, frame_times[i]))
                start_time = None
        
        # 处理最后一个区间
        if start_time is not None:
            voice_segments.append((start_time, frame_times[-1]))
        
        return voice_segments
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        提取音频特征
        
        Args:
            audio: 音频数据
            sr: 采样率
            
        Returns:
            Dict[str, float]: 音频特征字典
        """
        features = {}
        
        try:
            # 基础特征
            features['duration'] = len(audio) / sr
            features['rms_energy'] = float(np.sqrt(np.mean(audio**2)))
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
            
            # 频谱特征
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # 基频特征
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
            )
            f0_clean = f0[voiced_flag]
            if len(f0_clean) > 0:
                features['f0_mean'] = float(np.mean(f0_clean))
                features['f0_std'] = float(np.std(f0_clean))
            else:
                features['f0_mean'] = 0.0
                features['f0_std'] = 0.0
            
        except Exception as e:
            print(f"特征提取失败: {e}")
        
        return features
    
    def audio_to_base64(self, audio: np.ndarray, sr: int, format: str = "wav") -> str:
        """
        将音频数据转换为Base64编码
        
        Args:
            audio: 音频数据
            sr: 采样率
            format: 音频格式
            
        Returns:
            str: Base64编码的音频数据
        """
        buffer = io.BytesIO()
        
        # 确保音频数据在合理范围内
        audio = np.clip(audio, -1.0, 1.0)
        
        if format.lower() == "wav":
            sf.write(buffer, audio, sr, format='WAV')
        elif format.lower() == "ogg":
            sf.write(buffer, audio, sr, format='OGG')
        else:
            sf.write(buffer, audio, sr, format='WAV')
        
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def base64_to_audio(self, base64_data: str) -> Tuple[np.ndarray, int]:
        """
        将Base64编码转换为音频数据
        
        Args:
            base64_data: Base64编码的音频数据
            
        Returns:
            Tuple[np.ndarray, int]: 音频数据和采样率
        """
        try:
            audio_bytes = base64.b64decode(base64_data)
            buffer = io.BytesIO(audio_bytes)
            audio, sr = sf.read(buffer)
            return audio, sr
            
        except Exception as e:
            raise RuntimeError(f"Base64音频解码失败: {e}")
    
    def validate_audio(self, audio: np.ndarray, sr: int, 
                      min_duration: float = 0.5, max_duration: float = 30.0) -> Dict[str, any]:
        """
        验证音频数据
        
        Args:
            audio: 音频数据
            sr: 采样率
            min_duration: 最小时长(秒)
            max_duration: 最大时长(秒)
            
        Returns:
            Dict: 验证结果
        """
        issues = []
        
        # 检查音频长度
        duration = len(audio) / sr
        if duration < min_duration:
            issues.append(f"音频时长过短: {duration:.2f}s < {min_duration}s")
        elif duration > max_duration:
            issues.append(f"音频时长过长: {duration:.2f}s > {max_duration}s")
        
        # 检查音频幅度
        max_amplitude = np.abs(audio).max()
        if max_amplitude == 0:
            issues.append("音频为静音")
        elif max_amplitude > 1.0:
            issues.append(f"音频幅度过大: {max_amplitude:.3f} > 1.0")
        
        # 检查采样率
        if sr < 8000:
            issues.append(f"采样率过低: {sr}Hz < 8000Hz")
        elif sr > 48000:
            issues.append(f"采样率过高: {sr}Hz > 48000Hz")
        
        # 检查音频质量
        rms_energy = np.sqrt(np.mean(audio**2))
        if rms_energy < 0.001:
            issues.append(f"音频能量过低: RMS={rms_energy:.6f}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "duration": duration,
            "max_amplitude": max_amplitude,
            "rms_energy": rms_energy,
            "sample_rate": sr
        }
