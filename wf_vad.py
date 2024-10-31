import json
import os
import sys
import torch
import torchaudio
from tqdm import tqdm

torch.set_num_threads(1)
device = torch.device('cpu')
# 从 silero-vad 加载模型和工具函数
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False  # 设置为 True 可强制重新下载模型
)

model.to(device)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

def process_jsonl_file(input_jsonl_path):
    output_jsonl_path = os.path.splitext(input_jsonl_path)[0] + '_vad.jsonl'

    with open(input_jsonl_path, 'r', encoding='utf-8') as infile, \
         open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        drop_count = 0
        total_count = 0
        for line in tqdm(infile):
            data = json.loads(line.strip())
            audio_path = data['audio']['path']
            full_audio_path = audio_path  # 根据实际情况调整路径
            total_count += 1
            # 检查音频文件是否存在
            if not os.path.isfile(full_audio_path):
                print(f"音頻文件不存在: {full_audio_path}")
                continue

            # 读取音频文件
            wav, sr = torchaudio.load(full_audio_path)
            # 如果采样率不是 8000 或 16000，则重新采样到 16000
            if sr not in [8000, 16000]:
                #print(f"音頻採樣率不是 8000 或 16000: {full_audio_path}")
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wav = resampler(wav)
                sr = 16000
            # 如果是多通道音频，取第一条通道
            if wav.shape[0] > 1:
                wav = wav[0, :]
            else:
                wav = wav.squeeze(0)
            wav = wav.to(device)
            # 获取语音段的时间戳
            speech_timestamps = get_speech_timestamps(wav, model, threshold=0.5, sampling_rate=sr, min_speech_duration_ms=100)
            # 计算音频总时长（秒）
            total_duration = wav.shape[0] / sr
            # 计算语音部分的总时长
            speech_duration = 0
            for t in speech_timestamps:
                speech_duration += (t['end'] - t['start']) / sr
            # 计算人声比例
            speech_proportion = speech_duration / total_duration
            if speech_proportion < 0.02:
                print(f"Drop {audio_path} 人声比例小于1/32: {speech_proportion}")
                drop_count += 1
                continue

            # 更新 sentences 的 start 和 end 时间
            new_sentences = []
            start_time = speech_timestamps[0]['start'] / sr
            end_time = speech_timestamps[len(speech_timestamps)-1]['end'] / sr
            new_sentences.append({
                'start': start_time,
                'end': end_time,
                'text': data.get('sentence', '')
            })
                

            data['sentences'] = new_sentences
            data['duration'] = total_duration  # 更新 duration
            # 将更新后的数据写入新的 jsonl 文件
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
        print(f"Dropped {drop_count}/{total_count} audio files.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python script_name.py input_jsonl_path")
        sys.exit(1)
    input_jsonl_path = sys.argv[1]
    process_jsonl_file(input_jsonl_path)