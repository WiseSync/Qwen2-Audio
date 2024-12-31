from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import base64
import requests
import io
import re
import argparse
import uvicorn
from torchaudio.functional import forced_align
import traceback
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import platform
import regex as re
from itn.chinese.inverse_normalizer import InverseNormalizer
from tn.chinese.normalizer import Normalizer as ZhNormalizer
import logging

# 初始化 FastAPI 应用程序
app = FastAPI()

# 初始化模型等全局变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

zh_tn_model = ZhNormalizer(remove_erhua=False, remove_puncts=True, full_to_half=False, traditional_to_simple=False, remove_interjections=False,cache_dir='cache/tn/normalizer')
zh_itn_model = InverseNormalizer(enable_0_to_9=False,cache_dir='cache/tn/inverse')
#model_path = '/Users/jon/NTNU Dropbox/WiseSync/NAS/Models/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt'
model_path = '/workspace/Models/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt'
dtype = torch.float16
processor = Wav2Vec2Processor.from_pretrained(model_path,torch_dtype=dtype)
model = Wav2Vec2ForCTC.from_pretrained(model_path,torch_dtype=dtype,device_map=device)
model.eval()

# 定义响应模型
class WordSegment(BaseModel):
    start: float
    end: float
    word: str
    confidence: float

class OutSegment(BaseModel):
    start: float
    end: float
    text: str
    words: list[WordSegment]

import re

""" def replace_timestamps(text):
    #print(text)
    # 定义正则表达式模式，匹配行首的时间戳格式：数字,数字 -> 数字,数字
    pattern = r'^(?:\s*)(\d+(?:,\d+)?\s*->\s*\d+(?:,\d+)?)(\s*)'
    
    # 使用 splitlines 保留行尾的换行符
    lines = text.splitlines(keepends=True)
    
    # 用于跟踪是否是第一处时间戳
    first_timestamp = True
    
    # 存储处理后的行
    new_lines = []
    
    for line in lines:
        # 检查行首是否有时间戳
        match = re.match(pattern, line)
        if match:
            if first_timestamp:
                # 第一处时间戳，替换为空字符串
                new_line = re.sub(pattern, '', line, count=1)
                first_timestamp = False
            else:
                # 其他时间戳，替换为句号
                # 保留时间戳后面的空白字符（如空格或制表符）
                new_line = re.sub(pattern, '。', line, count=1)
            if(new_line[-1] == '\n'):
                new_line = new_line[:-1]
            else:
                new_line += '。'
            new_lines.append(new_line)
        else:
            # 行首没有时间戳，直接添加
            new_lines.append(line)
    
    # 将处理后的行重新组合成字符串
    result = ''.join(new_lines)
    pattern2 = r'(?:\d+(?:,{1,2}\d+)?\s*->\s*)+\d+(?:,{1,2}\d+)?'
    result = re.sub(pattern2, r'', result)
    return result """

def remove_repeated_phrases(text):
    # 移除句子末尾的空白字符
    text = text.rstrip()

    max_phrase_length = 30  # 最大短语长度，可根据需要调整
    min_repeats = 5  # 最小重复次数

    # 获取文本长度
    text_length = len(text)

    # 计算可能的最大短语长度，防止越界
    max_possible_length = min(max_phrase_length, text_length // min_repeats)

    # 从短语长度为 1 开始，逐渐增加
    for phrase_length in range(1, max_possible_length + 1):
        # 获取末尾的短语
        phrase = text[-phrase_length:]
        # 构建重复的短语
        repeated_phrase = phrase * min_repeats
        # 检查文本是否以重复的短语结尾
        if text.endswith(repeated_phrase):
            # 使用正则表达式查找重复的部分
            pattern = f'({re.escape(phrase)}){{{min_repeats},}}$'
            match = re.search(pattern, text)
            if match:
                # 将重复的部分替换为单个短语
                text = text[:match.start()] + ''
                ntlen = len(text)
                plen = len(phrase)
                index = None
                for i in range(0, plen):
                    if(text[ntlen-i-1] == phrase[plen-i-1]):
                        index = i
                        continue
                    else:
                        break
                if(index != None):
                    text = text+phrase[:-index-1]
                return text

    # 如果未找到重复的短语，返回原文本
    return text

# 转录函数
def transcribe(data):
    try:
        url = 'http://127.0.0.1:5000/v1/chat/completions'

        headers = {
            'Content-Type': 'application/json'
        }

        payload = data

        response = requests.post(url, headers=headers, json=payload)
        #print(payload)
        if not response.ok:
            raise Exception(f"HTTP error! status: {response.status_code}")

        data = response.json()

        if (isinstance(data.get('choices'), list) and len(data['choices']) > 0 and
                isinstance(data['choices'][0]['message']['content'], str)):
            #removed_timestamps =  replace_timestamps(data['choices'][0]['message']['content'])
            removed_repeats = remove_repeated_phrases(data['choices'][0]['message']['content'])

            return removed_repeats
        else:
            raise Exception("Transcription error: " + str(data))

    except Exception as error:
        logging.warning('Error:', error)
        raise error

# 将 base64 数据转换为原始音频数据
def base64_to_raw_data(data_uri):
    header, base64_str = data_uri.split(',', 1)
    audio_data = base64.b64decode(base64_str)
    return audio_data

def is_output_img():
    return False

def split_text(text):
    import re
    # Define Chinese punctuation marks
    chinese_punctuation = '，。！？;；'
    
    # Define the regular expression pattern
    pattern = (
        fr'(?<=[{chinese_punctuation}])'  # 在中文句末标点符号后面分割
        r'|(?<=[^\d\W])\.'               # 在非数字和非标点符号的字符后跟一个句号
        r'(?!\d|[a-zA-Z]\.|\.{1,})\s*'   # 句号后面不是数字、小写字母加句号或连续的句号
    )
    
    # 进行分割
    segments = re.split(pattern, text)
    
    # 去除每个段落首尾的空白字符，并移除空字符串
    segments = [s.strip() for s in segments if s.strip()]
    
    return segments

# 主处理函数
def process_audio(data):
    # 提取音频格式
    messages = data['messages']
    contents = messages[len(messages)-1]['content']
    audio_base64=None
    for content in contents:
        if content['type'] == 'audio_url':
            audio_base64 = content['audio_url']['url']
            break
    if audio_base64 is None:
        raise ValueError("No audio data found in the request")

    format_match = re.search(r'data:audio/(\w+);base64,', audio_base64)
    if format_match:
        audio_format = format_match.group(1)
    else:
        raise ValueError("Invalid data URI format")

    # 将 base64 数据转换为原始音频数据
    audio_data = base64_to_raw_data(audio_base64)

    # 加载音频
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data), format=audio_format)
    waveform = waveform.squeeze(0)
    #print(f"sample_rate: {sample_rate}")
    # 如果采样率不是 16kHz，重新采样
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    # 计算 frame_shift
    config = model.config
    if hasattr(config, 'conv_stride'):
        conv_stride = config.conv_stride
    elif hasattr(config, 'feat_extract_stride'):
        conv_stride = config.feat_extract_stride
    else:
        raise ValueError("Cannot find conv_stride or feat_extract_stride in model config.")

    total_stride = np.prod(conv_stride)
    frame_shift = total_stride / sample_rate  # 单位：秒

    #print(f"frame_shift: {frame_shift}")

    # 合并文本
    #transcript = "好，被撞擊而不幸身亡的蘇姓員警，他們家屬下午兩點來到了分駐所，進行招魂儀式。同袍也在門口列隊，要送他最後一程。詳細情況連線給記者林荷容、荷容。帶您來關注這起嫌犯開贓車撞死巡路員和員警的悲劇事件。就在今天下午的兩點，殉職員警的家屬也來到八堵的分駐所進行招魂儀式。可以看到門口兩兩排刑員警呢，都是…"
    transcript = transcribe(data)
    texts = split_text(transcript)
    segments = [{'normalized': zh_tn_model.normalize(text),'inversed':'', 'text':text} for text in texts]
    
    for segment in segments:
        segment['inversed'] = zh_itn_model.normalize(segment['text'])

    table = []
    transcript_s = ""   
    for i in range(len(segments)):
        transcript_s += segments[i]['normalized']
        for j in range(len(segments[i]['normalized'])):
            table.append((i, j))

    # 将简体中文文本编码为 tokens
    labels = processor.tokenizer.encode(transcript_s, add_special_tokens=False)

    # 准备输入
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = inputs.to(dtype=dtype).to(device)
    targets = torch.tensor(labels, dtype=torch.int32).unsqueeze(0).to(device)

    # 获取发射矩阵
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    emission = torch.log_softmax(logits, dim=-1)

    # 强制对齐


    alignment, scores = forced_align(
        log_probs=emission,
        targets=targets,
        blank=processor.tokenizer.pad_token_id  # 通常 blank_id 为 0
    )
    #print(f"Alignment scores: {scores}, {scores.shape}", {len(alignment[0])}, {len(labels)})
    # 获取 tokenizer 的 tokens
    tokens = processor.tokenizer.convert_ids_to_tokens(labels)

    # 解析对齐结果
    alignment = alignment[0].cpu().numpy()
    words = []
    current_token_id = None
    current_idx = None
    start_time = 0
    token_index = 0
    emission = emission.cpu()
    transcribe_index = 0
    for idx, token_id in enumerate(alignment):
        if token_id != current_token_id:
            if current_token_id is not None and current_token_id != processor.tokenizer.pad_token_id:
                end_time = idx * frame_shift
                token = tokens[token_index]
                text = ''
                while len(text.replace(' ', ''))<len(token) and transcribe_index < len(transcript_s):
                    if token == '[UNK]':
                        text+= transcript_s[transcribe_index]
                        transcribe_index += 1
                        break
                    text += transcript_s[transcribe_index]
                    transcribe_index += 1
                if(len(text)<=0):
                    logging.error(f"Warning: {token} != {text}")
                i, h = table[token_index]
                log_probs = emission[0][current_idx, current_token_id]
                # 计算平均概率作为置信度
                confidence = np.exp(log_probs).mean()
                #print(f"Token: {token}, Text: {text}, Confidence: {confidence}")       
                words.append((start_time, end_time, text, confidence, i))
                token_index += 1
            current_token_id = token_id
            current_idx = idx
            start_time = idx * frame_shift
    # 处理最后一个 token
    if current_token_id != processor.tokenizer.pad_token_id and token_index < len(tokens):
        end_time = len(alignment) * frame_shift
        token = tokens[token_index]
        text = ''
        while len(text.replace(' ', ''))<len(token) and transcribe_index < len(transcript_s):
            if token == '[UNK]':
                text+= transcript_s[transcribe_index]
                transcribe_index += 1
                break
            text += transcript_s[transcribe_index]
            transcribe_index += 1
        if(len(text)<=0):
            logging.error(f"Warning: {token} != {text}")
        i, h = table[token_index]
        log_probs = emission[0][current_idx, current_token_id]
        # 计算平均概率作为置信度
        confidence = np.exp(log_probs).mean()
        #Check if the last token is a [UNK] or not
        #print(f"Token: {token}, Text: {text}, Confidence: {confidence}")         
        words.append((start_time, end_time, text, log_probs,i))
        token_index += 1
    #print(words)

    # 将 words 按 segments 分组
    outSegments = [{'start':0,'end':0,'words':[]} for _ in range(len(segments))]
    word_index = 0
    for i, segment in enumerate(segments):
        segment_words = []
        while word_index < len(words) and words[word_index][2] in segment['normalized']:
            segment_words.append({
                'start': words[word_index][0],
                'end': words[word_index][1],
                'word': words[word_index][2],
                'confidence': words[word_index][3].item()
            })
            word_index += 1
        if segment_words:
            outSegments[i]['start'] = segment_words[0]['start']
            outSegments[i]['end'] = segment_words[-1]['end']
            outSegments[i]['words'] = segment_words
            outSegments[i]['text'] = segment['inversed']
        else:
            # 如果没有匹配的词，使用 segment 的开始和结束时间
            outSegments[i]['start'] = 0
            outSegments[i]['end'] = 0
            outSegments[i]['text'] = segment['inversed']

    if is_output_img():
        # 创建时间轴
        duration = waveform.shape[0] / sample_rate
        xtime = np.linspace(0, duration, waveform.shape[0])

        plt.figure(figsize=(40, 12))
        plt.plot(xtime, waveform.numpy(), label='Waveform')

        # 在波形上标注对齐结果
        for start_time, end_time, token, score,i in words:
            # 绘制对齐区间
            plt.axvspan(start_time, end_time, color='green', alpha=0.3)
            # 在区间中间位置标注文本
            plt.text((start_time + end_time) / 2, np.max(waveform.numpy()) * 0.8, token,
                    horizontalalignment='center', fontsize=8, color='red')
            plt.text((start_time + end_time) / 2, -np.max(waveform.numpy()) * 0.8, f"{score:.2f}",
                    horizontalalignment='center', fontsize=6, color='red')

        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Alignment Results with Waveform')
        plt.legend()
        plt.tight_layout()

        # 显示图像
        plt.savefig('alignment.png')

    return outSegments

# 定义 FastAPI 路由
@app.post("/speech2text")
async def speech_to_text(request: Request):
    try:
        #start_time = time.time()
        data = await request.json()
        outSegments = process_audio(data)
        #print(f"Time elapsed: {time.time() - start_time}")
        return JSONResponse(content={"segments": outSegments})
    except Exception as e:
        logging.warning(str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def main():
    if is_output_img():
        # 查找系统中的中文字体
        if platform.system() == 'Linux':
            zh_font = fm.FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')  # Linux 示例
        elif platform.system() == 'Darwin':
            zh_font = fm.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')  # macOS 示例
        elif platform.system() == 'Windows':
            zh_font = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')  # Windows 示例
        else:
            raise EnvironmentError("Unsupported platform")


        plt.rcParams['font.family'] = zh_font.get_name()
        plt.rcParams['axes.unicode_minus'] = False


    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=5001)
    args = parser.parse_args()
    # 启动服务器
    uvicorn.run(app, host='0.0.0.0', port=args.port)

if __name__ == '__main__':
    main()