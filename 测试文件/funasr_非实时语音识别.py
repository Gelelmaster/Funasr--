import pyaudio
import numpy as np
import time
import io
import wave  # 添加 wave 模块的导入
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 模型目录
model_dir = "iic/SenseVoiceSmall"

# 初始化模型
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",  
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# 音频设置
CHUNK = 1024  # 每个缓冲区的音频帧数
FORMAT = pyaudio.paInt16  # 采样位深，16位
CHANNELS = 1  # 单声道
RATE = 16000  # 采样率，16kHz
SILENCE_THRESHOLD = 500  # 静默检测阈值
SILENCE_DURATION = 3  # 静默持续时长 (秒)

def record_audio():
    """ 录制音频并返回音频数据 """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("开始录音，说话后将自动检测结束录音...")
    frames = []
    silent_start_time = None

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        
        # 将数据转换为 numpy 数组进行音量检测
        audio_data = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio_data).mean()

        if volume < SILENCE_THRESHOLD:
            if silent_start_time is None:
                silent_start_time = time.time()  # 记录开始静默的时间
            
            # 计算静默的持续时间
            silent_duration = time.time() - silent_start_time
            if silent_duration >= SILENCE_DURATION:
                print("检测到静默，录音结束")
                break
        else:
            silent_start_time = None  # 如果检测到非静默，重置静默计时

    # 停止录音并关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 将录音数据存储到内存中的 BytesIO 对象
    audio_buffer = io.BytesIO()
    wf = wave.open(audio_buffer, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    
    # 不需要调用 seek，因为我们将在转录时读取音频数据
    return audio_buffer  # 返回音频数据流

def transcribe_audio(audio_buffer):
    """ 使用 FunASR 模型从音频数据流中提取文本 """
    # 将 BytesIO 对象转换为 numpy 数组
    audio_data = np.frombuffer(audio_buffer.getvalue(), dtype=np.int16)  # 使用 getvalue 读取内容
    
    # 将音频数据转换为所需格式
    audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max  # 归一化处理
    res = model.generate(
        input=audio_data,
        cache={},
        language="auto",  # 根据需要设置语言
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )
    text = rich_transcription_postprocess(res[0]["text"])
    return text

if __name__ == "__main__":
    while True:
        audio_buffer = record_audio()  # 录制音频
        text = transcribe_audio(audio_buffer)  # 识别音频
        print("识别结果:", text)  # 输出识别结果