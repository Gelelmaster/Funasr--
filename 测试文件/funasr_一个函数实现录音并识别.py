import pyaudio
import numpy as np
import time
import io
import wave  # 导入 wave 模块
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

def record_and_transcribe():
    """ 录制音频并直接进行转录 """
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

    # 直接将 BytesIO 对象转换为 numpy 数组进行转录
    audio_data = np.frombuffer(audio_buffer.getvalue(), dtype=np.int16)  # 使用 getvalue 读取内容
    audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max  # 归一化处理
    
    # 进行转录
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
        text = record_and_transcribe()  # 录制音频并识别
        print("识别结果:", text)  # 输出识别结果