from funasr import AutoModel
def transcribe_audio(audio_file_path):
    """ 使用 FunASR 模型从音频文件中提取文本 """
    
    model = AutoModel(model="paraformer-zh", disable_update=True) # 初始化模型，禁用更新检查

    res = model.generate(input=audio_file_path) # 生成文本内容

    # 提取文本
    if res and isinstance(res, list):  # 确保 res 是一个列表且不为空
        for entry in res:
            if 'text' in entry:  # 检查字典中是否有 'text' 键
                text = entry['text'].replace(" ", "")  # 去掉中间的空格
                return text

    return None  # 如果没有找到文本，返回 None

if __name__ == "__main__":
    audio_path = r"D:\Desktop\project\CosyVoice\output.wav"
    text = transcribe_audio(audio_path)
    print(text)