# coding=gb2312
import numpy as np
from PIL import Image
from scipy.io import wavfile
from hilbertcurve.hilbertcurve import HilbertCurve
from pydub import AudioSegment

TARGET_SIZE = (512, 512)            # 目标分辨率：512×512适配9阶希尔伯特曲线,1024*1024适配10阶希尔伯特曲线
PIXEL_SAMPLE_NUM = 10               # 每像素采样数
SAMPLING_RATE = 44100               # 音频采样率（Hz）,这个值一般固定不动
HILBERT_ORDER = 9                   # 希尔伯特曲线的阶数,9阶时2^9=512,对应512*512分辨率
#需调整的参数
SELECTED_RESIZE_MODE = 2            # 1-等比缩放+黑边填充 | 2-直接拉伸 | 3-等比缩放+中心裁剪
INPUT_IMAGE_PATH = "sunrise.jpg"    # 替换为你的图片路径（如："test.png"、"D:/photo.jpg"）

def Resize_Image(image_path, resize_mode):
    """
    按指定方式将传入图片调整为512*512
    
    :param image_path: 图片路径
    :param resize_mode: 处理方式
                        1-等比缩放+黑边填充(无拉伸/裁剪,短边补充黑边)
                        2-直接拉伸(强制适配512*512,可能变形)
                        3-等比缩放+中心裁剪(无黑边,以最短边为边长裁剪,保留中间区域)
    """

    img_pil = Image.open(image_path).convert("RGB")
    w, h = img_pil.size
    target_w, target_h = TARGET_SIZE

    if resize_mode == 1:
        #使用LANCZOS算法进行缩放
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_scaled = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        #创建黑色背景,将缩放后的图片居中粘贴（黑边填充）
        new_image = Image.new("RGB", TARGET_SIZE, (0, 0, 0))  # 黑色背景
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        new_image.paste(img_scaled, (offset_x, offset_y))
        return new_image
    
    elif resize_mode == 2:
        return img_pil.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    
    elif resize_mode == 3:
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_scaled = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        #中心裁剪至512×512
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        return img_scaled.crop((left, top, right, bottom))
    
def Image_to_Audio(gray_image, Audio_Save_Path):
    """
    利用希尔伯特曲线映射生成音频
    
    :param gray_image: 灰度图片
    :param Audio_Save_Path: 音频保存地址
    """

    #转为numpy数组（0-255灰度值）
    gray_np = np.array(gray_image, dtype=np.float32)

    #生成一条覆盖512×512图像所有像素的希尔伯特曲线坐标序列
    hilbert = HilbertCurve(HILBERT_ORDER, 2)
    total_pixels = TARGET_SIZE[0] * TARGET_SIZE[1]
    pixel_indices = np.arange(total_pixels)
    coords = hilbert.points_from_distances(pixel_indices)
    coords = np.array(coords)  # 转为numpy数组方便索引

    # 按希尔伯特曲线顺序提取灰度值
    gray_sequence = []
    for x, y in coords:
        gray_sequence.append(gray_np[int(y), int(x)])  # numpy数组是[y,x]（行,列），对应PIL的[x,y]
    gray_sequence = np.array(gray_sequence)
    

    # 灰度值归一化为音频振幅（-1 ~ 1）
    amplitude = (gray_sequence - 127.5) / 127.5
    # 每像素采样数为10，重复每个振幅值
    audio_samples = np.repeat(amplitude, PIXEL_SAMPLE_NUM)
    # 转为16位整型（WAV格式标准，范围-32768~32767）
    audio_samples_int16 = (audio_samples * 32767).astype(np.int16)

    # 保存音频文件
    wavfile.write(Audio_Save_Path, SAMPLING_RATE, audio_samples_int16)
    print(f"音频已保存至：{Audio_Save_Path}")
    return audio_samples_int16

def Audio_to_Image(audio_path, output_image_path):
    """
    利用希尔伯特曲线逆向映射生成图片
    
    :param audio_path: 音频路径
    :param output_image_path: 生成图片路径
    """

    #读取音频文件
    sampling_rate, audio_samples_int16 = wavfile.read(audio_path)
    assert sampling_rate == SAMPLING_RATE, f"音频采样率不匹配！要求{SAMPLING_RATE}Hz,实际{sampling_rate}Hz"

    #音频振幅反归一化（转回0-255灰度值）
    audio_samples = audio_samples_int16 / 32767
    gray_sequence = audio_samples * 127.5 + 127.5
    #10个一行,对每一行内部的10个采样点计算均值,输出一个按照Hilbert曲线序列排列的灰度一维数组
    gray_sequence_restored = np.mean(gray_sequence.reshape(-1, PIXEL_SAMPLE_NUM), axis=1)

    #希尔伯特曲线逆映射为512×512图像
    hilbert = HilbertCurve(HILBERT_ORDER, 2)
    total_pixels = TARGET_SIZE[0] * TARGET_SIZE[1]
    pixel_indices = np.arange(total_pixels)
    coords = hilbert.points_from_distances(pixel_indices)
    coords = np.array(coords)

    #初始化空白灰度图数组（0=黑色）
    restored_gray_np = np.zeros(TARGET_SIZE, dtype=np.uint8)
    #按希尔伯特曲线顺序填充灰度值
    for i, (x, y) in enumerate(coords):
        if i < len(gray_sequence_restored):
            #确保坐标和灰度值在有效范围
            x_clamped = np.clip(int(x), 0, TARGET_SIZE[0]-1)
            y_clamped = np.clip(int(y), 0, TARGET_SIZE[1]-1)
            gray_val = np.clip(gray_sequence_restored[i], 0, 255).astype(np.uint8)
            restored_gray_np[y_clamped, x_clamped] = gray_val

    #保存还原的灰度图
    restored_img = Image.fromarray(restored_gray_np, mode="L")
    restored_img.save(output_image_path)
    print(f"还原图像已保存至：{output_image_path}")
    return restored_img

def mp3_to_wav(mp3_path, wav_path):
    """
    将mp3格式音频转为wav
    
    :param mp3_path: mp3文件路径
    :param wav_path: 生成wav文件路径
    """
    audio = AudioSegment.from_mp3(mp3_path)
    # 转为单声道 + 44100Hz 采样率 + 16 位编码
    audio = audio.set_channels(1).set_frame_rate(44100)
    audio.export(wav_path, format="wav", codec="pcm_s16le")
    print(f"转换完成：{mp3_path} → {wav_path}")
    return wav_path

if __name__ == "__main__":
    #调整图片
    resized_img = Resize_Image(INPUT_IMAGE_PATH, SELECTED_RESIZE_MODE)
    resized_img.save("Image_Resized.jpg")
    print("调整后的图片已保存:Image_Resized.jpg")

    #转为灰度图
    gray_img = resized_img.convert("L")

    #生成音频文件
    Image_to_Audio(gray_img, "Image.wav")

    #音频逆向还原图像
    #Audio_to_Image("Image.wav", "Image_Restored.jpg")

    #mp3转换wav
    #mp3_to_wav("qunqing.mp3", "qunqing.wav")
 