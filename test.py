from modelscope import snapshot_download
from pathlib import Path
# import torch
# import torchaudio
from funasr import AutoModel
import sounddevice as sd
import soundfile as sf
path = Path('senseVoiceSmall')
# print(path.absolute())
model_dir = snapshot_download(model_id= 'iic/SenseVoiceSmall', local_dir= str(path.absolute()))
print(model_dir)


model = AutoModel(
    model=model_dir,
    trust_remote_code=False,
    # remote_code="./model.py",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    ban_emo_unk=True
)

res,mata= model.generate(
              input='eric-angry.wav',
              cache={},
              language="yue",  # "zn", "en", "yue", "ja", "ko", "nospeech"
              use_itn=True,
              batch_size_s=60,
              merge_vad=True,  #
              merge_length_s=15,
          )
print(res)
# print(res[0]["text"])