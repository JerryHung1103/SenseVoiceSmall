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

duration = 30  # You can change this to the desired duration

# Specify the sample rate and the filename for the WAV file
while True:
    sample_rate = 44100
    filename = "recorded_voice.wav"

    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
    sd.wait()

    # Save the recorded audio to a WAV file
    sf.write(filename, audio_data, sample_rate)
    res = model.generate(
              input=filename,
              cache={},
              language="en",  # "zn", "en", "yue", "ja", "ko", "nospeech"
              use_itn=True,
              merge_vad=True,  #
              merge_length_s=15,
          )
    print(res[0]["text"])