import torch
from pyannote.audio import Pipeline

hf_token = "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"
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token)

# send pipeline to GPU (when available)

pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
diarization = pipeline("hxq_1.mp3")

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...