from pathlib import Path
from faster_whisper import WhisperModel
import argparser

model_size = "medium"
audio = "audio_large.mp3"
download_root = Path.cwd() / "src"

args = argparser.GetArguments()

if (args.download_root) is None:
    args.download_root = download_root

model = WhisperModel(
    model_size_or_path=args.model_size_or_path,
    device=args.device,
    device_index=args.device_index,
    compute_type=args.compute_type,
    cpu_threads=args.cpu_threads,
    num_workers=args.num_workers,
    download_root=args.download_root,
    local_files_only=args.local_files_only,
)

segments, info = model.transcribe(audio)

print(
    "Detected language '%s' with probability %f"
    % (info.language, info.language_probability)
)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
