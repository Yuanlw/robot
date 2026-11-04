"""Download verified upstream Chinese speech assets without replacing unrelated data."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'tts'
RELEASE = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/'
FILES = {
    'kokoro-v1.1-zh.onnx': (RELEASE + 'kokoro-v1.1-zh.onnx', '859f9ded9f53be16c24857cdab3254a45da53c3afd5ba6ef134c7de3f822e326'),
    'voices-v1.1-zh.bin': (RELEASE + 'voices-v1.1-zh.bin', '14cb6186c99e4f6016871405f62046c5df863ae27465cbdc4ee08be7dd703acd'),
    'config.json': ('https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/raw/main/config.json', 'bc333efa5ce4ceff433c8c8e5d027a1eca0166001e4e4a62bea2d26ff7a46890'),
}

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in FILES.items():
        target = ROOT / name
        if target.is_file() and digest(target) == expected:
            print(f'{name}: verified')
            continue
        temporary = target.with_suffix(target.suffix + '.part')
        try:
            with urlopen(url, timeout=60) as source, temporary.open('wb') as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            if digest(temporary) != expected:
                raise RuntimeError(f'{name}: upstream content changed; checksum mismatch')
            temporary.replace(target)
            print(f'{name}: downloaded and verified')
        finally:
            temporary.unlink(missing_ok=True)
