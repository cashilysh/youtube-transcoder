# YoutubeTranscoder

Encode any file into an 8K black-and-white data video, upload it to YouTube, and decode it back — byte-for-byte identical to the original.

> **Legal notice:** This tool is for personal archival and legitimate data storage only. Do not use it to distribute copyrighted material or anything illegal. Any file you share via YouTube video should be an **encrypted archive** (e.g. a password-protected `.7z` or `.zip`) — YouTube videos are public by default, and unencrypted files expose your data to anyone who downloads them.

---

<!-- screenshot: encoding in progress -->

---

## How it works

Each frame is a 7680×4320 (8K) image made up of 4×4 pixel blocks. Every block is either pure black or pure white — one bit per block, 1920×1080 blocks per frame. No colour, no gradients; just a binary grid that YouTube's compression pipeline cannot meaningfully corrupt.

**Redundancy (zfec erasure coding)**

The file is split into segments. Each segment is encoded into **M=15 shares** (frames), of which only **K=12 are needed** to reconstruct it — meaning up to 3 corrupted or missing frames per segment are tolerated. YouTube applies two lossy passes (your HEVC upload → VP9 re-encode for some clients), and the erasure coding absorbs that damage. Each frame also carries a double SHA-256 guard: if either copy mismatches, the frame is silently erased and zfec reconstructs the segment from the remaining clean shares instead.

**Encoders**

| | CPU (x265) | GPU (hevc_nvenc) |
|---|---|---|
| Flag | *(default)* | `--nvenc` |
| Quality | CRF 20, `superfast` preset | QP 4, p4 preset |
| Notes | Slower, no GPU required | Fast, requires NVIDIA GPU |

---

## Requirements

- Python 3.12 (tested on Arch Linux and Windows)
- `ffmpeg` on PATH or in the script folder
- `yt-dlp` on PATH or in the script folder *(decode from URL only)*
- Python packages: `numpy`, `zfec`

```
pip install numpy zfec yt-dlp
```

The script will locate `ffmpeg` and `yt-dlp` automatically — checking PATH first, then falling back to the directory containing the script. It validates each binary before use, so a broken PyInstaller bundle on PATH won't silently cause failures.

---

## Usage

**Encode a file**
```
python YoutubeTranscoder.py -e myfile.7z
python YoutubeTranscoder.py -e myfile.7z --nvenc        # GPU encoder
python YoutubeTranscoder.py -e myfile.7z out.mp4        # explicit output path
```

**Decode a local video**
```
python YoutubeTranscoder.py -d out.mp4
python YoutubeTranscoder.py -d out.mp4 myfile.7z        # explicit output path
```

**Decode directly from YouTube (streaming)**
```
python YoutubeTranscoder.py -d https://www.youtube.com/watch?v=XXXXXXXXXXX
```
No temporary file is written. yt-dlp pipes the video stream directly into ffmpeg, which feeds frames to the decoder. The 8K stream must be available — YouTube can take minutes to hours to process 8K after upload.

---

<!-- screenshot: streaming decode in progress -->

---

**Round-trip self-test**
```
python YoutubeTranscoder.py --trip myfile.7z
python YoutubeTranscoder.py --trip myfile.7z --nvenc
```
Encodes, then immediately decodes, then verifies the decoded file is byte-identical to the source via a 3-way SHA-256 check (source == manifest == decoded). Use this to verify your setup before uploading.

---

## Security advice

Always encrypt your files before encoding. A password-protected archive means even if someone downloads your YouTube video and decodes it, they cannot read the contents without your key.

```
# Example: encrypt with 7-Zip before encoding
7z a -p -mhe=on archive.7z myfile.pdf
python YoutubeTranscoder.py -e archive.7z --nvenc
```

---

## Configuration

Edit the constants near the bottom of `YoutubeTranscoder.py`:

| Setting | Default | Description |
|---|---|---|
| `ZFEC_K` | 12 | Shares needed to reconstruct a segment |
| `ZFEC_M` | 15 | Total shares per segment (frames per segment) |
| `CPU_CRF` | 20 | x265 quality (lower = larger file) |
| `CPU_PRESET` | `superfast` | x265 speed preset |
| `NVENC_QP` | 4 | hevc_nvenc quality (lower = larger file, safe range 0–14) |
| `NVENC_PRESET` | `p4` | hevc_nvenc speed preset (p1–p7) |
| `ENCODE_FPS` | 30 | Output frame rate (does not affect data capacity) |
