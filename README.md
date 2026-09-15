# YoutubeTranscoder

Encode any file into an 8K black-and-white data video, upload it to YouTube, and decode it back — byte-for-byte identical to the original. Created with Claude AI.

> **Legal notice:** This tool is for personal archival and legitimate data storage only. Do not use it to distribute copyrighted material or anything illegal. Any file you share via YouTube video should be an **encrypted archive** (e.g. a password-protected `.7z` or `.zip`) — YouTube videos are public by default, and unencrypted files expose your data to anyone who downloads them.

> **Encoding recommendation:** Use NVENC (`--nvenc`). CPU encoding is **not recommended as of now** — it produces a much larger video output with **more errors and corrupt frames**.

---

## How It Works

Each frame is a 7680×4320 (8K) image made up of 4×4 pixel blocks. Every block is either pure black or pure white — one bit per block, 1920×1080 blocks per frame. No colour, no gradients; just a binary grid that YouTube's compression pipeline cannot meaningfully corrupt.

### Redundancy: zfec erasure coding

The file is split into segments. Each segment is encoded into **M=15 shares** (frames), of which only **K=12 are needed** to reconstruct it — meaning up to 3 corrupted or missing frames per segment are tolerated.

### Integrity: double SHA-256 guard

Each frame also carries a double SHA-256 guard. If either copy mismatches, the frame is silently erased and zfec reconstructs the segment from the remaining clean shares instead.

The manifest also stores a SHA-256 checksum of the original file (along with other metadata), enabling a 3-way verification (`source == manifest == decoded`) during the round-trip self-test.

---

## Requirements

- Python (tested with 3.12)
- `ffmpeg` on PATH or in the script folder
- `yt-dlp` on PATH or in the script folder *(decode from URL only)*
- Python packages: `numpy`, `zfec`

```bash
pip install numpy zfec yt-dlp
```

Tested on CachyOS (Arch) and Windows 10.

---

## Usage

### Encode a file

```bash
python YoutubeTranscoder.py -e myfile.7z
python YoutubeTranscoder.py -e myfile.7z --nvenc        # GPU encoder
python YoutubeTranscoder.py -e myfile.7z out.mp4        # explicit output path
```

### Decode a local video

```bash
python YoutubeTranscoder.py -d out.mp4
python YoutubeTranscoder.py -d out.mp4 myfile.7z        # explicit output path
```

### Decode directly from YouTube (streaming)

```bash
python YoutubeTranscoder.py -d https://www.youtube.com/watch?v=XXXXXXXXXXX
```

No temporary file is written. `yt-dlp` pipes the video stream directly into `ffmpeg`, which feeds frames to the decoder.

The 8K stream must be available — YouTube can take minutes to hours to process 8K after upload.

### Round-trip self-test

```bash
python YoutubeTranscoder.py --trip myfile.7z
python YoutubeTranscoder.py --trip myfile.7z --nvenc
```

Encodes, then immediately decodes, then verifies the decoded file is byte-identical to the source via a 3-way SHA-256 check (`source == manifest == decoded`). Use this to verify your setup before uploading.

### Performance notes

These are rough estimates and depend on the source file, settings, hardware, and YouTube processing.

| Task | Hardware | Approx. time | Output / notes |
|---|---:|---:|---|
| NVENC encode, 300 MB source | RTX 3080 Ti | ~2 minutes | ~2 GB MP4; recommended |
| CPU encode, 300 MB source | Ryzen 9 7900X3D | ~20 minutes | 3–5 GB output; more errors and corrupt frames — not recommended |
| Decode / stream decode, 300 MB source | — | ~2–3 minutes | Local video or direct YouTube URL |

For best results, use `--nvenc` when encoding.
