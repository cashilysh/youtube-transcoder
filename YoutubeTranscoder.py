#!/usr/bin/env python3
"""
YoutubeTranscoder.py -- encode any file into an 8K data video for YouTube
                        upload, and decode it back.  Includes an end-to-end
                        round-trip self-test mode.

Compatible with Windows, macOS, and Linux.  Requires ffmpeg and (for GPU
encoding or URL streaming) hevc_nvenc / yt-dlp in PATH.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENCODING SCHEME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2-level black & white palette (1 bit/block):
  Each 4x4 pixel block encodes exactly one bit using only pure Black or White.
  Chroma channels are never used (R=G=B for every block), so YouTube's
  YUV420p chroma-subsampling (2:1 in both dimensions) cannot corrupt data.

  Palette (luma levels, decision boundary at luma 127):
    Symbol 0 -> Black (  0,   0,   0)   luma = 0
    Symbol 1 -> White (255, 255, 255)   luma = 255

  The single decision boundary gives +/-127 luma units of noise margin --
  the maximum possible for any luma-only palette.  YouTube's Y-channel
  compression drifts uniform blocks by at most ~15 luma units, well inside
  that margin.

Border / calibration reference:
  The outer 1-block border of every frame is a fixed B&W checkerboard.
  On decode, the top-left corner block is checked: its luma must be below 64
  or above 191 (i.e. clearly Black or White).  Any other value indicates a
  corrupt or non-data frame, which is discarded without further processing.

Block size: 4x4 pixels (hardcoded).
  This is the minimum block size that survives area-averaging downscale
  during decode while keeping each block's averaged luma unambiguous.

Resolution: 8K (7680x4320), hardcoded.
  Frames are rendered at block resolution (1920x1080 blocks) and upscaled
  to 8K with nearest-neighbour interpolation before encoding.  The decoder
  always downscales to the canonical 1920x1080 block grid with area averaging
  regardless of what resolution YouTube served.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REDUNDANCY AND INTEGRITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

zfec erasure coding (configured via ZFEC_K and ZFEC_M):
  The source file is split into segments of (ZFEC_K * _CHUNK_SIZE) bytes.
  Each segment is encoded into ZFEC_M shares (frames); only ZFEC_K shares
  are needed to reconstruct the segment.  Up to (ZFEC_M - ZFEC_K) frames
  per segment can be lost or corrupted and the file is still recovered
  byte-for-byte.  Default: K=12, M=15 -> 3 bad frames per segment (~20%).

Double SHA-256 guard per data frame:
  Each data frame stores its share payload as:
    sha256(share)[32 B] || share_bytes[258,146 B] || sha256(share)[32 B]
  On decode:
    Step 1 -- if prefix != suffix, the guard fields themselves were corrupted;
              the frame is discarded immediately (before reading the payload).
    Step 2 -- if sha256(share_bytes) != prefix, the payload was corrupted;
              the frame is erased (treated as missing by zfec, not as data).
  This double-copy catches single-bit flips in either hash copy before any
  payload comparison is attempted.

Manifest frame (frame 0):
  The first frame of every encoded video is a manifest, not a data frame.
  It stores the original filename, SHA-256 of the source file, and the zfec
  K/M parameters.  The same double-copy guard is used: the payload is written
  at two spatially separated locations within the frame (left-aligned in the
  header row, and right-aligned at the bottom of the data area).  Both copies
  must be byte-identical before the manifest is accepted.  This lets the
  decoder reconstruct the original filename, verify the decoded file, and
  know the K/M values without relying on the video's filename or container.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENCODERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CPU encoder (default) -- libx265 via ffmpeg:
  Flags tuned for uniform B&W block data:
    all-intra (keyint=1), no B-frames (bframes=0), tune=stillimage,
    AQ disabled (aq-mode=0), psychovisual RD off (psy-rd=0, psy-rdoq=0),
    deblocking filter off (deblock=0,0), no lookahead (rc-lookahead=0).
  Quality/size controlled by CPU_CRF (default 28; 0 = lossless).
  Speed/size trade-off controlled by CPU_PRESET (default 'medium').
  Output is tagged full-range BT.709 (-color_range pc) so YouTube's pipeline
  does not apply a limited->full range expansion that would shift luma values
  and break the 0/255 palette thresholds.

GPU encoder (--nvenc) -- hevc_nvenc via ffmpeg:
  Frames are upscaled on the CPU (nearest-neighbour) before hwupload_cuda,
  avoiding scale_cuda whose nearest-neighbour support varies by driver.
  Flags tuned for uniform B&W block data:
    constant-QP mode (-rc constqp), all-intra (-forced-idr 1), no B-frames
    (-bf 0), NVENC pipeline buffer disabled (-surfaces 0), spatial/temporal
    AQ disabled (-spatial-aq 0 -temporal-aq 0).
  Quality controlled by NVENC_QP (default 10; safe range 0-14; above 14 risks
  luma drift exceeding the +/-127-unit margin).
  Speed controlled by NVENC_PRESET p1..p7 (default 'p4').
  Same full-range BT.709 tagging as the CPU path.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLATFORM NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Windows:  SIGTERM is not registered (unavailable); only SIGINT (Ctrl-C) is.
            ANSI cursor-up/erase sequences are enabled via the Win32 API
            (Virtual Terminal Processing).  If that fails, progress lines
            print sequentially rather than updating in place.
            stdout/stderr are reconfigured to UTF-8 with errors='replace'.
            Temp files use tempfile.mkdtemp() rather than /tmp/.
            multiprocessing uses the 'spawn' start method; the
            `if __name__ == '__main__'` guard at the bottom is required.

  macOS:    Same 'spawn' multiprocessing method as Windows.  All other
            behaviour is identical to Linux.

  Linux:    Uses 'fork' for multiprocessing (faster worker startup).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIGURATION  (edit the CONFIGURATION block near the bottom of this file)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ZFEC_K       shares required to reconstruct one segment  (default 12)
  ZFEC_M       total shares produced per segment           (default 15)
               -> (M-K)/M = ~20% of frames per segment can be lost/corrupt
  ENCODE_FPS   output frame rate                           (default 30)
               affects video duration only, not data capacity or correctness
  CPU_CRF      x265 CRF quality ceiling (default 20; 0=lossless; lower=larger file)
               Only used as hard ceiling in ABR mode; sole control in pure-CRF mode.
  CPU_PRESET   x265 preset       (default 'superfast'; faster -> larger file)
  CPU_BITRATE  Target bitrate for CPU path in bits/sec (default 80_000_000 = 80 Mbps).
               Matches NVENC QP 4 output size on 8K B&W block content.
               Set to None for pure-CRF mode (original behaviour, ~3-4x larger).
  NVENC_QP     hevc_nvenc constant QP  (default 10; safe range 0-14)
  NVENC_PRESET hevc_nvenc preset p1..p7 (default 'p4')

  DECODED_SUFFIX  suffix appended to the original filename on decode
                  (default '_from_video'; change here to rename all outputs)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python3 YoutubeTranscoder.py -e myfile.zip              # -> myfile.zip.mp4  (CPU x265)
  python3 YoutubeTranscoder.py -e myfile.zip --nvenc      # -> myfile.zip.mp4  (GPU NVENC)
  python3 YoutubeTranscoder.py -e myfile.zip out.mp4      # explicit output path
  python3 YoutubeTranscoder.py -d out.mp4                 # -> <origname>_from_video.<ext>
  python3 YoutubeTranscoder.py -d out.mp4 myfile.zip      # explicit output path
  python3 YoutubeTranscoder.py -d https://youtu.be/XXXXX  # stream directly from YouTube
  python3 YoutubeTranscoder.py -d https://youtu.be/XXXXX out.bin  # stream + explicit path
  python3 YoutubeTranscoder.py --trip myfile.zip          # encode -> decode -> verify
  python3 YoutubeTranscoder.py --trip myfile.zip --nvenc  # same, GPU encoder

Default output paths:
  -e / --encode   <input>.mp4
  -d / --decode   <original_basename>_from_video<original_ext>
                  written to the current directory when the source is a URL
  --trip          video:   <input>.mp4
                  decoded: <input>.roundtrip

Streaming decode (-d with a URL):
  yt-dlp selects the best available video-only stream and pipes it directly
  into ffmpeg.  No temporary file is written.  The best stream must be 8K
  (7680x4320); a warning is printed if YouTube has not yet made the 8K
  version available.  yt-dlp download progress (speed, ETA) is shown live.
  Audio is never downloaded (this tool stores no audio payload).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FRAME LAYOUT  (all units: 4x4 px blocks; 8K resolution hardcoded)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  8K grid = 1920x1080 blocks
  +- Outer 1-block border: fixed B&W checkerboard (calibration / sanity check)
  +- Inner 1918x1078 blocks:
       Block row 0  (header row):
         23 bytes = 184 bits -> 184 blocks, left-aligned, remainder black.
         Stores: segment_id(4B) | frame_in_seg(1B) | total_segments(4B) |
                 total_frames(4B) | file_size(8B) | k(1B) | m(1B)
         For the manifest frame, segment_id == MANIFEST_MAGIC (0x4D414E49).

       Block rows 1..1077  (data rows):
         1077 x 1918 = 2,065,686 blocks x 1 bit = 258,210 bytes capacity.
         Layout: sha256_prefix(32B) | share_payload(258,146B) | sha256_suffix(32B)
         The two 32-byte SHA-256 copies must be identical; if not, the frame
         is discarded before the payload is read.

  Manifest frame data area (same 1077x1918 block data rows):
         Copy 1 of manifest payload: left-aligned from the start of block row 1
                                     (the header row position, re-read as data).
         Copy 2 of manifest payload: right-aligned at the end of the data area,
                                     last bit at block [_ROWS-2, _COLS-2].
         All other data-area blocks are black (symbol 0).

  Manifest payload structure (8 + N + 32 bytes, N <= 200):
         Offset  Size  Field
              0     4  Magic = 0x4D414E49 ("MANI")
              4     2  filename_len: byte length of UTF-8 filename (uint16 BE)
              6     1  k: zfec K parameter
              7     1  m: zfec M parameter
              8     N  filename: UTF-8 encoded original basename (no path)
            8+N    32  sha256: SHA-256 digest of the original file content
"""

import sys, os, struct, math, subprocess, argparse, signal, time, hashlib
import tempfile, platform
import numpy as np
import zfec
from multiprocessing import Pool, cpu_count, get_context

# ── platform helpers ──────────────────────────────────────────────────────────
_IS_WINDOWS = platform.system() == 'Windows'

# Ensure UTF-8 output on Windows so Unicode status characters display correctly.
if _IS_WINDOWS:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass  # Python < 3.7 -- best-effort

# Persistent temp dir for ffmpeg log files (survives for the process lifetime).
_TMPDIR = tempfile.mkdtemp(prefix='YoutubeTranscoder_')

def _tmp(name):
    """Return a platform-safe temp file path inside our session directory."""
    return os.path.join(_TMPDIR, name)


def _ansi_cursor_up(n):
    """Move the terminal cursor up n lines (ANSI ESC[nA).
    Falls back to a blank print on Windows where ANSI may not be available."""
    if _IS_WINDOWS:
        # Enable Virtual Terminal Processing if possible (Windows 10+).
        # If that fails we just print newlines — no in-place overwrite.
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            sys.stdout.write(f"\x1b[{n}A")
            return
        except Exception:
            pass
        # Fallback: don't try to move cursor; caller will just print below.
        return
    sys.stdout.write(f"\x1b[{n}A")


def _ansi_erase_line():
    """Erase to end-of-line (ANSI ESC[K). No-op on unsupported terminals."""
    if _IS_WINDOWS:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            sys.stdout.write("\x1b[K")
            return
        except Exception:
            pass
        return
    sys.stdout.write("\x1b[K")

# ── decode output naming ──────────────────────────────────────────────────────
# When the decoder reads the original filename from the manifest frame it
# appends this suffix before writing the output file.  Change here to rename
# all decoded files globally.
DECODED_SUFFIX = '_from_video'

# ── constants ────────────────────────────────────────────────────────────
BLOCK          = 4          # pixels per data block (4x4), hardcoded
W8K, H8K       = 7680, 4320 # canonical encoding resolution, hardcoded

HEADER_FMT     = '>IBIIQBB'
HEADER_BYTES   = struct.calcsize(HEADER_FMT)   # 23
HEADER_BITS    = HEADER_BYTES * 8              # 184
BITS_PER_BLOCK = 1          # 2-level black & white palette -> 1 bit/block

# ── per-frame double SHA-256 guard ───────────────────────────────────────────
# Each data frame encodes:
#   sha256(share_bytes) [32 B prefix] || share_bytes || sha256(share_bytes) [32 B suffix]
# On decode both 32-byte fields are read:
#   1. If prefix != suffix  → the hash fields themselves are corrupted; discard frame.
#   2. If sha256(share_bytes) != prefix → payload corrupted; erase frame for zfec.
# The double-copy means a single bit flip in a hash field is caught by the
# mismatch between the two copies before any payload comparison is attempted.
FRAME_SHA_BYTES = 32   # bytes per SHA-256 copy (two copies per data frame)
FRAME_SHA_GUARD = FRAME_SHA_BYTES * 2   # total bytes consumed by both guards

# 2-level black & white: luma 0 (Black) / 255 (White), 255 units apart.
# Single decision boundary at luma 127 — ±127 units of margin.
# Pure luma (R=G=B) means chroma subsampling has zero effect on decoding.
PALETTE = np.array([
    [  0,   0,   0],   # 0 Black
    [255, 255, 255],   # 1 White
], dtype=np.uint8)
PALETTE_LUMA = np.array([0, 255], dtype=np.int32)  # Y values for fast 1D classify
PALETTE_GRAY = np.array([0, 255], dtype=np.uint8)  # single-channel encode output

# Precomputed: BITS_PER_BLOCK-bit representation of each symbol, big-endian
SYM_BITS = np.array([[int(b) for b in f'{i:0{BITS_PER_BLOCK}b}']
                     for i in range(2 ** BITS_PER_BLOCK)], dtype=np.uint8)


# ── fixed geometry (encoder and decoder always use this layout) ────────────
# Hardcoded to 8K / BLOCK=4.  If the actual video is at a different
# resolution the decoder still decodes against this canonical layout.
_COLS        = W8K // BLOCK          # 1920
_ROWS        = H8K // BLOCK          # 1080
_INNER_COLS  = _COLS - 2             # 1918
_INNER_ROWS  = _ROWS - 2             # 1078
_DATA_ROWS   = _INNER_ROWS - 1       # 1077
_BLOCKS_PER_FRAME = _DATA_ROWS * _INNER_COLS              # 2,065,686
# Total capacity of the data area in bytes.
_FRAME_DATA_BYTES = (_BLOCKS_PER_FRAME * BITS_PER_BLOCK) // 8   # 258,210
# Usable share payload per frame: total capacity minus 64 bytes for the two
# SHA-256 guards (32-byte prefix + 32-byte suffix around the share payload).
# Layout: sha256_prefix(32) || share_bytes(_CHUNK_SIZE) || sha256_suffix(32)
_CHUNK_SIZE  = _FRAME_DATA_BYTES - FRAME_SHA_GUARD         # 258,146 bytes/frame
_HEADER_BLOCKS = math.ceil(HEADER_BITS / BITS_PER_BLOCK)   # 184

# ── manifest frame ────────────────────────────────────────────────────────────
# Frame 0 of every encoded video is a manifest frame.  It carries the original
# filename, SHA-256 of the source file, and the zfec K/M parameters so the
# decoder can:
#   (a) reconstruct the correct output filename without relying on the video
#       filename,
#   (b) verify the decoded payload byte-for-byte against the embedded digest,
#   (c) know the redundancy level (K/M) without relying solely on per-frame
#       headers — enabling cross-compatible decode across different encode runs.
#
# The manifest payload is written TWICE in spatially separated locations,
# mirroring the double SHA-256 guard used by every data frame (prefix at the
# start of the data area, suffix at the end).  On decode both copies are read
# from their respective locations and compared; only if they are identical are
# the embedded filename and SHA-256 accepted.  A single-bit flip in either
# copy causes the two copies to differ, triggering rejection before any
# filename or hash is trusted.
#
# Layout — single-copy payload structure (8+N+32 bytes, N≤200):
#
#   Offset           Size  Field
#   ──────           ────  ─────────────────────────────────────────────────────
#      0               4   Magic = MANIFEST_MAGIC (0x4D414E49 = "MANI")
#      4               2   filename_len: length of UTF-8 filename (uint16 BE)
#      6               1   k: zfec K parameter (shares required per segment)
#      7               1   m: zfec M parameter (shares produced per segment)
#      8             N≤200 filename: UTF-8 encoded original basename (no path)
#    8+N              32   sha256: SHA-256 of the original file content
#
# The 8K resolution and block/grid layout are NOT stored in the manifest —
# they remain HARDCODED constants (W8K, H8K, BLOCK) so the canonical frame
# geometry is always known without reading it from the stream.
#
# Placement of the two copies within the frame:
#
#   Copy 1 (prefix) — inner header row (grid row 1), LEFT-ALIGNED from col 1.
#                     Reads left-to-right from the beginning of the row.
#                     Capacity: 1918 blocks = 239 bytes (payload ≤ 240 bytes ✓)
#
#   Copy 2 (suffix) — bottom-right corner of the inner data area.
#                     The 1077×1918-block data area is treated as a flat
#                     row-major sequence; the payload is RIGHT-ALIGNED so
#                     its last bit lands at the very last inner data block
#                     (grid[_ROWS-2, _COLS-2] = bottom-right corner).
#                     Mirrors data-frame layout: sha_prefix at start of data,
#                     sha_suffix at end of data.
#
# On decode the manifest is the first recognised valid frame.  Its segment_id
# field (first 4 bytes of the inner header) will equal MANIFEST_MAGIC, which
# the decoder checks before handing frames to zfec.
MANIFEST_MAGIC    = 0x4D414E49   # "MANI" -- marks a manifest frame
MANIFEST_HDR_FMT  = '>IHBB'      # magic (4B) + filename_len (2B) + k (1B) + m (1B)
MANIFEST_HDR_SIZE = struct.calcsize(MANIFEST_HDR_FMT)   # 8
MANIFEST_SHA_SIZE = 32           # SHA-256 output bytes
MANIFEST_MAX_NAME = 200          # max UTF-8 filename bytes (fits 239-byte header row)


# ── graceful shutdown (SIGINT / SIGTERM) ──────────────────────────────────
_cleanups = []


class _Cleanup:
    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        _cleanups.append(self.fn)
        return self

    def __exit__(self, *exc):
        try:
            _cleanups.remove(self.fn)
        except ValueError:
            pass


def _handle_signal(signum, frame):
    name = signal.Signals(signum).name
    print(f"\nInterrupted ({name}) -- cleaning up...", file=sys.stderr)
    for fn in reversed(_cleanups):
        try:
            fn()
        except Exception:
            pass
    os._exit(128 + signum)


def install_signal_handlers():
    # SIGTERM is not available on Windows; SIGINT (Ctrl-C) always is.
    sigs = [signal.SIGINT]
    if hasattr(signal, 'SIGTERM'):
        sigs.append(signal.SIGTERM)
    for sig in sigs:
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            pass


def fmt_hms(seconds):
    """Format a duration in seconds as HH:MM:SS."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================================
#  ENCODE
# ============================================================================

def make_border():
    """
    Build the canonical 1920x1080 block border grid (B&W, 2D).
    Alternating Black/White checkerboard on all four edges.
    Called once and reused for every frame.  Single channel matches the
    gray pix_fmt sent to ffmpeg (1 byte/block vs 3 for rgb24).
    """
    # Use symbols 0 (Black/0) and 1 (White/255) — the only two palette
    # values, which sit on opposite sides of the decision boundary (127) and
    # both clear the decoder's sanity check (luma < 64 or > 191).
    g = np.zeros((_ROWS, _COLS), dtype=np.uint8)
    for c in range(_COLS):
        g[0, c]         = PALETTE_GRAY[1] if  c           % 2 == 0 else PALETTE_GRAY[0]
        g[_ROWS-1, c]   = PALETTE_GRAY[1] if (_ROWS-1+c)  % 2 == 0 else PALETTE_GRAY[0]
    for r in range(1, _ROWS-1):
        g[r, 0]         = PALETTE_GRAY[1] if  r            % 2 == 0 else PALETTE_GRAY[0]
        g[r, _COLS-1]   = PALETTE_GRAY[1] if (r+_COLS-1)   % 2 == 0 else PALETTE_GRAY[0]
    return g


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of a bytes object."""
    return hashlib.sha256(data).hexdigest()


def build_manifest_frame(orig_filename, file_sha256_hex, border, k, m):
    """
    Render the manifest frame as a raw greyscale byte string at block
    resolution (1920x1080 blocks).

    orig_filename   – basename of the original source file (str)
    file_sha256_hex – 64-char hex SHA-256 of the source file content
    border          – pre-built border grid from make_border()
    k               – zfec K parameter stored in the manifest (cross-compat)
    m               – zfec M parameter stored in the manifest (cross-compat)

    The manifest payload is written in TWO spatially separated locations,
    mirroring the double SHA-256 guard used by data frames:

      Copy 1 -- inner header row (grid row 1), LEFT-ALIGNED from column 1.
               Same position as before; acts as the "prefix" guard.

      Copy 2 -- bottom-right corner of the inner data area (last inner rows),
               RIGHT-ALIGNED ending at column _INNER_COLS.  Laid out in
               row-major order so the last bit of the payload lands at the
               very last inner data block (grid[_ROWS-2, _COLS-2]).
               Acts as the "suffix" guard, exactly like data frames.

    On decode both copies are read from their respective positions and compared;
    only if they are byte-identical are the embedded filename and SHA-256
    accepted.  A bit-flip in either copy causes the two to differ and the
    manifest is rejected.  Inner data rows not used by copy 2 are left black.

    The 8K resolution and block layout are intentionally NOT stored here --
    they remain hardcoded constants so the canonical grid is always implicit.

    Raises ValueError if the filename is too long to encode.
    """
    name_bytes = orig_filename.encode('utf-8')
    if len(name_bytes) > MANIFEST_MAX_NAME:
        raise ValueError(
            f"Filename too long for manifest: {len(name_bytes)} bytes "
            f"(max {MANIFEST_MAX_NAME}).  Use a shorter filename.")

    sha_raw = bytes.fromhex(file_sha256_hex)   # 32 bytes

    # Single copy of the manifest payload (now includes k and m)
    single = (struct.pack(MANIFEST_HDR_FMT, MANIFEST_MAGIC, len(name_bytes), k, m)
              + name_bytes
              + sha_raw)

    grid = border.copy()

    single_bits = np.unpackbits(np.frombuffer(single, dtype=np.uint8))
    n_blocks    = math.ceil(len(single_bits) / BITS_PER_BLOCK)
    syms        = bits_to_symbols(single_bits, n_blocks)

    # ── Copy 1: header row (row 1), left-aligned ──────────────────────────
    n_place = min(n_blocks, _INNER_COLS)
    grid[1, 1:1+n_place] = PALETTE_GRAY[syms[:n_place]]
    if n_place < _INNER_COLS:
        grid[1, 1+n_place:1+_INNER_COLS] = PALETTE_GRAY[0]

    # ── Copy 2: bottom-right corner of inner data area, right-aligned ─────
    # The inner data rows span grid rows 2 .. _ROWS-2 (inclusive), giving
    # _DATA_ROWS rows × _INNER_COLS cols of data blocks.  We place the payload
    # right-aligned so the LAST bit lands at the very last data block
    # (grid[_ROWS-2, _COLS-2]).  Blocks to the left / above remain black.
    total_data_blocks = _DATA_ROWS * _INNER_COLS
    data_flat = np.zeros(total_data_blocks, dtype=np.uint8)   # all black
    n_place2  = min(n_blocks, total_data_blocks)
    # Right-align: fill the last n_place2 positions with the payload symbols.
    data_flat[total_data_blocks - n_place2:] = PALETTE_GRAY[syms[:n_place2]]
    grid[2:2+_DATA_ROWS, 1:1+_INNER_COLS] = data_flat.reshape(_DATA_ROWS, _INNER_COLS)

    return grid.tobytes()


def parse_manifest_frame(frame_grey):
    """
    Try to parse a full block-resolution frame as a manifest frame.

    frame_grey: (_ROWS, _COLS) uint8 -- full frame at block resolution

    The manifest payload is stored in TWO spatially separated locations:

      Copy 1 -- inner header row (frame_grey[1, 1:1+_INNER_COLS]), left-aligned.
      Copy 2 -- bottom-right corner of the inner data area (frame_grey[2:2+_DATA_ROWS,
               1:1+_INNER_COLS] flattened), right-aligned so the last bit lands
               at the very last inner data block.

    Both copies are decoded independently.  Only if they are byte-identical are
    the embedded filename, SHA-256, k, and m returned.  A bit flip in either
    copy causes the two to differ and None is returned.

    Returns (orig_filename, sha256_hex, k, m) on success, or None on any failure.
    Never raises.
    """
    def _decode_from_bits_at(bits, start_bit, single_size):
        """
        Decode one manifest copy from flat bit array 'bits' starting at
        'start_bit'.  'single_size' is the expected payload length in bytes
        (already known from copy 1 or computed for copy 1 on the fly).
        Returns (orig_filename, sha256_hex, k, m, name_len) or None.
        """
        total_needed = start_bit + single_size * 8
        if len(bits) < total_needed:
            return None
        hdr_bytes = np.packbits(bits[start_bit:start_bit + MANIFEST_HDR_SIZE * 8]).tobytes()
        try:
            magic, name_len, k_val, m_val = struct.unpack(MANIFEST_HDR_FMT, hdr_bytes)
        except struct.error:
            return None
        if magic != MANIFEST_MAGIC:
            return None
        if name_len == 0 or name_len > MANIFEST_MAX_NAME:
            return None
        if k_val == 0 or m_val == 0 or k_val >= m_val or m_val > 255:
            return None
        payload_bytes = np.packbits(bits[start_bit:total_needed]).tobytes()
        off      = MANIFEST_HDR_SIZE
        name_raw = payload_bytes[off:off + name_len]
        sha_raw  = payload_bytes[off + name_len:off + name_len + MANIFEST_SHA_SIZE]
        try:
            orig_filename = name_raw.decode('utf-8')
        except UnicodeDecodeError:
            return None
        return orig_filename, sha_raw.hex(), k_val, m_val, name_len

    try:
        # ── Copy 1: header row (row 1), left-aligned ──────────────────────
        # Probe to discover name_len first, then decode fully.
        hdr_grey = frame_grey[1, 1:1+_INNER_COLS].astype(np.int32)
        hdr_syms = classify_luma(hdr_grey)
        hdr_bits = symbols_to_bits(hdr_syms)

        # Peek at the header to get name_len without committing to single_size.
        if len(hdr_bits) < MANIFEST_HDR_SIZE * 8:
            return None
        hdr_peek = np.packbits(hdr_bits[:MANIFEST_HDR_SIZE * 8]).tobytes()
        try:
            magic_peek, name_len_peek, k_peek, m_peek = struct.unpack(MANIFEST_HDR_FMT, hdr_peek)
        except struct.error:
            return None
        if magic_peek != MANIFEST_MAGIC:
            return None
        if name_len_peek == 0 or name_len_peek > MANIFEST_MAX_NAME:
            return None
        single_size = MANIFEST_HDR_SIZE + name_len_peek + MANIFEST_SHA_SIZE

        copy1 = _decode_from_bits_at(hdr_bits, 0, single_size)
        if copy1 is None:
            return None
        orig_filename1, sha256_hex1, k1, m1, _ = copy1

        # ── Copy 2: bottom-right corner of inner data area, right-aligned ─
        # The encoder placed the payload right-aligned in the flat data area
        # (row-major).  We know single_size from copy 1, so the start bit is
        # simply (total_data_bits - single_size * 8).
        data_grey = frame_grey[2:2+_DATA_ROWS, 1:1+_INNER_COLS].ravel().astype(np.int32)
        data_syms = classify_luma(data_grey)
        data_bits = symbols_to_bits(data_syms)

        total_data_bits = len(data_bits)
        start_bit_copy2 = total_data_bits - single_size * 8
        if start_bit_copy2 < 0:
            return None
        copy2 = _decode_from_bits_at(data_bits, start_bit_copy2, single_size)
        if copy2 is None:
            return None
        orig_filename2, sha256_hex2, k2, m2, _ = copy2

        # Both copies must be identical — any mismatch signals corruption.
        if (orig_filename1 != orig_filename2 or sha256_hex1 != sha256_hex2
                or k1 != k2 or m1 != m2):
            return None   # bit-flip detected between the two copies

        return orig_filename1, sha256_hex1, k1, m1
    except Exception:
        return None


def bits_to_symbols(bits, n_blocks):
    """Pack bits into BITS_PER_BLOCK-bit symbols.  Pads with zeros if short."""
    needed = n_blocks * BITS_PER_BLOCK
    if len(bits) < needed:
        bits = np.concatenate([bits, np.zeros(needed - len(bits), dtype=np.uint8)])
    bits = bits[:needed].reshape(n_blocks, BITS_PER_BLOCK)
    weights = (1 << np.arange(BITS_PER_BLOCK - 1, -1, -1, dtype=np.uint8))
    return (bits * weights).sum(axis=1).astype(np.uint8)


def render_frame(args):
    """
    Render one frame into a raw greyscale byte string at block resolution
    (1920x1080 blocks = 7680x4320 pixels after GPU upscale).
    Single channel (gray) -- 1 byte per block, 1/3 the pipe volume of rgb24.

    args = (segment_id, frame_in_seg, total_segments, total_frames,
            file_size, k, m, share_bytes, border)
    """
    (segment_id, frame_in_seg, total_segments, total_frames,
     file_size, k, m, share_bytes, border) = args

    grid = border.copy()

    # ── header row (inner row 0) ──
    hdr = struct.pack(HEADER_FMT,
                      segment_id, frame_in_seg, total_segments, total_frames,
                      file_size, k, m)
    hdr_bits = np.unpackbits(np.frombuffer(hdr, dtype=np.uint8))
    hdr_syms = bits_to_symbols(hdr_bits, _HEADER_BLOCKS)
    n_hdr    = min(_HEADER_BLOCKS, _INNER_COLS)
    grid[1, 1:1+n_hdr] = PALETTE_GRAY[hdr_syms[:n_hdr]]
    if n_hdr < _INNER_COLS:
        grid[1, 1+n_hdr:1+_INNER_COLS] = PALETTE_GRAY[0]

    # ── data rows ──
    # Layout: sha256_prefix(32) || share_bytes(_CHUNK_SIZE) || sha256_suffix(32)
    # Both hash copies are identical (sha256 of share_bytes).  On decode:
    #   if prefix != suffix  → hash field itself corrupted, discard frame
    #   if sha256(payload) != prefix → payload corrupted, erase for zfec
    share_hash = hashlib.sha256(share_bytes).digest()   # 32 bytes
    payload    = share_hash + share_bytes + share_hash  # 32 + _CHUNK_SIZE + 32 = _FRAME_DATA_BYTES
    share_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    data_syms  = bits_to_symbols(share_bits, _DATA_ROWS * _INNER_COLS)
    grid[2:2+_DATA_ROWS, 1:1+_INNER_COLS] = PALETTE_GRAY[data_syms].reshape(_DATA_ROWS, _INNER_COLS)

    return grid.tobytes()


def encode(input_path, output_path, fps=30, k=10, m=15,
           crf=20, preset='superfast', use_nvenc=False, qp=4,
           nvenc_preset='p4', bitrate=None):
    _init_tools(need_ytdlp=False)
    if k >= m:   raise ValueError(f"k ({k}) must be < m ({m})")
    if m > 255:  raise ValueError("m cannot exceed 255 (zfec limit)")

    redundancy_pct = (m - k) / m * 100.0
    overhead_pct   = (m - k) / k * 100.0
    print(f"Resolution  : {W8K}x{H8K}  ->  block grid {_COLS}x{_ROWS}  (block={BLOCK}px)")
    print(f"Bits/block  : {BITS_PER_BLOCK}  (2-level black & white palette)")
    print(f"Chunk size  : {_CHUNK_SIZE:,} bytes/frame  |  Header: {HEADER_BYTES} bytes  |  SHA-256 guard: {FRAME_SHA_BYTES}x2 bytes/frame")
    if use_nvenc:
        safe = "safe" if qp <= 14 else "RISKY -- may corrupt data after YouTube VP9 re-encode"
        print(f"Encoder     : hevc_nvenc (GPU)  |  preset: {nvenc_preset}  |  QP: {qp}  ({safe})")
    else:
        if bitrate is not None:
            print(f"Encoder     : x265 (CPU)  |  preset: {preset}  |  ABR: {bitrate//1_000_000} Mbps  (CRF ceiling: {crf})")
        else:
            print(f"Encoder     : x265 (CPU)  |  preset: {preset}  |  CRF: {crf}")
    print(f"Redundancy  : k={k} m={m}  (~{redundancy_pct:.1f}% of frames/segment can be "
          f"lost or corrupted; {m/k:.2f}x data, ~{overhead_pct:.1f}% storage overhead)")

    with open(input_path, 'rb') as f:
        raw = f.read()
    file_size         = len(raw)
    orig_basename     = os.path.basename(input_path)
    file_sha256       = sha256_bytes(raw)
    segment_data_size = k * _CHUNK_SIZE
    pad_len           = (-file_size) % segment_data_size
    padded            = raw + b'\x00' * pad_len
    total_segments    = len(padded) // segment_data_size
    total_frames      = total_segments * m   # does not count the manifest frame
    duration_s        = total_frames / fps

    print(f"File        : {file_size:,} bytes  ({file_size/1024**2:.2f} MB)")
    print(f"SHA-256     : {file_sha256}")
    print(f"Filename    : {orig_basename}")
    print(f"Segments    : {total_segments}  x  {segment_data_size:,} bytes")
    print(f"Frames      : 1 manifest + {total_frames} data  ({duration_s:.1f}s @ {fps}fps)")

    border = make_border()

    # Input to ffmpeg: block-resolution greyscale (gray) over stdin.
    # Block res is 1920×1080; ffmpeg upscales to 7680×4320 with nearest-neighbor.
    # Sending gray (1 byte/px) instead of rgb24 (3 bytes/px) cuts pipe throughput by 2/3.
    input_args = [
        _FFMPEG, '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{_COLS}x{_ROWS}',
        '-pix_fmt', 'gray', '-r', str(fps),
        '-i', 'pipe:0',
    ]

    if use_nvenc:
        # GPU path: gray → yuv420p (CPU) → scale (CPU, nearest-neighbour to 8K)
        #           → hwupload_cuda → hevc_nvenc.
        #
        # We upscale on the CPU with scale=flags=neighbor BEFORE hwupload so
        # that scale_cuda is not needed.  scale_cuda's nearest-neighbour
        # support is driver-version-dependent and has caused silent quality
        # regressions; the CPU scaler is rock-solid and the upscale cost is
        # negligible compared to encode time for this content (all-intra,
        # static black-and-white blocks).
        #
        # Filter order matters:
        #   format=yuv420p   gray → yuv420p (flat Cb/Cr at 128); luma unchanged.
        #   scale=…          nearest-neighbour CPU upscale 1920×1080 → 7680×4320.
        #   hwupload_cuda    transfer the 8K yuv420p frame to GPU VRAM.
        #   (hevc_nvenc takes it from there)
        #
        # NVENC-specific encoder flags (hevc_nvenc only — x265 names differ):
        #   -rc constqp          constant QP mode; no VBR/CBR rate logic.
        #   -qp N                10-14 is safe: palette spacing is 255 luma
        #                        units; constqp+HEVC drifts uniform blocks by
        #                        ≲10 units at QP 14, well inside ±127 margin.
        #   -g 1                 every frame is an IDR (all-intra).
        #   -forced-idr 1        force IDR (not just I-frame) on every keyframe.
        #   -bf 0                no B-frames.
        #   -surfaces 0          disable NVENC internal frame pipeline buffering
        #                        (reduces latency and VRAM; safe for all-intra).
        #   -spatial-aq 0        spatial adaptive quantization is counter-
        #   -temporal-aq 0       productive for uniform block data; disable.
        #   -no-scenecut 1       DOES NOT EXIST in hevc_nvenc — omitted.
        #   -rc-lookahead 0      NVENC lookahead requires B-frames; omit when
        #                        bf=0 (some driver versions reject it explicitly).
        #
        # VP9 compatibility: YouTube keeps the original HEVC for capable
        # clients and re-encodes VP9 as fallback.  At QP 10-14 the output
        # bitrate for 8K all-intra HEVC is ~60-120 Mbps, safely above
        # YouTube's 8K VP9 target (~50 Mbps), so the VP9 re-encode has
        # enough signal to preserve the ±127-unit luma margins.
        vf = (f'format=yuv420p,'
              f'scale={W8K}:{H8K}:flags=neighbor,'
              f'hwupload_cuda')
        cmd = input_args + [
            '-vf', vf,
            '-c:v', 'hevc_nvenc',
            '-preset', nvenc_preset,
            '-rc', 'constqp',
            '-qp', str(qp),
            # All-intra: -forced-idr 1 makes every frame an IDR.
            # DO NOT set -g 1: NVENC requires GOP length > B-frames + 1,
            # so -g 1 with -bf 0 fails with "Gop Length should be greater
            # than number of B frames + 1" (rc=234).  Omitting -g lets
            # NVENC use its default GOP; -forced-idr 1 overrides that
            # per-frame, producing a fully all-intra stream in practice.
            '-forced-idr', '1',
            '-bf', '0',
            '-surfaces', '0',
            '-spatial-aq', '0',
            '-temporal-aq', '0',
            # Tag output as full-range BT.709 (see CPU path comment above).
            '-color_range', 'pc',
            '-colorspace', 'bt709',
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-tag:v', 'hvc1',
            '-movflags', '+faststart',
            output_path,
        ]
    else:
        # CPU path: gray input → nearest-neighbor upscale → x265.
        # yuv420p: luma-only data, flat chroma planes, subsampling harmless.
        #
        # x265-params tuned for uniform B&W block data AND size parity
        # with the NVENC path:
        #
        #   keyint=1          all-intra stream; frames are independent, mirrors
        #                     NVENC -forced-idr 1.
        #   no-b-adapt=1      disable B-frame adaptive decisions (no B-frames).
        #   bframes=0         no B-frames (matches NVENC -bf 0).
        #   aq-mode=0         disable adaptive quantisation — AQ redistributes
        #                     bits toward "texture" regions; our frames are
        #                     uniform blocks so AQ wastes bits (mirrors NVENC
        #                     -spatial-aq 0 -temporal-aq 0).
        #   psy-rd=0          disable psychovisual RD optimisation — psy-rd and
        #   psy-rdoq=0        psy-rdoq add grain/detail to fool the eye; useless
        #                     for B&W data and inflate file size.
        #   deblock=0,0       disable the deblocking filter — it softens sharp
        #                     block edges to reduce perceived artefacts, but
        #                     those sharp edges ARE the signal; filtering wastes
        #                     bits fighting its own output.
        #   rc-lookahead=0    no lookahead needed for all-intra; saves memory.
        #   scenecut=0        no scene-cut detection needed for all-intra.
        #   weightp=0         disable weighted prediction (inter only; no-op for
        #                     all-intra but suppresses a log warning).
        #   rect=0            disable rectangular partitions — x265 'medium'
        #   amp=0             and above explore rect/amp CU splits extensively
        #                     for natural content; for 4×4-aligned binary blocks
        #                     they never win and cost significant encode time,
        #                     inflating output size vs NVENC.
        #   max-merge=1       limit merge candidates to 1; merge optimisation is
        #                     irrelevant for all-intra and wastes RD budget.
        #
        # Preset choice: 'superfast' skips the expensive analyses (rect, amp,
        # full RDO mode decisions) that bloat file size on uniform block content
        # while still applying the x265 entropy coder properly.  Combined with
        # CRF 20 this closely matches NVENC QP 4 output size in practice.
        # 'medium' or slower presets INCREASE file size here because they spend
        # more bits on RD optimisations that yield no quality gain on binary data.
        x265_params = (
            'keyint=1:no-b-adapt=1:bframes=0'
            ':aq-mode=0'
            ':psy-rd=0:psy-rdoq=0'
            ':deblock=0,0'
            ':rc-lookahead=0'
            ':scenecut=0'
            ':weightp=0'
            ':rect=0:amp=0'
            ':max-merge=1'
        )
        if crf == 0:
            x265_params += ':lossless=1'
        cpu_cmd = input_args + [
            '-vf', f'scale={W8K}:{H8K}:flags=neighbor',
            '-vcodec', 'libx265',
            '-preset', preset,
            '-pix_fmt', 'yuv420p',
        ]
        if bitrate is not None and crf != 0:
            # Constrained ABR mode: target the given bitrate with CRF as a
            # quality ceiling.  x265 will never exceed what CRF would produce,
            # but is further capped to the target bitrate.  This matches NVENC
            # output size on B&W block content where pure CRF over-allocates bits.
            # -b:v sets the target; -maxrate/-bufsize enforce a hard ceiling so
            # the muxer stream never spikes above the target (a 1-second buffer
            # is sufficient for all-intra content where every frame is independent).
            cpu_cmd += [
                '-b:v', str(bitrate),
                '-maxrate', str(bitrate),
                '-bufsize', str(bitrate),   # 1-second VBV buffer at target rate
                '-crf', str(crf),           # quality ceiling -- x265 never exceeds this
            ]
        else:
            cpu_cmd += ['-crf', str(crf)]
        cpu_cmd += [
            '-x265-params', x265_params,
            # Tag output as full-range BT.709 so that ffmpeg (and YouTube's
            # pipeline) never apply a limited→full range expansion on decode.
            # Without this, YouTube re-tags the stream as BT.709 limited-range
            # and any consumer that honours the tag (including our own decoder)
            # will shift Y values by (Y-16)*255/219, breaking all thresholds.
            '-color_range', 'pc',
            '-colorspace', 'bt709',
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-tag:v', 'hvc1',
            '-movflags', '+faststart',
            output_path,
        ]
        cmd = cpu_cmd

    ffmpeg_log = open(_tmp('ffmpeg_encode.log'), 'w', errors='replace')
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=ffmpeg_log)

    enc = zfec.Encoder(k, m)

    def _read_ffmpeg_log(n_tail=15):
        """Flush and return the last n_tail lines of the ffmpeg log."""
        try:
            ffmpeg_log.flush()
        except Exception:
            pass
        try:
            with open(_tmp('ffmpeg_encode.log'), errors='replace') as fh:
                lines = fh.read().strip().splitlines()
            return '\n  '.join(lines[-n_tail:]) if lines else '(no output)'
        except OSError:
            return '(log unreadable)'

    def _die_on_ffmpeg_exit(during=''):
        """If ffmpeg has exited, print its log and call sys.exit(1)."""
        if proc.poll() is not None:
            tag = f" (during {during})" if during else ""
            print(f"\nFFmpeg exited early (rc={proc.returncode}){tag}:\n"
                  f"  {_read_ffmpeg_log()}", flush=True)
            sys.exit(1)

    def cleanup():
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass
        try:
            ffmpeg_log.close()
        except Exception:
            pass
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass

    def _write_frame(data, label=''):
        """Write one frame to ffmpeg stdin; on BrokenPipeError show the log."""
        try:
            proc.stdin.write(data)
        except BrokenPipeError:
            # ffmpeg died during the write — wait briefly for it to flush stderr
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"\nBroken pipe{' (' + label + ')' if label else ''} -- "
                  f"ffmpeg rc={proc.returncode}:\n  {_read_ffmpeg_log()}", flush=True)
            sys.exit(1)

    frame_counter = 0
    start_time = time.time()
    with _Cleanup(cleanup):
        # ── frame 0: manifest (filename + SHA-256 + k/m, written twice) ──
        manifest_bytes = build_manifest_frame(orig_basename, file_sha256, border, k, m)
        _die_on_ffmpeg_exit('pre-manifest')
        _write_frame(manifest_bytes, 'manifest')
        frame_counter += 1   # counts toward total for progress display

        for seg_idx in range(total_segments):
            seg_start = seg_idx * segment_data_size
            seg_bytes = padded[seg_start:seg_start + segment_data_size]
            pieces    = [seg_bytes[i*_CHUNK_SIZE:(i+1)*_CHUNK_SIZE] for i in range(k)]
            shares    = enc.encode(pieces)
            for share_idx, share in enumerate(shares):
                _die_on_ffmpeg_exit(f'seg={seg_idx} share={share_idx}')
                frame_data = render_frame(
                    (seg_idx, share_idx, total_segments, total_frames,
                     file_size, k, m, bytes(share)[:_CHUNK_SIZE], border))
                _write_frame(frame_data, f'seg={seg_idx} share={share_idx}')
                frame_counter += 1
                if frame_counter % 10 == 0 or frame_counter == total_frames:
                    pct = frame_counter / total_frames * 100
                    cur_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                    exp_size = (cur_size * total_frames / frame_counter) if cur_size else 0
                    elapsed  = time.time() - start_time
                    eta      = elapsed * (total_frames - frame_counter) / frame_counter
                    exp_str  = f"~{exp_size/1024**2:.1f} MB" if exp_size else "?"
                    print(f"  Frame {frame_counter:6d}/{total_frames}  ({pct:5.1f}%)  "
                          f"out {cur_size/1024**2:6.1f} MB / {exp_str}  "
                          f"[{fmt_hms(elapsed)} / ETA {fmt_hms(eta)}]", end='\r')

    proc.stdin.close()
    rc = proc.wait()
    ffmpeg_log.close()
    print()
    if rc != 0:
        print(f"FFmpeg error (rc={rc}):\n  {_read_ffmpeg_log()}")
        sys.exit(1)

    out_size = os.path.getsize(output_path)
    print(f"Done -> {output_path}  ({out_size/1024**2:.1f} MB)")


# ============================================================================
#  DECODE
# ============================================================================

def classify_luma(grey_blocks):
    """
    Classify N luma values to the nearest 2-level palette symbol.

    grey_blocks: (N,) int32 -- averaged luma per block (0-255)
    Returns:     (N,) uint8 -- symbol indices 0 (Black) or 1 (White)
    """
    # PALETTE_LUMA = [0, 255]; single midpoint at 127.
    syms = np.zeros(len(grey_blocks), dtype=np.uint8)
    syms[grey_blocks >= 127] = 1
    return syms


def symbols_to_bits(syms):
    """syms: (N,) uint8 -> bits: (N*2,) uint8"""
    return SYM_BITS[syms].ravel()


def is_url(s):
    """Return True if s looks like an http(s) or other network URL."""
    return s.startswith(('http://', 'https://', 'ytdl://', 'yt-dlp://'))


def _resolve_tool(name):
    """
    Locate an external tool (ffmpeg, yt-dlp) and return the full path to use.

    Search order:
      1. Every entry on PATH, in order -- but we validate each candidate by
         actually running '<candidate> --version' before accepting it.  This
         catches broken PyInstaller bundles (e.g. a yt-dlp.exe that fails to
         load its bundled Python DLL) that would otherwise silently corrupt
         the pipeline deep inside a worker pool.
      2. The directory that contains this script file.  Handy when the user
         has dropped ffmpeg.exe / yt-dlp.exe next to YoutubeTranscoder.py
         without adding them to PATH.

    Returns the resolved path string (suitable for use as argv[0] in
    subprocess calls) so every Popen / check_output call uses the same
    validated binary.

    Raises SystemExit with a clear, actionable error message if no working
    candidate is found anywhere.
    """
    import shutil

    # Build the list of candidates to try, in priority order.
    candidates = []

    # 1. All PATH hits for this name, in PATH order.
    #    shutil.which() returns the first hit; we want *all* hits so we can
    #    skip broken ones and keep looking.
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    exts = ['']
    if _IS_WINDOWS:
        exts = os.environ.get('PATHEXT', '.EXE;.CMD;.BAT').split(';')
    for d in path_dirs:
        for ext in exts:
            candidate = os.path.join(d, name + ext)
            if os.path.isfile(candidate) and candidate not in candidates:
                candidates.append(candidate)

    # 2. Script's own directory (fallback for bundled binaries).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for ext in exts:
        candidate = os.path.join(script_dir, name + ext)
        if os.path.isfile(candidate) and candidate not in candidates:
            candidates.append(candidate)

    # Try each candidate; accept the first one that actually runs.
    for path in candidates:
        try:
            result = subprocess.run(
                [path, '--version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            # Return code 0 = success.  Some builds return non-zero for
            # --version (unusual but harmless); what matters is that the
            # process started and exited cleanly rather than crashing.
            # We accept any return code as long as no exception was raised.
            print(f"  [{name}] Using: {path}", flush=True)
            return path
        except (FileNotFoundError, OSError):
            # Executable not runnable (e.g. wrong architecture).
            continue
        except subprocess.TimeoutExpired:
            # Hung on --version — skip it.
            continue
        except Exception:
            # Crashed (e.g. broken PyInstaller DLL) — stderr will contain
            # the error; skip this candidate and try the next one.
            continue

    # Nothing worked — print a clear error and exit.
    tried = '\n    '.join(candidates) if candidates else '(none found)'
    print(f"\nERROR: No working '{name}' binary could be found.", file=sys.stderr)
    print(f"  Searched PATH and script directory ({script_dir}).", file=sys.stderr)
    if candidates:
        print(f"  Candidates tried (all failed to run):\n    {tried}", file=sys.stderr)
    if name == 'yt-dlp':
        print(
            "\n  To fix:\n"
            "    pip install yt-dlp                        (recommended)\n"
            "    -- or --\n"
            "    Download the standalone binary from:\n"
            "      https://github.com/yt-dlp/yt-dlp/releases\n"
            "    and place yt-dlp.exe next to this script OR add it to PATH.\n"
            "    (The standalone .exe requires its _internal/ folder beside it;\n"
            "     pip install avoids that dependency entirely.)",
            file=sys.stderr,
        )
    elif name == 'ffmpeg':
        print(
            "\n  To fix:\n"
            "    winget install ffmpeg                     (Windows, recommended)\n"
            "    -- or --\n"
            "    Download from https://ffmpeg.org/download.html\n"
            "    and place ffmpeg.exe next to this script OR add it to PATH.",
            file=sys.stderr,
        )
    sys.exit(1)


# Resolved tool paths -- populated on first call to encode() or decode()
# so worker processes (which never call those functions) don't resolve tools.
_FFMPEG = None
_YTDLP  = None


def _init_tools(need_ytdlp=False):
    """Resolve ffmpeg (always) and yt-dlp (when streaming) and cache the paths."""
    global _FFMPEG, _YTDLP
    print("Locating external tools...", flush=True)
    _FFMPEG = _resolve_tool('ffmpeg')
    if need_ytdlp:
        _YTDLP = _resolve_tool('yt-dlp')
    print()


def ytdlp_probe(url):
    """
    Use yt-dlp --print to fetch video metadata without downloading anything.

    Returns a dict with keys: title, id, duration, uploader, view_count,
    webpage_url, and ext (the container extension of the best video stream).
    Any field may be None if yt-dlp does not report it.  Never raises --
    returns an empty dict on error so callers can degrade gracefully.
    """
    fields = {
        'title':      '%(title)s',
        'id':         '%(id)s',
        'duration':   '%(duration)s',
        'uploader':   '%(uploader)s',
        'view_count': '%(view_count)s',
        'ext':        '%(ext)s',
    }
    # Build one --print per field so output is line-by-line in a stable order.
    cmd = [_YTDLP, '--no-warnings', '--quiet']
    keys = list(fields.keys())
    for k in keys:
        cmd += ['--print', fields[k]]
    cmd += ['-f', 'bestvideo', '--skip-download', url]

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=30)
        lines = out.decode('utf-8', errors='replace').splitlines()
        result = {}
        for i, k in enumerate(keys):
            val = lines[i].strip() if i < len(lines) else None
            result[k] = None if (val in (None, 'NA', 'None', '')) else val
        return result
    except Exception:
        return {}


def stream_raw_frames(video_path):
    """
    Decode video frames to the canonical block grid (1920x1080) as greyscale.
    Area-averaging downscale preserves luma fidelity for each block.
    Yields (frame_index, frame_grey) where frame_grey is (1080, 1920) uint8.

    Colorspace note: YouTube re-tags uploaded videos as BT.709 limited-range
    (Y: 16–235).  If ffmpeg honours that tag it applies a limited->full range
    expansion (Y_out = (Y_in - 16) * 255/219) before writing gray pixels,
    which shifts the two palette levels (0/255) away from their expected
    values and causes symbol mis-classification -- visible as the
    bottom ~90% of the image decoding to grey noise while the top rows look
    correct.

    Fix: pass -vf "scale,format=gray" with an explicit setparams filter that
    overrides the container's colorspace metadata and marks the stream as
    full-range BEFORE the format conversion.  This tells ffmpeg the Y values
    are already [0–255] and no range expansion should be applied.
    """
    vf = (f'scale={_COLS}:{_ROWS}:flags=area,'
          f'setparams=range=pc:colorspace=bt709:color_primaries=bt709'
          f':color_trc=bt709,'
          f'format=gray')
    cmd = [
        _FFMPEG,
        '-color_range', 'pc',    # hint to demuxer: treat input as full-range
        '-i', video_path,
        '-vf', vf,
        '-f', 'rawvideo', '-pix_fmt', 'gray',
        'pipe:1'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=open(_tmp('ffmpeg_decode.log'), 'w', errors='replace'))

    def cleanup():
        try:
            proc.kill()
        except Exception:
            pass

    frame_bytes = _COLS * _ROWS   # gray = 1 byte/pixel
    idx = 0
    try:
        with _Cleanup(cleanup):
            while True:
                raw = proc.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((_ROWS, _COLS))
                yield idx, frame
                idx += 1
    finally:
        proc.stdout.close()
        proc.wait()


def stream_raw_frames_from_url(url):
    """
    Stream-decode a YouTube (or any yt-dlp-supported) URL directly to raw
    greyscale frames without writing any temp file.

    Pipeline:
        yt-dlp  -f bestvideo  -o -  <URL>   (video bytes -> stdout)
             |
        ffmpeg  -i pipe:0  -vf scale+setparams+format=gray  pipe:1
             |
        this generator  ->  (frame_idx, frame_grey) pairs

    Why 'bestvideo' and not 'bestvideo+bestaudio':
        Merging two streams into an mp4 container requires random-write access
        (moov atom must precede mdat, so ffmpeg needs to seek back after
        writing the mux trailer).  A pipe cannot seek.  Since this codec
        carries no audio payload and the data is entirely in the video stream,
        selecting only the video track avoids the merge step entirely and
        allows true streaming -- yt-dlp pipes the container bytes in real time
        and ffmpeg decodes them frame by frame as they arrive.

    yt-dlp progress lines are captured from stderr and printed to the
    terminal via a background thread so the user can see download speed and
    ETA without blocking the frame generator.

    Yields (frame_index, frame_grey) exactly like stream_raw_frames().
    """
    import threading

    vf = (f'scale={_COLS}:{_ROWS}:flags=area,'
          f'setparams=range=pc:colorspace=bt709:color_primaries=bt709'
          f':color_trc=bt709,'
          f'format=gray')

    # ── yt-dlp: pipe best video stream bytes to stdout ─────────────────────
    ytdlp_cmd = [
        _YTDLP,
        '-f', 'bestvideo',     # video-only -- avoids merge (needs seekable output)
        '--no-warnings',
        '--newline',           # one progress line per update (easier to parse)
        '--progress',
        '-o', '-',             # write container bytes to stdout
        url,
    ]
    ytdlp_proc = subprocess.Popen(
        ytdlp_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,   # capture for progress forwarding
    )

    # ── ffmpeg: read container from stdin, emit raw gray frames to stdout ──
    ffmpeg_cmd = [
        _FFMPEG,
        '-loglevel', 'warning',
        '-color_range', 'pc',
        '-i', 'pipe:0',        # container bytes arrive on stdin from yt-dlp
        '-vf', vf,
        '-f', 'rawvideo', '-pix_fmt', 'gray',
        'pipe:1',
    ]
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=ytdlp_proc.stdout,   # wire yt-dlp stdout -> ffmpeg stdin
        stdout=subprocess.PIPE,
        stderr=open(_tmp('ffmpeg_stream_decode.log'), 'w', errors='replace'),
    )
    # After handing ytdlp_proc.stdout to ffmpeg, close our own reference so
    # SIGPIPE propagates correctly if ffmpeg exits early.
    ytdlp_proc.stdout.close()

    # ── background thread: forward yt-dlp progress lines to terminal ───────
    _ytdlp_done = threading.Event()

    def _forward_ytdlp_stderr():
        try:
            for raw_line in ytdlp_proc.stderr:
                line = raw_line.decode('utf-8', errors='replace').rstrip()
                if not line:
                    continue
                # yt-dlp progress lines start with '[download]'; print them
                # over a carriage-return so they update in place.
                if line.startswith('[download]'):
                    print(f"  yt-dlp  {line}", end='\r', flush=True)
                else:
                    # Info/warning lines — print on their own line
                    print(f"  yt-dlp  {line}", flush=True)
        except Exception:
            pass
        finally:
            _ytdlp_done.set()

    stderr_thread = threading.Thread(target=_forward_ytdlp_stderr, daemon=True)
    stderr_thread.start()

    # ── cleanup ─────────────────────────────────────────────────────────────
    def cleanup():
        for p in (ffmpeg_proc, ytdlp_proc):
            try:
                p.kill()
            except Exception:
                pass

    # ── yield frames ────────────────────────────────────────────────────────
    frame_size = _COLS * _ROWS   # gray = 1 byte/pixel
    idx = 0
    try:
        with _Cleanup(cleanup):
            while True:
                raw = ffmpeg_proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((_ROWS, _COLS))
                yield idx, frame
                idx += 1
    finally:
        ffmpeg_proc.stdout.close()
        ffmpeg_proc.wait()
        ytdlp_proc.wait()
        _ytdlp_done.wait(timeout=5)
        print()   # clear the yt-dlp progress line


def decode_frame(args):
    """
    Decode one block-resolution greyscale frame.
    Called in worker pool.

    args = (frame_idx, frame_grey)   frame_grey: (_ROWS, _COLS) uint8
    Returns (frame_idx, result_tuple) on success, or
            (frame_idx, str)          on failure -- str is a short reason tag
            used by the caller to tally rejection counts by cause.
    """
    frame_idx, frame_grey = args

    # ── sanity: top-left corner block must be Black or White ──
    # The border checkerboard is encoded with symbols 0 (luma=0) and 1 (luma=255).
    # Both are far outside the mid-grey range (64–191) in the expected
    # direction.  Any other value means we're looking at garbage or a frame
    # from a different video stream.
    tl_luma = int(frame_grey[0, 0])
    if not (tl_luma < 64 or tl_luma > 191):
        return (frame_idx, f'border_luma={tl_luma}')

    # ── decode header row (inner row 0) ──
    hdr_grey = frame_grey[1, 1:1+_INNER_COLS].astype(np.int32)   # (1918,)
    hdr_syms = classify_luma(hdr_grey)                            # (1918,)

    hdr_bits = symbols_to_bits(hdr_syms[:_HEADER_BLOCKS])        # 184 bits (92x2)
    hdr_bits = hdr_bits[:HEADER_BITS]

    if len(hdr_bits) < HEADER_BITS:
        return (frame_idx, 'short_header')

    pad        = (8 - HEADER_BITS % 8) % 8
    hdr_padded = np.concatenate([hdr_bits, np.zeros(pad, dtype=np.uint8)])
    hdr_bytes  = np.packbits(hdr_padded).tobytes()[:HEADER_BYTES]

    try:
        segment_id, frame_in_seg, total_segments, total_frames, \
            file_size, k, m = struct.unpack(HEADER_FMT, hdr_bytes)
    except struct.error:
        return (frame_idx, 'struct_unpack')

    # ── manifest frame detection ──
    # The first 4 bytes of the per-frame header are segment_id.  When they
    # equal MANIFEST_MAGIC the frame carries filename + SHA-256 + k/m, not
    # zfec data.  Return a special tuple so decode() can extract it without
    # zfec interference.  parse_manifest_frame reads copy 1 from the header
    # row (top-left) and copy 2 from the bottom-right corner of the inner
    # data area; both must match before the manifest is accepted.
    if segment_id == MANIFEST_MAGIC:
        parsed = parse_manifest_frame(frame_grey)
        if parsed is not None:
            orig_filename, sha256_hex, mk, mm = parsed
            return (frame_idx, ('__manifest__', orig_filename, sha256_hex, mk, mm))
        # Manifest magic present but payload unreadable — treat as corrupt but
        # don't count it as a data-frame rejection.
        return (frame_idx, 'manifest_corrupt')

    if k == 0 or m == 0 or k > m or m > 255:
        return (frame_idx, f'bad_km(k={k},m={m})')
    if total_segments == 0 or total_segments > 1_000_000:
        return (frame_idx, f'bad_total_segments={total_segments}')
    if frame_in_seg >= m:
        return (frame_idx, f'frame_in_seg={frame_in_seg}>={m}')

    # ── decode data rows ──
    # Layout: sha256_prefix(32) || share_bytes(_CHUNK_SIZE) || sha256_suffix(32)
    # Step 1: if prefix != suffix  → hash fields corrupted → discard frame
    # Step 2: if sha256(share_bytes) != prefix → payload corrupted → erase
    data_grey = frame_grey[2:2+_DATA_ROWS, 1:1+_INNER_COLS].ravel().astype(np.int32)
    data_syms = classify_luma(data_grey)
    data_bits = symbols_to_bits(data_syms)

    total_bits   = _DATA_ROWS * _INNER_COLS * BITS_PER_BLOCK
    frame_bytes  = total_bits // 8                    # == _FRAME_DATA_BYTES
    data_bits    = data_bits[:frame_bytes * 8]
    pad2         = (8 - len(data_bits) % 8) % 8
    data_padded  = np.concatenate([data_bits, np.zeros(pad2, dtype=np.uint8)])
    frame_data   = np.packbits(data_padded).tobytes()[:frame_bytes]

    # Unpack: prefix(32) | payload(_CHUNK_SIZE) | suffix(32)
    sha_prefix  = frame_data[:FRAME_SHA_BYTES]
    share_bytes = frame_data[FRAME_SHA_BYTES:FRAME_SHA_BYTES + _CHUNK_SIZE]
    sha_suffix  = frame_data[FRAME_SHA_BYTES + _CHUNK_SIZE:
                             FRAME_SHA_BYTES + _CHUNK_SIZE + FRAME_SHA_BYTES]

    # Step 1: both copies of the guard must be identical
    if sha_prefix != sha_suffix:
        return (frame_idx, f'sha_guard_mismatch(seg={segment_id},share={frame_in_seg})')

    # Step 2: guard must match actual payload hash
    sha_computed = hashlib.sha256(share_bytes).digest()
    if sha_computed != sha_prefix:
        return (frame_idx, f'sha_fail(seg={segment_id},share={frame_in_seg})')

    return (frame_idx, (segment_id, frame_in_seg, total_segments,
                        total_frames, file_size, k, m, share_bytes))


def decode(video_path, output_path=None, workers=None, verbose=False):
    """
    Decode a data video back to its original file.

    video_path  – local file path  OR  a URL supported by yt-dlp
                  (http://, https://, …).  When a URL is given the video is
                  streamed frame-by-frame via yt-dlp -> ffmpeg without writing
                  any temporary file to disk.
    output_path – explicit destination path, or None to derive from the
                  filename embedded in the video's manifest frame
                  (original_basename + DECODED_SUFFIX).
    verbose     – ignored (kept for API compatibility); output is always full.

    Returns the resolved output path so callers (roundtrip) can inspect it.
    Also returns the embedded SHA-256 hex string (or None if no manifest was
    found) so callers can do their own verification reporting.
    """
    workers = workers or cpu_count()

    # ── URL vs local file ──────────────────────────────────────────────────
    streaming = is_url(video_path)

    if streaming:
        _init_tools(need_ytdlp=True)
        print(f"Source         : {video_path}")
        print(f"Mode           : streaming via yt-dlp -> ffmpeg (no temp file)")
        print()
        print("Probing video metadata...", flush=True)
        meta_info = ytdlp_probe(video_path)
        if meta_info:
            title    = meta_info.get('title')    or '(unknown)'
            vid_id   = meta_info.get('id')       or '(unknown)'
            duration = meta_info.get('duration')
            uploader = meta_info.get('uploader') or '(unknown)'
            views    = meta_info.get('view_count')
            dur_str  = fmt_hms(float(duration)) if duration else '(unknown)'
            views_str = f"{int(views):,}" if views else '(unknown)'
            print(f"  Title    : {title}")
            print(f"  ID       : {vid_id}")
            print(f"  Duration : {dur_str}")
            print(f"  Uploader : {uploader}")
            print(f"  Views    : {views_str}")

            # ── 8K stream check ───────────────────────────────────────────
            # yt-dlp --print with -f bestvideo returns the best available
            # video format.  For this codec to decode correctly the stream
            # MUST be 8K (7680×4320); any other resolution means YouTube
            # has not yet made the 8K version available.
            ytdlp_res_cmd = [
                _YTDLP, '--no-warnings', '--quiet',
                '--print', '%(width)sx%(height)s',
                '-f', 'bestvideo',
                '--skip-download', video_path,
            ]
            try:
                res_out = subprocess.check_output(
                    ytdlp_res_cmd, stderr=subprocess.DEVNULL, timeout=30)
                res_str = res_out.decode('utf-8', errors='replace').strip()
                parts   = res_str.lower().split('x')
                w_avail = int(parts[0]) if len(parts) == 2 and parts[0].isdigit() else 0
                h_avail = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
            except Exception:
                w_avail, h_avail, res_str = 0, 0, '(unknown)'

            if w_avail == W8K and h_avail == H8K:
                print(f"  Resolution : {res_str} [OK] (8K -- correct for this codec)")
            else:
                warn_res = res_str if res_str != '(unknown)' else f'{w_avail}x{h_avail}'
                print()
                print("######################################################################")
                print("######                        WARNING                           ######")
                print("######################################################################")
                print(f"######  Best available stream : {warn_res:<36s}######")
                print(f"######  Required resolution   : {W8K}x{H8K} (8K hardcoded)    ######")
                print("######                                                          ######")
                print("######  This video will NOT decode correctly at this            ######")
                print("######  resolution. The codec grid is fixed at 8K;             ######")
                print("######  downscaled streams produce misaligned blocks and       ######")
                print("######  corrupt data reads.                                    ######")
                print("######                                                          ######")
                print("######  Wait for YouTube to process and serve the 8K stream,   ######")
                print("######  then retry.  This can take minutes to hours after      ######")
                print("######  the initial upload.                                    ######")
                print("######################################################################")
                print()
        else:
            print("  (metadata probe failed -- continuing anyway)")
        print()
        frame_source = stream_raw_frames_from_url(video_path)
    else:
        _init_tools(need_ytdlp=False)
        print(f"Probing        : {video_path}")
        frame_source = stream_raw_frames(video_path)

    print(f"Canonical grid : {_COLS}x{_ROWS} blocks  (8K / {BLOCK}px -- hardcoded)")
    print(f"Workers        : {workers}")
    print(f"Bits/block     : {BITS_PER_BLOCK}  (2-level black & white nearest-threshold)")
    print(f"Frame guard    : double SHA-256 ({FRAME_SHA_BYTES}x2 B) -- "
          f"hash mismatch -> erasure, not silent corruption")
    print()

    segments           = {}
    meta               = None      # (total_segments, file_size, k, m, total_frames)
    completed          = set()
    results            = {}
    dec_cache          = {}
    decoded_bytes      = 0
    frames_seen        = 0
    frames_ok          = 0
    frames_sha_fail    = 0   # shares erased due to SHA-256 payload mismatch
    frames_guard_mism  = 0   # frames discarded because the two guard copies differ
    reject_counts      = {}  # reason -> count
    manifest           = None  # (orig_filename, sha256_hex, k, m) once found

    # max_broken_frames: computed once meta (or manifest) gives us k, m, total_segments.
    # Formula: each segment tolerates (m-k) bad frames; summed = (m-k)*total_segments.
    # This is the upper bound of individually broken frames that zfec can still recover.
    max_broken_frames  = None

    # Live progress: 3 updating lines printed in-place with ANSI.
    _PROG_LINES = 3
    _prog_started = False

    def _print_progress(pct, frames_cur, frames_total,
                        checksum_fails, max_broken, elapsed, eta):
        """Overwrite the last _PROG_LINES lines with fresh progress."""
        nonlocal _prog_started
        # Move cursor up to overwrite previous block (after the first render).
        if _prog_started:
            _ansi_cursor_up(_PROG_LINES)
        _prog_started = True

        # Line 1 — overall progress + frame counter
        frames_str = f"{frames_cur}/{frames_total}" if frames_total else f"{frames_cur}/?"
        pct_str    = f"{pct:6.2f}%" if pct is not None else "  ?.??%"
        line1 = f"  Progress {pct_str}     Frames {frames_str}"

        # Line 2 — checksum failures vs tolerance budget
        if max_broken is not None:
            fail_str = f"{checksum_fails}/{max_broken} frames"
        else:
            fail_str = f"{checksum_fails}/? frames"
        line2 = f"  Checksum Fails {fail_str}"

        # Line 3 — runtime / ETA
        eta_str = fmt_hms(eta) if eta is not None else "?"
        line3 = f"  Runtime {fmt_hms(elapsed)}     ETA {eta_str}"

        # Clear each line to the right before printing (handles shrinking content).
        for line in (line1, line2, line3):
            sys.stdout.write(line)
            _ansi_erase_line()
            sys.stdout.write("\n")
        sys.stdout.flush()

    def get_decoder(k, m):
        if (k, m) not in dec_cache:
            dec_cache[(k, m)] = zfec.Decoder(k, m)
        return dec_cache[(k, m)]

    pool = None

    def cleanup():
        try:
            if pool is not None:
                pool.terminate()
        except Exception:
            pass

    # Print 3 blank lines so the first _print_progress overwrite has room.
    print("\n" * _PROG_LINES, end="")

    # Use 'spawn' on Windows/macOS to avoid fork-related issues; 'fork' on Linux
    # for speed.  'spawn' requires all worker functions to be importable at the
    # module level (decode_frame is defined at module scope, so this is safe).
    _mp_method = 'fork' if platform.system() == 'Linux' else 'spawn'
    _mp_ctx = get_context(_mp_method)
    with _Cleanup(cleanup), _mp_ctx.Pool(workers) as pool:
        start_time = time.time()
        for frame_idx, result in pool.imap_unordered(
                decode_frame,
                frame_source,
                chunksize=workers * 2):

            frames_seen += 1
            elapsed = time.time() - start_time

            # ── rejected frame ──
            if isinstance(result, str):
                reason = result
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                is_sha_fail   = reason.startswith('sha_fail')
                is_guard_mism = reason.startswith('sha_guard_mismatch')
                if is_sha_fail:
                    frames_sha_fail += 1
                elif is_guard_mism:
                    frames_guard_mism += 1
                # Update live display (checksum count changed or just cadence).
                total_frames_known = meta[4] if meta else None
                pct = (frames_seen / total_frames_known * 100
                       if total_frames_known else None)
                eta = (elapsed * (total_frames_known - frames_seen) / frames_seen
                       if total_frames_known and frames_seen else None)
                _print_progress(pct, frames_seen, total_frames_known,
                                frames_sha_fail + frames_guard_mism,
                                max_broken_frames, elapsed, eta)
                continue

            # ── manifest frame ──
            if isinstance(result, tuple) and len(result) == 5 and result[0] == '__manifest__':
                _, orig_filename, sha256_hex, mk, mm = result
                if manifest is None:
                    manifest = (orig_filename, sha256_hex, mk, mm)
                    # Move cursor above progress block, print manifest line, reprint progress.
                    _ansi_cursor_up(_PROG_LINES)
                    print(f"  [OK] MANIFEST  filename={orig_filename!r}  "
                          f"sha256={sha256_hex[:16]}...  k={mk}  m={mm}")
                    _prog_started = False   # force re-render below this new line
                    total_frames_known = meta[4] if meta else None
                    pct = (frames_seen / total_frames_known * 100
                           if total_frames_known else None)
                    eta = (elapsed * (total_frames_known - frames_seen) / frames_seen
                           if total_frames_known and frames_seen else None)
                    _print_progress(pct, frames_seen, total_frames_known,
                                    frames_sha_fail + frames_guard_mism,
                                    max_broken_frames, elapsed, eta)
                continue

            frames_ok += 1
            segment_id, frame_in_seg, total_segments, total_frames, \
                file_size, k, m, share_bytes = result

            if meta is None:
                total_frames_with_manifest = total_frames + 1  # +1 for manifest frame
                meta = (total_segments, file_size, k, m, total_frames_with_manifest)
                max_broken_frames = (m - k) * total_segments
                # Print metadata line above progress.
                _ansi_cursor_up(_PROG_LINES)
                print(f"  [OK] METADATA  segs={total_segments}  "
                      f"file={file_size/1024**2:.2f} MB  k={k}  m={m}  "
                      f"frames={total_frames}  "
                      f"max_broken={max_broken_frames}")
                _prog_started = False

            if segment_id in completed:
                # Still update the display on cadence.
                total_frames_known = meta[4]
                pct = frames_seen / total_frames_known * 100
                eta = (elapsed * (total_frames_known - frames_seen) / frames_seen
                       if frames_seen else None)
                _print_progress(pct, frames_seen, total_frames_known,
                                frames_sha_fail + frames_guard_mism,
                                max_broken_frames, elapsed, eta)
                continue

            seg = segments.setdefault(segment_id, {})
            # seg['shares']      : dict  frame_in_seg -> share_bytes (good shares only)
            # seg['tried_combos']: set   of frozensets of share-ids already attempted
            seg.setdefault('shares', {})
            seg.setdefault('tried_combos', set())
            seg['shares'].setdefault(frame_in_seg, share_bytes)

            total_frames_known = meta[4]
            done = len(completed)
            pct  = frames_seen / total_frames_known * 100
            eta  = (elapsed * (total_frames_known - frames_seen) / frames_seen
                    if frames_seen else None)
            _print_progress(pct, frames_seen, total_frames_known,
                            frames_sha_fail + frames_guard_mism,
                            max_broken_frames, elapsed, eta)

            good_shares = seg['shares']
            if len(good_shares) >= k:
                # Try every untried combination of k shares out of what we have.
                # Itertools would generate C(n,k) combos; in practice n is at most
                # m=15 and k=12, so the worst case is C(15,12)=455 attempts --
                # negligible.  We stop as soon as one combination decodes cleanly.
                import itertools
                dec = get_decoder(k, m)
                all_ids = sorted(good_shares.keys())
                decoded_ok = False
                for combo in itertools.combinations(all_ids, k):
                    combo_key = frozenset(combo)
                    if combo_key in seg['tried_combos']:
                        continue
                    seg['tried_combos'].add(combo_key)
                    shares = [good_shares[i] for i in combo]
                    try:
                        pieces = dec.decode(shares, list(combo))
                        results[segment_id] = b''.join(pieces)
                        decoded_bytes += len(results[segment_id])
                        completed.add(segment_id)
                        del segments[segment_id]
                        done_now = len(completed)
                        _ansi_cursor_up(_PROG_LINES)
                        print(f"  [OK] SEG {segment_id:5d} DONE  "
                              f"({done_now}/{meta[0]} segs  "
                              f"{decoded_bytes/1024**2:.1f}/{file_size/1024**2:.1f} MB)")
                        _prog_started = False
                        _print_progress(pct, frames_seen, total_frames_known,
                                        frames_sha_fail + frames_guard_mism,
                                        max_broken_frames, elapsed, eta)
                        decoded_ok = True
                        break
                    except Exception:
                        # This combo didn't work; a share in it is silently corrupt
                        # (passed SHA but carries bad data).  Keep trying other combos
                        # as more shares arrive.
                        continue

            if meta and len(completed) == meta[0]:
                _ansi_cursor_up(_PROG_LINES)
                print(f"  [OK] All {meta[0]} segments decoded -- "
                      f"stopping early at frame {frame_idx}  [{fmt_hms(elapsed)}]")
                _prog_started = False
                _print_progress(100.0, frames_seen, total_frames_known,
                                frames_sha_fail + frames_guard_mism,
                                max_broken_frames, elapsed, 0)
                pool.terminate()
                break

    print()

    # ── frame statistics summary ──
    reject_total = sum(reject_counts.values())
    print("-" * 72)
    print(f"  Frames seen  : {frames_seen}")
    print(f"  Frames good  : {frames_ok}")
    if frames_sha_fail:
        print(f"  SHA-256 fail : {frames_sha_fail}  (payload corrupted -> erased for zfec)")
    if frames_guard_mism:
        print(f"  Guard mismatch: {frames_guard_mism}  (hash copies differ -> frame discarded)")

    # Compute lost frames: segments where we never received enough shares
    if meta:
        total_segs, _fs, k2, m2, _tf = meta
        incomplete_segs = [(sid, len(segments.get(sid, {}).get('shares', {})))
                           for sid in range(total_segs)
                           if sid not in completed]
        if incomplete_segs:
            lost_shares = sum(max(0, k2 - cnt) for _, cnt in incomplete_segs)
            print(f"  Lost frames  : >={lost_shares}  "
                  f"({len(incomplete_segs)} segment(s) under-replicated)")
        else:
            print(f"  Lost frames  : 0")

    other = reject_total - frames_sha_fail - frames_guard_mism
    if other:
        bucketed = {}
        for reason, cnt in reject_counts.items():
            if reason.startswith('sha_fail') or reason.startswith('sha_guard_mismatch'):
                continue
            bucket = reason.split('(')[0] if '(' in reason else (
                     reason.split('=')[0] if '=' in reason else reason)
            bucketed[bucket] = bucketed.get(bucket, 0) + cnt
        print(f"  Other rejects: {other}")
        for bucket, cnt in sorted(bucketed.items(), key=lambda kv: -kv[1]):
            print(f"    {bucket:30s}  {cnt:6d}")
    if not reject_total:
        print(f"  Integrity    : [OK] all frames clean (no SHA failures)")
    print("-" * 72)

    if meta is None:
        hint = ''
        if reject_counts:
            top_reason = max(reject_counts, key=reject_counts.get)
            if top_reason.startswith('border_luma'):
                try:
                    luma = int(top_reason.split('=')[1])
                except (IndexError, ValueError):
                    luma = None
                if luma is not None and 64 <= luma <= 191:
                    hint = (f"\n  Hint: border luma={luma} is in the mid-grey range "
                            f"(64–191).\n"
                            f"  The encoder's make_border() likely used a mid-grey "
                            f"value instead of\n"
                            f"  symbol 0 (Black) / symbol 1 (White).\n"
                            f"  Re-encode with the fixed encoder.")
        print(f"ERROR: no valid frames decoded{hint}")
        sys.exit(1)

    total_segments, file_size, k, m, _total_frames = meta
    print()

    missing = [i for i in range(total_segments) if i not in completed]
    if missing:
        print(f"ERROR: {len(missing)} segments missing: {missing[:20]}"
              f"{'...' if len(missing) > 20 else ''}")
        sys.exit(1)

    # ── resolve output path ──
    # Priority: explicit CLI arg > derived from manifest > fallback
    if output_path is None:
        if manifest is not None:
            orig_filename, _, _mk, _mm = manifest
            base, ext = os.path.splitext(orig_filename)
            derived   = base + DECODED_SUFFIX + ext
            # For URLs, os.path.dirname() returns the URL prefix which is
            # not a valid local directory — always write to CWD instead.
            out_dir   = '.' if streaming else (os.path.dirname(video_path) or '.')
            output_path = os.path.join(out_dir, derived)
            print(f"Output path : {output_path}  (from manifest + suffix '{DECODED_SUFFIX}')")
        else:
            # No manifest found — fall back to a sensible local name.
            if streaming:
                output_path = 'decoded_from_stream'
            elif video_path.endswith('.mp4'):
                output_path = video_path[:-len('.mp4')]
            else:
                output_path = video_path + '.decoded'
            print(f"Output path : {output_path}  (no manifest; derived from source)")

    # ── reassemble ──
    print(f"Reassembling {total_segments} segments...")
    raw = b''.join(results[i] for i in range(total_segments))
    raw = raw[:file_size]

    with open(output_path, 'wb') as f:
        f.write(raw)

    actual_sha256 = sha256_bytes(raw)
    print(f"Done -> {output_path}  ({file_size:,} bytes, {file_size/1024**2:.2f} MB)")
    print(f"SHA-256 (decoded) : {actual_sha256}")

    # ── integrity verification ──
    embedded_sha256 = None
    if manifest is not None:
        _, embedded_sha256, _mk, _mm = manifest
        print(f"SHA-256 (manifest): {embedded_sha256}")
        if actual_sha256 == embedded_sha256:
            print("Integrity : PASS [OK]  decoded file matches embedded SHA-256")
        else:
            print("Integrity : FAIL [!!]  SHA-256 mismatch -- decoded file is corrupt!")
            print("  This may indicate data loss in the video or a bug in the codec.")
            sys.exit(1)
    else:
        print("Integrity : (no manifest frame found -- SHA-256 verification skipped)")

    return output_path, embedded_sha256


# ============================================================================
#  ROUND-TRIP (encode -> decode -> verify)
# ============================================================================

def roundtrip(input_path, video_path, decoded_path, fps, k, m,
              workers, crf, preset, use_nvenc, qp, nvenc_preset, verbose,
              bitrate=None):
    print("=" * 72)
    print("ROUND-TRIP: encode -> decode -> verify")
    print("=" * 72)

    # ── [1/3] encode ──
    print(f"[1/3] Encoding {input_path} -> {video_path}")
    with open(input_path, 'rb') as f:
        original_bytes = f.read()
    sha_original = sha256_bytes(original_bytes)
    print(f"      Source SHA-256 : {sha_original}")

    encode(input_path, video_path, fps=fps, k=k, m=m,
           crf=crf, preset=preset, use_nvenc=use_nvenc, qp=qp,
           nvenc_preset=nvenc_preset, bitrate=bitrate)

    # ── [2/3] decode ──
    print()
    print(f"[2/3] Decoding {video_path} -> {decoded_path}")
    # Pass decoded_path explicitly so the roundtrip always writes where
    # expected, regardless of the embedded filename.
    actual_decoded_path, embedded_sha256 = decode(
        video_path, decoded_path, workers=workers, verbose=verbose)

    with open(actual_decoded_path, 'rb') as f:
        decoded_bytes_content = f.read()
    sha_decoded = sha256_bytes(decoded_bytes_content)

    # ── [3/3] hash-chain verification ──
    # The chain we care about:
    #   source file SHA  ==  SHA embedded in video manifest  ==  decoded file SHA
    # All three being identical proves the codec preserved the file exactly.
    # The video container's own SHA is intentionally excluded — it's a property
    # of the mp4 wrapper/compression, not of the data payload.
    print()
    print("=" * 72)
    print("[3/3] Hash chain verification")
    print("=" * 72)
    col = 18
    print(f"  {'Source file':{col}}: {input_path}")
    print(f"  {'SHA-256':{col}}: {sha_original}")
    print()
    print(f"  {'Encoded video':{col}}: {video_path}")
    print(f"  {'Manifest SHA-256':{col}}: {embedded_sha256 if embedded_sha256 else '(no manifest)'}")
    print()
    print(f"  {'Decoded file':{col}}: {actual_decoded_path}")
    print(f"  {'SHA-256':{col}}: {sha_decoded}")
    print()

    checks = []

    # Check 1: the SHA the encoder embedded in the video matches the source
    if embedded_sha256 is not None:
        if embedded_sha256 == sha_original:
            checks.append(("Source SHA == Manifest SHA", True,
                            "encoder correctly embedded the source SHA-256 into the video"))
        else:
            checks.append(("Source SHA == Manifest SHA", False,
                            "manifest SHA-256 does not match the source -- encoder bug?"))

    # Check 2: the decoded file matches the SHA embedded in the video
    if embedded_sha256 is not None:
        if sha_decoded == embedded_sha256:
            checks.append(("Decoded SHA == Manifest SHA", True,
                            "decoded file SHA-256 matches the SHA embedded in the video"))
        else:
            checks.append(("Decoded SHA == Manifest SHA", False,
                            "decoded file SHA-256 does not match the manifest -- data was lost in the video"))

    # Check 3: direct byte comparison (redundant if checks 1+2 pass, but explicit)
    if sha_decoded == sha_original:
        checks.append(("Decoded SHA == Source SHA", True,
                        "decoded file is byte-identical to the source"))
    else:
        checks.append(("Decoded SHA == Source SHA", False,
                        "decoded file differs from the source"))

    all_pass = all(ok for _, ok, _ in checks)
    for label, ok, detail in checks:
        icon = "[OK]" if ok else "[!!]"
        status = "PASS" if ok else "FAIL"
        print(f"  [{icon}] {status}  {label}")
        print(f"          {detail}")

    print()
    if all_pass:
        print(f"ROUND-TRIP PASS -- all {len(checks)} checks passed.")
    else:
        failed = sum(1 for _, ok, _ in checks if not ok)
        print(f"ROUND-TRIP FAIL -- {failed}/{len(checks)} check(s) failed.")
        if original_bytes != decoded_bytes_content:
            n = min(len(original_bytes), len(decoded_bytes_content))
            for i in range(n):
                if original_bytes[i] != decoded_bytes_content[i]:
                    print(f"  First differing byte at offset {i}")
                    break
        sys.exit(1)


# ============================================================================
#  CONFIGURATION  -- edit these values to change encoder behaviour
# ============================================================================

# -- zfec redundancy ---------------------------------------------------------
# The source file is split into segments of (ZFEC_K * _CHUNK_SIZE) bytes.
# Each segment is encoded into ZFEC_M shares (one share per frame); only
# ZFEC_K shares are needed to reconstruct the segment.  Up to (ZFEC_M - ZFEC_K)
# frames per segment can be lost or corrupted and the file is still recovered
# byte-for-byte.  With K=12, M=15 that is 3 bad frames per segment (~20%).
# Constraints: ZFEC_K < ZFEC_M <= 255.
# With K=10, M=15: tolerates 5 bad frames per segment (~33%) -- recommended
# for YouTube uploads, which apply two lossy passes (HEVC -> VP9 re-encode)
# and can corrupt 3-5 frames per segment in a single cluster.
# With K=12, M=15: tolerates only 3 bad frames per segment (~20%) -- may
# fail on segments that catch a cluster of YouTube VP9 artefacts.
ZFEC_K = 12   # shares required to reconstruct one segment
ZFEC_M = 15   # total shares (frames) produced per segment

# -- frame rate --------------------------------------------------------------
# Affects video duration and playback speed only.
# Data capacity and correctness are independent of frame rate.
ENCODE_FPS = 30

# -- CPU encoder (libx265 via ffmpeg) ----------------------------------------
# CPU_CRF:      x265 Constant Rate Factor ceiling.  Only active when
#               CPU_BITRATE is None (pure-CRF mode) or as an upper-quality
#               bound in constrained-bitrate mode.  Set to 0 for lossless
#               (x265-params lossless=1 is appended automatically).
#               NOTE: for this content, pure CRF produces output 3-4x larger
#               than NVENC at the same QP because x265's quality model allocates
#               many bits to the hard block edges.  Use CPU_BITRATE to match
#               NVENC output size instead.
# CPU_PRESET:   x265 encoding speed preset.
#               'superfast' is recommended for B&W block data: it skips the
#               expensive rect/amp partition and merge analyses that bloat file
#               size on uniform block content without improving data fidelity.
#               Slower presets (medium, slow, ...) INCREASE output size here.
# CPU_BITRATE:  Target average bitrate for the CPU path, in bits/second.
#               When set, x265 uses ABR mode with this target and CPU_CRF as
#               a quality ceiling (frames are never larger than CRF would
#               produce, but bitrate is capped to this value).  This is the
#               recommended way to match NVENC output size on the CPU path.
#               Set to None to use pure CRF mode (original behaviour).
#               80_000_000 (80 Mbps) matches NVENC QP 4 on 8K all-intra B&W
#               content (~2.1 GB for a 430 MB source file).
CPU_CRF     = 20
CPU_PRESET  = 'superfast'
CPU_BITRATE = 80_000_000   # bits/sec; set to None for pure-CRF mode

# -- GPU encoder (hevc_nvenc via ffmpeg, requires --nvenc flag) --------------
# NVENC_QP:     Constant QP value.  Lower = better quality, larger file.
#               4 is the recommended default: low enough that YouTube's two
#               re-encode passes (HEVC upload -> VP9 fallback) do not drift
#               luma values outside the ±127-unit palette margin.  QP 10 was
#               the prior default but caused frame corruption on some uploads.
#               Safe range: 0–14.  Values above 14 risk luma drift that can
#               cause bit errors after YouTube's VP9 re-encode.
# NVENC_PRESET: hevc_nvenc speed preset, p1 (fastest) to p7 (slowest).
#               p4 is a good default; p7 provides no meaningful quality gain
#               for all-intra static B&W block content.
#               Frames are upscaled on the CPU before hwupload_cuda to avoid
#               driver-dependent behaviour in scale_cuda.
NVENC_QP     = 4
NVENC_PRESET = 'p4'


# ============================================================================
#  CLI
# ============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        prog='YoutubeTranscoder.py',
        description=(
            'Encode any file into an 8K HEVC data video for YouTube upload, '
            'or decode one back to the original file.\n\n'
            'Encode (-e): renders the file as a B&W block-data video using '
            'libx265 (CPU) or hevc_nvenc (--nvenc GPU), tagged full-range '
            'BT.709 so YouTube\'s pipeline does not corrupt luma values.\n\n'
            'Decode (-d): INPUT may be a local .mp4 path OR any '
            'yt-dlp-supported URL (http/https).  When a URL is given the '
            'video is streamed frame-by-frame via yt-dlp -> ffmpeg with no '
            'temporary file written to disk.  The best stream must be 8K; '
            'a warning is shown if it is not.\n\n'
            'Round-trip (--trip): encodes, then decodes, then verifies the '
            'decoded file is byte-identical to the source via a 3-way '
            'SHA-256 hash chain (source == manifest == decoded).\n\n'
            'Encoder settings (ZFEC_K/M, FPS, CRF, QP, presets) are '
            'constants in the CONFIGURATION block at the bottom of this file.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s -e myfile.zip                      encode -> myfile.zip.mp4  (CPU x265)
  %(prog)s -e myfile.zip --nvenc              encode with NVIDIA GPU (hevc_nvenc)
  %(prog)s -e myfile.zip out.mp4              encode to explicit output path
  %(prog)s -d out.mp4                         decode -> <origname>_from_video.<ext>
  %(prog)s -d out.mp4 myfile.zip              decode to explicit output path
  %(prog)s -d https://youtu.be/XXXXXXXXXXX    stream from YouTube, decode directly
  %(prog)s -d https://youtu.be/XXXXXXXXXXX out.bin  stream + explicit output path
  %(prog)s --trip myfile.zip                  encode, decode, and verify (self-test)
  %(prog)s --trip myfile.zip --nvenc          same, using GPU encoder

output path defaults:
  -e / --encode   <input>.mp4
  -d / --decode   <original_basename>_from_video<original_ext>  (in CWD for URLs)
  --trip          video: <input>.mp4   decoded: <input>.roundtrip

streaming notes:
  yt-dlp selects the best video-only stream (audio is not downloaded).
  The stream must be 8K (7680x4320); a warning is printed if YouTube has
  not yet made the 8K version available -- decoding a lower-resolution
  stream will likely fail or produce corrupt output.
  yt-dlp download speed and ETA are shown live during streaming decode.

progress display:
  During decode, three lines are updated in-place using ANSI cursor control.
  On Windows, Virtual Terminal Processing is enabled automatically (Win 10+).
  If ANSI is unavailable the lines print sequentially instead.
""")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('-e', '--encode', action='store_true',
                       help='Encode a file into a data video')
    mode.add_argument('-d', '--decode', action='store_true',
                       help='Decode a local video file or a YouTube/yt-dlp URL '
                            '(streamed directly, no temp file)')
    mode.add_argument('--trip', '--roundtrip', dest='trip', action='store_true',
                       help='Encode, then decode, then verify the result is '
                            'byte-identical to the original (self-test)')

    p.add_argument('input',
                   help='Input file path (-e/--trip), local video path (-d), '
                        'or URL to stream from (-d)')
    p.add_argument('output', nargs='?', default=None,
                    help='Output path (optional). Defaults: encode -> <input>.mp4, '
                         'decode -> original filename from manifest (in CWD for URLs).')

    p.add_argument('--nvenc', action='store_true',
                   help='Use NVIDIA NVENC (hevc_nvenc) GPU encoder instead of x265')

    return p


def main():
    install_signal_handlers()
    parser = build_parser()
    args = parser.parse_args()

    if args.encode:
        output = args.output or (args.input + '.mp4')
        encode(args.input, output, fps=ENCODE_FPS, k=ZFEC_K, m=ZFEC_M,
               crf=CPU_CRF, preset=CPU_PRESET, use_nvenc=args.nvenc,
               qp=NVENC_QP, nvenc_preset=NVENC_PRESET,
               bitrate=None if args.nvenc else CPU_BITRATE)

    elif args.decode:
        # Pass None to let decode() derive the output name from the manifest
        # frame (original basename + DECODED_SUFFIX).
        output = args.output if args.output else None
        decode(args.input, output, workers=None, verbose=False)

    elif args.trip:
        video_path   = args.output or (args.input + '.mp4')
        decoded_path = args.input + '.roundtrip'
        roundtrip(args.input, video_path, decoded_path,
                  fps=ENCODE_FPS, k=ZFEC_K, m=ZFEC_M,
                  workers=None, crf=CPU_CRF, preset=CPU_PRESET,
                  use_nvenc=args.nvenc, qp=NVENC_QP,
                  nvenc_preset=NVENC_PRESET,
                  verbose=False,
                  bitrate=None if args.nvenc else CPU_BITRATE)


if __name__ == '__main__':
    # Required on Windows and macOS (multiprocessing 'spawn' method) so that
    # worker processes importing YoutubeTranscoder as a module do not
    # re-execute main().  On Linux ('fork') this guard is harmless.
    main()
