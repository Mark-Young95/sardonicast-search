# download_audio.py
import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Optional
import yt_dlp

WINGET_FFMPEG_BIN = Path(
    r"C:\Users\markw\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin"
)


def extract_video_id(item: str) -> str:
    """Accepts a full YouTube URL or a raw 11-char video ID and returns the video ID."""
    item = item.strip().strip('"').strip("'")
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", item):
        return item
    match = re.search(r"(?:v=|youtu\.be/|/live/|/shorts/|/embed/)([A-Za-z0-9_-]{11})", item)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract a YouTube video ID from: '{item}'")


def find_ffmpeg_location() -> Optional[str]:
    """Resolves the FFmpeg bin directory: local ./bin -> WinGet path -> system PATH."""
    local_bin = Path("bin")
    if (local_bin / "ffmpeg.exe").exists() or (local_bin / "ffmpeg").exists():
        return str(local_bin.resolve())
    if (WINGET_FFMPEG_BIN / "ffmpeg.exe").exists():
        return str(WINGET_FFMPEG_BIN)
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return str(Path(sys_ffmpeg).parent)
    return None


def download_podcast_audio(
    inputs: list[str],
    output_dir: str = "audio_cache",
    browser: str = "firefox",
    cookies_file: str = "",
    overwrite: bool = False,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ffmpeg_loc = find_ffmpeg_location()
    if not ffmpeg_loc:
        raise RuntimeError("FFmpeg not found in ./bin, WinGet packages, or system PATH.")

    deno_path = shutil.which("deno")

    ydl_opts = {
        "format": "bestaudio/best",
        "ffmpeg_location": ffmpeg_loc,
        "outtmpl": str(out_path / "%(id)s.%(ext)s"),
        "noplaylist": True,
        # Convert directly to 16kHz mono WAV for WhisperX & Pyannote
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "postprocessor_args": {
            "ExtractAudio": [
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",      # Force mono channel
            ],
        },
        # Bot-check & rate-limit pacing (10-20s sleep between downloads, 2s between API calls, 2 MB/s cap)
        "sleep_interval_requests": 2,
        "sleep_interval": 10,
        "max_sleep_interval": 20,
        "ratelimit": 2_000_000,
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 60,
        "quiet": False,
        # Intentionally NO extractor_args/player_client override so yt-dlp uses working defaults
    }

    if deno_path:
        ydl_opts["js_runtimes"] = {"deno": {"path": deno_path}}
    else:
        print("⚠️ Warning: 'deno' not found on PATH. Install via: pip install deno")

    if cookies_file:
        ydl_opts["cookiefile"] = str(Path(cookies_file).resolve())
    elif browser:
        ydl_opts["cookiesfrombrowser"] = (browser,)

    succeeded, skipped, failed = [], [], []

    for raw_item in inputs:
        try:
            vid = extract_video_id(raw_item)
        except ValueError as e:
            print(f"❌ {e}")
            failed.append(raw_item)
            continue

        wav_file = out_path / f"{vid}.wav"
        if wav_file.exists() and wav_file.stat().st_size >= 1024 and not overwrite:
            print(f"✅ Already downloaded: {wav_file}")
            skipped.append(vid)
            continue

        # Clean up any leftover .part files from interrupted runs
        for part_file in out_path.glob(f"{vid}.*.part"):
            part_file.unlink(missing_ok=True)

        url = f"https://www.youtube.com/watch?v={vid}"
        print(f"\n⬇️ Downloading audio stream for {vid} ({url})...")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            if wav_file.exists() and wav_file.stat().st_size >= 1024:
                size_mb = wav_file.stat().st_size / (1024 ** 2)
                print(f"🎉 Saved 16kHz mono WAV: {wav_file} ({size_mb:.1f} MB)")
                succeeded.append(vid)
            else:
                raise FileNotFoundError(f"yt-dlp finished, but '{wav_file}' was not created.")
        except Exception as e:
            print(f"❌ Failed {vid} (continuing batch): {e}")
            failed.append(vid)

    print("\n" + "=" * 60)
    print(f"📊 Download Summary: {len(succeeded)} downloaded | {len(skipped)} already cached | {len(failed)} failed")
    if failed:
        print(f"🔁 Retry failed:\n   python download_audio.py --urls \"{','.join(failed)}\"")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download YouTube podcast audio as 16kHz mono WAV using Firefox cookies + Deno."
    )
    parser.add_argument(
        "positional_urls",
        nargs="*",
        help="Optional space-separated YouTube URLs or video IDs.",
    )
    parser.add_argument(
        "--urls",
        "--video-ids",
        dest="urls",
        default="",
        help="Comma-separated YouTube URLs or video IDs (matches modal_whisperx_worker style).",
    )
    parser.add_argument("--output-dir", default="audio_cache", help="Target directory for WAV files.")
    parser.add_argument("--browser", default="firefox", help="Browser to extract cookies from (default: firefox).")
    parser.add_argument("--cookies-file", default="", help="Optional path to cookies.txt (overrides --browser).")
    parser.add_argument("--overwrite", action="store_true", help="Re-download even if WAV already exists.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    items = []
    if args.urls:
        items.extend(x.strip() for x in args.urls.split(",") if x.strip())
    if args.positional_urls:
        for arg in args.positional_urls:
            items.extend(x.strip() for x in arg.split(",") if x.strip())

    if not items:
        print("Usage:")
        print('  python download_audio.py --urls "https://www.youtube.com/watch?v=YO9wkuvTsJE,https://youtu.be/krAxPEEpLTc"')
        print('  python download_audio.py --video-ids "YO9wkuvTsJE,krAxPEEpLTc"')
        sys.exit(1)

    download_podcast_audio(
        items,
        output_dir=args.output_dir,
        browser=args.browser,
        cookies_file=args.cookies_file,
        overwrite=args.overwrite,
    )


# Sard 1: Bonus oscars discussion
# python download_audio.py --video-ids pO0gVwXDPfg

    # 224-228
# python download_audio.py --video-ids rJFforuuODU,cTQycmFvUSY,3cNkmDJC9DI,hOBqxvI167k

# 220-223
# python download_audio.py --video-ids nmKCUFGq0hA,9OZAYwBtTHo,b6F755TZa6g,3AMXphnjJGc

# 215-219
# python download_audio.py --video-ids kdlzHIbwQfs,Ojz0sskxFBo,5rL7bBwOuio,FXv2XUshrNE,WCA8Jt_qrG4

# 210-214
# python download_audio.py --video-ids 7qxWAC4A0yk,xVFrlN1aQIE,egz-Hx4iLDw,AExOwJa58no,0OYydixtVA8

# 205-209
# python download_audio.py --video-ids leOu1G1faBE,aQBFixFqv6M,HPVFbp_CBFA,AnNI-4ihV9c,6jGjbEm99CI

# 199-204
# python download_audio.py --video-ids 9MTD8kOpGhk,nMwkk5jozAw,6QxmbwpFRF4,8nP1uXKmwSY,7nlPtTUtHrw





# 194-198
# python download_audio.py --video-ids 1MVRQj2vr9Q,SYjANODJkzc,u4_9_elgJyg,4IP8hfMPzas,xtSnzvJFyYU

# 189-193
# python download_audio.py --video-ids tVG-PoH-n80,AhY_XvQt2KU,BZLbkzMX57Y,NiTtWpPrVp4,kRQHSycmvd0

# 184-188
# python download_audio.py --video-ids abxgAZ2a-RQ,ANsRcUCwrbI,Zb9bB0t4tyU,lScrhK9Za6k,e4e0e7APJT0

# 179-183
# python download_audio.py --video-ids AuUHVdAKS_Y,_bvT65wMeoE,mPWvB0WEwZw,NP-YRmcWzqE,bUKJkaLWeGw

# 174-178
# python download_audio.py --video-ids VfBes3X1ovk,LT_rSGNHHtA,b7OrCQP5EC4,anKdXQ-kfyc,bvxlnZ5UFcs

# 169-173
# python download_audio.py --video-ids 6O5bEy7_bSM,YXdqlGFOmNw,WZP1ab8i9nk,stfN4kzaDcw,PWyWirZiHsg

# 164-168
# python download_audio.py --video-ids Xuu-qqq25hU,RFetquVwQiE,a4YvDSgzUYg,cS0jXqcBZFg,dpwbfTta0XE

# 158-163
# python download_audio.py --video-ids stCaFJ77Nc0,TVZBFjs8EAI,NWRf6vTheB4,0dRTNWui2qM,Tuiv84ZDRyM

# 153-157
# python download_audio.py --video-ids W43k4lgx6pM,ym-hIVRp-4Q,Wl-4iuEgXoo,01Fcmcwk15E,9FqJN-6iq9I

# 148-152
# python download_audio.py --video-ids I7IycKwKZ0c,VqoBXdHw87g,QQ7X33bzLBQ,GkAxbUVqHVQ,STFAcONCXRM





# 143-147
# python download_audio.py --video-ids f-OmVBjIC5M,jdfSNUgJxCI,a9lovmzo94g,g66tJLCQmlQ,leN-vjrW1eU

# 138-142
# python download_audio.py --video-ids fExN7E8vtZ0,Ds_FVmolrrM,nnC-qeutqFM,ZEz20RdBlQA,Q_jR1U82wvc

# 133-137
# python download_audio.py --video-ids cTNEIRS50D8,vxObcVWDpJY,FGNmMheGw8c,1s5SWZNR0Z4,QCv6CJNVJi8

# 128-132
# python download_audio.py --video-ids cis2raB_Bdo,Xaal5pQ-Zl0,Nf2wmGN3mkQ,QKA13IdBZd4,Ce3IJL3pPwg

# 123-127
# python download_audio.py --video-ids WELb_tDvSf0,UxaoKwWDUvI,wIkPmEqewRs,HnDnmxspTFo,n8uX0X9eiVg

# 118-122
# python download_audio.py --video-ids Kgiw3C2e7W0,B4K72jClGhE,04K88bwT558,S4gLhgS_Btc,3YKhtvSLvLo

# 113-117
# python download_audio.py --video-ids kSuuuxolw-U,HjWABCVmzU0,74RSO6AKcIY,LArpDeGO7L4,m5BkmRovPfQ

# 108-112
# python download_audio.py --video-ids JjVLu6gW1QA,p26Jg_0aZME,joOYxzdUhAk,K6qLLeiOwuA,RLm3t3V-0Ys

# 103-107
# python download_audio.py --video-ids 2y77PyEVsUY,t9O3Rtfqrkg,ZgJCKrCrIvs,ohzJhlzULU8,RxrhZNcbPss

# 98-102
# python download_audio.py --video-ids igEAXji4CK8,rBlyN8R6zFU,99kau024yso,rf2FNwHif_I,BJLFZhHN3Ls





# 93-97
# python download_audio.py --video-ids kzsb9SlOqb0,tuNhA-H1zbs,_5zw4p257rY,16LuBAskfOs,xWxy9I5JGsM

# 88-92
# python download_audio.py --video-ids zrry2eQp1_8,gHkeMxKr1xM,0M15ka97iVA,czkI09RVRjA,5SOziaQapMI

# 83-87
# python download_audio.py --video-ids xAV2jXtKjW8,rxshEE7NC7E,NTxV9b4V0Ow,qR6gD-esmbE,qzwXZqZqcpI

# 78-82
# python download_audio.py --video-ids PJIhUkG-bAY,z0dUUHx7TJE,LjtFPxabYvY,yBq71Tc7R8I,x1cfOLa6_kI

# 73-77
# python download_audio.py --video-ids h3xPjF4TGWk,dZrdd2MmkNs,6Mx9pSn-Y0g,obNj3gaJros,qU36qE6JaG0

# 68-72
# python download_audio.py --video-ids miwkDPCDO-E,yoINw3EkUg8,9KpIgO6y2PQ,js7B-K6BtJs,IybB6sCakxI

# 63-67
# python download_audio.py --video-ids zxUEH_Nak8Y,WW2XiHenNTI,rXb8vjfszLY,jtuSSzFx4bA,BgPz4MmstQQ

# 58-62
# python download_audio.py --video-ids bzblLkTJJmM,0rx44FdnEIY,AZuzd3RWxTA,XwN6xIEf3Es,L2yoBj6k2gw

# 53-57
# python download_audio.py --video-ids DvgONdaCpe0,bAw40bYAbuQ,v_V-8gBwQ0M,KprWFzZHuvs,kubdjG-4N0Q

# 48-52
# modal run modal_whisperx_worker.py --video-ids 1PaTzu-auA8,nK33Iu1WeVs,XnRdkcs7Tqo,tajqOLlRWq0,w-OlJK-GRyc





# 43-47
# python download_audio.py --video-ids yp8GigxUc8A,JtEkJ22uDAY,MCwW-vnG5S8,GDiZzicN71w,jRGf2_A9jxI

# 38-42
# python download_audio.py --video-ids aVbILMqNxgQ,Xdrf3vy7Edk,A0Wy9u6DPj8,kTElzwepV88,9dyGSglqflk

# 33-37
# python download_audio.py --video-ids WpC75FsTKgI,XknVaaXJY_k,djAw3gYe0eA,MH0_ccLF_wQ,jhsLkaovRf4

# 27-32
# python download_audio.py --video-ids _w5b_rSMZKA,KHL6MIhGGL4,qk6gQ0Be5ac,aMSwB0hx4CE,8zNF77wosvQ

# 22-26
# python download_audio.py --video-ids 50a38cwyvdM,Z1rsMoca8Jo,V2YtTeafjvE,0Seji1U6-4g,M_XUa3L8Z4E

# 17-21
# python download_audio.py --video-ids 9TVhA0iKI-8,kjtJIsUpoSQ,aGBDLj4LWEY,9JIveRHOVVQ,bsK6U6Lpbic