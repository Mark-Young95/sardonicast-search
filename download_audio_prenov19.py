# download_audio.py
import os
from pathlib import Path
import yt_dlp

def download_podcast_audio(video_ids: list[str], output_dir: str = "audio_cache"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        # Pull ONLY the best audio stream
        "format": "bestaudio/best",

        # Point to the containing folder, NOT ffmpeg.exe directly
        "ffmpeg_location": r"C:\Users\markw\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin",

        # Bypass web player cipher & bot check
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios"],
            }
        },

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
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for vid in video_ids:
            wav_file = out_path / f"{vid}.wav"
            if wav_file.exists():
                print(f"✅ Already downloaded: {vid}.wav")
                continue
            url = f"https://www.youtube.com/watch?v={vid}"
            print(f"⬇️ Downloading audio stream for {vid}...")
            try:
                ydl.download([url])
            except Exception as e:
                print(f"❌ Failed {vid}: {e}")

if __name__ == "__main__":
    # Example test batch
    test_ids = ["YO9wkuvTsJE", "cKIXdtntMac"] #["qUOExb6VZS4"] #["WW2XiHenNTI", "_w5b_rSMZKA"] #["0Z7GcEBmFIo"]  #["xtSnzvJFyYU"] #["6LHCYUmb-xc", "rM6qPSeEvcA", "Wl-4iuEgXoo", "Q_jR1U82wvc", "AZuzd3RWxTA", "abYG_qkqhHk", "_w5b_rSMZKA", "joOYxzdUhAk", "Ds_FVmolrrM", "x1cfOLa6_kI"]
    download_podcast_audio(test_ids)

# 200 livestream
#modal run modal_whisperx_worker.py --video-id YO9wkuvTsJE --centroids-file host_centroids.json
# 5 needledrop
#modal run modal_whisperx_worker.py --video-id cKIXdtntMac --centroids-file host_centroids.json
# 28 madagascar
#modal run modal_whisperx_worker.py --video-id abYG_qkqhHk --centroids-file host_centroids.json
# 162 talking pony
#modal run modal_whisperx_worker.py --video-id rM6qPSeEvcA --centroids-file host_centroids.json
# 226 bad boy bubby
#modal run modal_whisperx_worker.py --video-id 6LHCYUmb-xc --centroids-file host_centroids.json



# modal run modal_whisperx_worker.py --video-ids abYG_qkqhHk,YO9wkuvTsJE,6LHCYUmb-xc,rM6qPSeEvcA



# 224-228
# modal run modal_whisperx_worker.py --video-ids rJFforuuODU,cTQycmFvUSY,3cNkmDJC9DI,hOBqxvI167k

# 220-223
# modal run modal_whisperx_worker.py --video-ids nmKCUFGq0hA,9OZAYwBtTHo,b6F755TZa6g,3AMXphnjJGc

# 215-219
# modal run modal_whisperx_worker.py --video-ids kdlzHIbwQfs,Ojz0sskxFBo,5rL7bBwOuio,FXv2XUshrNE,WCA8Jt_qrG4

# 210-214
# modal run modal_whisperx_worker.py --video-ids 7qxWAC4A0yk,xVFrlN1aQIE,egz-Hx4iLDw,AExOwJa58no,0OYydixtVA8

# 205-209
# modal run modal_whisperx_worker.py --video-ids leOu1G1faBE,aQBFixFqv6M,HPVFbp_CBFA,AnNI-4ihV9c,6jGjbEm99CI

# 199-204
# modal run modal_whisperx_worker.py --video-ids 9MTD8kOpGhk,nMwkk5jozAw,6QxmbwpFRF4,8nP1uXKmwSY,7nlPtTUtHrw





# 194-198
# modal run modal_whisperx_worker.py --video-ids 1MVRQj2vr9Q,SYjANODJkzc,u4_9_elgJyg,4IP8hfMPzas,xtSnzvJFyYU

# 189-193
# modal run modal_whisperx_worker.py --video-ids tVG-PoH-n80,AhY_XvQt2KU,BZLbkzMX57Y,NiTtWpPrVp4,kRQHSycmvd0

# 184-188
# modal run modal_whisperx_worker.py --video-ids abxgAZ2a-RQ,ANsRcUCwrbI,Zb9bB0t4tyU,lScrhK9Za6k,e4e0e7APJT0

# 179-183
# modal run modal_whisperx_worker.py --video-ids AuUHVdAKS_Y,_bvT65wMeoE,mPWvB0WEwZw,NP-YRmcWzqE,bUKJkaLWeGw

# 174-178
# modal run modal_whisperx_worker.py --video-ids VfBes3X1ovk,LT_rSGNHHtA,b7OrCQP5EC4,anKdXQ-kfyc,bvxlnZ5UFcs

# 169-173
# modal run modal_whisperx_worker.py --video-ids 6O5bEy7_bSM,YXdqlGFOmNw,WZP1ab8i9nk,stfN4kzaDcw,PWyWirZiHsg

# 164-168
# modal run modal_whisperx_worker.py --video-ids Xuu-qqq25hU,RFetquVwQiE,a4YvDSgzUYg,cS0jXqcBZFg,dpwbfTta0XE

# 158-163
# modal run modal_whisperx_worker.py --video-ids stCaFJ77Nc0,TVZBFjs8EAI,NWRf6vTheB4,0dRTNWui2qM,Tuiv84ZDRyM

# 153-157
# modal run modal_whisperx_worker.py --video-ids W43k4lgx6pM,ym-hIVRp-4Q,Wl-4iuEgXoo,01Fcmcwk15E,9FqJN-6iq9I

# try 50: 103-152
# started nov 22nd, 4:32pm, at 6:22ish, won't let me use anymore... not sure, need to call modal?
# modal run modal_whisperx_worker.py --video-ids 2y77PyEVsUY,t9O3Rtfqrkg,ZgJCKrCrIvs,ohzJhlzULU8,RxrhZNcbPss,JjVLu6gW1QA,p26Jg_0aZME,joOYxzdUhAk,K6qLLeiOwuA,RLm3t3V-0Ys,kSuuuxolw-U,HjWABCVmzU0,74RSO6AKcIY,LArpDeGO7L4,m5BkmRovPfQ,Kgiw3C2e7W0,B4K72jClGhE,04K88bwT558,S4gLhgS_Btc,3YKhtvSLvLo,WELb_tDvSf0,UxaoKwWDUvI,wIkPmEqewRs,HnDnmxspTFo,n8uX0X9eiVg,cis2raB_Bdo,Xaal5pQ-Zl0,Nf2wmGN3mkQ,QKA13IdBZd4,Ce3IJL3pPwg,cTNEIRS50D8,vxObcVWDpJY,FGNmMheGw8c,1s5SWZNR0Z4,QCv6CJNVJi8,fExN7E8vtZ0,Ds_FVmolrrM,nnC-qeutqFM,ZEz20RdBlQA,Q_jR1U82wvc,f-OmVBjIC5M,jdfSNUgJxCI,a9lovmzo94g,g66tJLCQmlQ,leN-vjrW1eU,I7IycKwKZ0c,VqoBXdHw87g,QQ7X33bzLBQ,GkAxbUVqHVQ,STFAcONCXRM

# 148-152
# modal run modal_whisperx_worker.py --video-ids I7IycKwKZ0c,VqoBXdHw87g,QQ7X33bzLBQ,GkAxbUVqHVQ,STFAcONCXRM,





# 143-147
# modal run modal_whisperx_worker.py --video-ids f-OmVBjIC5M,jdfSNUgJxCI,a9lovmzo94g,g66tJLCQmlQ,leN-vjrW1eU

# 138-142
# modal run modal_whisperx_worker.py --video-ids fExN7E8vtZ0,Ds_FVmolrrM,nnC-qeutqFM,ZEz20RdBlQA,Q_jR1U82wvc

# 133-137
# modal run modal_whisperx_worker.py --video-ids cTNEIRS50D8,vxObcVWDpJY,FGNmMheGw8c,1s5SWZNR0Z4,QCv6CJNVJi8

# 128-132
# modal run modal_whisperx_worker.py --video-ids cis2raB_Bdo,Xaal5pQ-Zl0,Nf2wmGN3mkQ,QKA13IdBZd4,Ce3IJL3pPwg

# 123-127
# modal run modal_whisperx_worker.py --video-ids WELb_tDvSf0,UxaoKwWDUvI,wIkPmEqewRs,HnDnmxspTFo,n8uX0X9eiVg

# 118-122
# modal run modal_whisperx_worker.py --video-ids Kgiw3C2e7W0,B4K72jClGhE,04K88bwT558,S4gLhgS_Btc,3YKhtvSLvLo

# 113-117
# modal run modal_whisperx_worker.py --video-ids kSuuuxolw-U,HjWABCVmzU0,74RSO6AKcIY,LArpDeGO7L4,m5BkmRovPfQ

# 108-112
# modal run modal_whisperx_worker.py --video-ids JjVLu6gW1QA,p26Jg_0aZME,joOYxzdUhAk,K6qLLeiOwuA,RLm3t3V-0Ys

# 103-107
# modal run modal_whisperx_worker.py --video-ids 2y77PyEVsUY,t9O3Rtfqrkg,ZgJCKrCrIvs,ohzJhlzULU8,RxrhZNcbPss

# try 30: 73-102
# started nov 21st, 9:18pm. finished right at 10:45pm! 30 (including 6hr one) took 87 minutes on 5 containers
# modal run modal_whisperx_worker.py --video-ids h3xPjF4TGWk,dZrdd2MmkNs,6Mx9pSn-Y0g,obNj3gaJros,qU36qE6JaG0,PJIhUkG-bAY,z0dUUHx7TJE,LjtFPxabYvY,yBq71Tc7R8I,x1cfOLa6_kI,xAV2jXtKjW8,rxshEE7NC7E,NTxV9b4V0Ow,qR6gD-esmbE,qzwXZqZqcpI,zrry2eQp1_8,gHkeMxKr1xM,0M15ka97iVA,czkI09RVRjA,5SOziaQapMI,kzsb9SlOqb0,tuNhA-H1zbs,_5zw4p257rY,16LuBAskfOs,xWxy9I5JGsM,igEAXji4CK8,rBlyN8R6zFU,99kau024yso,rf2FNwHif_I,BJLFZhHN3Ls

# 98-102
# modal run modal_whisperx_worker.py --video-ids igEAXji4CK8,rBlyN8R6zFU,99kau024yso,rf2FNwHif_I,BJLFZhHN3Ls





# 93-97
# modal run modal_whisperx_worker.py --video-ids kzsb9SlOqb0,tuNhA-H1zbs,_5zw4p257rY,16LuBAskfOs,xWxy9I5JGsM

# 88-92
# modal run modal_whisperx_worker.py --video-ids zrry2eQp1_8,gHkeMxKr1xM,0M15ka97iVA,czkI09RVRjA,5SOziaQapMI

# 83-87
# modal run modal_whisperx_worker.py --video-ids xAV2jXtKjW8,rxshEE7NC7E,NTxV9b4V0Ow,qR6gD-esmbE,qzwXZqZqcpI

# 78-82
# modal run modal_whisperx_worker.py --video-ids PJIhUkG-bAY,z0dUUHx7TJE,LjtFPxabYvY,yBq71Tc7R8I,x1cfOLa6_kI

# 73-77
# modal run modal_whisperx_worker.py --video-ids h3xPjF4TGWk,dZrdd2MmkNs,6Mx9pSn-Y0g,obNj3gaJros,qU36qE6JaG0

# combined 53-72
# started nov 19th, 6:10pm, 
# modal run modal_whisperx_worker.py --video-ids DvgONdaCpe0,bAw40bYAbuQ,v_V-8gBwQ0M,KprWFzZHuvs,kubdjG-4N0Q,bzblLkTJJmM,0rx44FdnEIY,AZuzd3RWxTA,XwN6xIEf3Es,L2yoBj6k2gw,zxUEH_Nak8Y,WW2XiHenNTI,rXb8vjfszLY,jtuSSzFx4bA,BgPz4MmstQQ,miwkDPCDO-E,yoINw3EkUg8,9KpIgO6y2PQ,js7B-K6BtJs,IybB6sCakxI

# 68-72
# modal run modal_whisperx_worker.py --video-ids miwkDPCDO-E,yoINw3EkUg8,9KpIgO6y2PQ,js7B-K6BtJs,IybB6sCakxI

# 63-67
# modal run modal_whisperx_worker.py --video-ids zxUEH_Nak8Y,WW2XiHenNTI,rXb8vjfszLY,jtuSSzFx4bA,BgPz4MmstQQ

# 58-62
# modal run modal_whisperx_worker.py --video-ids bzblLkTJJmM,0rx44FdnEIY,AZuzd3RWxTA,XwN6xIEf3Es,L2yoBj6k2gw

# 53-57
# modal run modal_whisperx_worker.py --video-ids DvgONdaCpe0,bAw40bYAbuQ,v_V-8gBwQ0M,KprWFzZHuvs,kubdjG-4N0Q

# combined 33-52
# started nov 19th, 4:42pm, finished 6:08pm (3 containers)
# modal run modal_whisperx_worker.py --video-ids WpC75FsTKgI,XknVaaXJY_k,djAw3gYe0eA,MH0_ccLF_wQ,jhsLkaovRf4,aVbILMqNxgQ,Xdrf3vy7Edk,A0Wy9u6DPj8,kTElzwepV88,9dyGSglqflk,yp8GigxUc8A,JtEkJ22uDAY,MCwW-vnG5S8,GDiZzicN71w,jRGf2_A9jxI,1PaTzu-auA8,nK33Iu1WeVs,XnRdkcs7Tqo,tajqOLlRWq0,w-OlJK-GRyc

# 48-52
# modal run modal_whisperx_worker.py --video-ids 1PaTzu-auA8,nK33Iu1WeVs,XnRdkcs7Tqo,tajqOLlRWq0,w-OlJK-GRyc





# 43-47
# modal run modal_whisperx_worker.py --video-ids yp8GigxUc8A,JtEkJ22uDAY,MCwW-vnG5S8,GDiZzicN71w,jRGf2_A9jxI

# 38-42
# modal run modal_whisperx_worker.py --video-ids aVbILMqNxgQ,Xdrf3vy7Edk,A0Wy9u6DPj8,kTElzwepV88,9dyGSglqflk

# 33-37
# modal run modal_whisperx_worker.py --video-ids WpC75FsTKgI,XknVaaXJY_k,djAw3gYe0eA,MH0_ccLF_wQ,jhsLkaovRf4

# combined  17-32
# started nov 19th, 2:07pm finished by  4:39pm (2 containers)
# modal run modal_whisperx_worker.py --video-ids 9TVhA0iKI-8,kjtJIsUpoSQ,aGBDLj4LWEY,9JIveRHOVVQ,bsK6U6Lpbic,50a38cwyvdM,Z1rsMoca8Jo,V2YtTeafjvE,0Seji1U6-4g,M_XUa3L8Z4E,_w5b_rSMZKA,KHL6MIhGGL4,qk6gQ0Be5ac,aMSwB0hx4CE,8zNF77wosvQ

# 27-32
# modal run modal_whisperx_worker.py --video-ids _w5b_rSMZKA,KHL6MIhGGL4,qk6gQ0Be5ac,aMSwB0hx4CE,8zNF77wosvQ

# 22-26
# modal run modal_whisperx_worker.py --video-ids 50a38cwyvdM,Z1rsMoca8Jo,V2YtTeafjvE,0Seji1U6-4g,M_XUa3L8Z4E

# 17-21
# modal run modal_whisperx_worker.py --video-ids 9TVhA0iKI-8,kjtJIsUpoSQ,aGBDLj4LWEY,9JIveRHOVVQ,bsK6U6Lpbic

# 12-16
# started nov 19th, 11:32am messed with usage limit mid-run... finished by 2:06pm
# modal run modal_whisperx_worker.py --video-ids MwdXyumoXiE,R6oPFvxIoJk,ut4ZvbEnzuU,7bSKhAzFusU,JzS6NDPTuuM

# 7-11
# started nov 19th, 10:20am finished by 11:31am
# modal run modal_whisperx_worker.py --video-ids krAxPEEpLTc,sqyyO9kIQV8,yfGkuJBY2SE,9nkBXAyaJUQ,Dj4Mg9FeE-M

# 1-6
# started nov 19th, 8:40am finished by 9:33am
# modal run modal_whisperx_worker.py --video-ids URfbzdgxFmE,cSkEMK0Z3Eo,s5_aI6T9OIQ,lRCDT1iMdRg,1V1yTmH44XI