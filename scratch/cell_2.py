# ── CELL 3: NEURAL AUDIO GENERATION (XL-SUM & ARCD CORPUS) ───────────────
import asyncio, time, json, os, glob, edge_tts, soundfile as sf, numpy as np
import concurrent.futures

VOICE = "ar-SA-HamedNeural"
manifest_path = os.path.join(OUT_DIR, "tts_manifest.json")
tts_manifest = {}
if os.path.exists(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        tts_manifest = json.load(f)

async def _synthesize_clip_async(text, out_path, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            communicate = edge_tts.Communicate(text, VOICE)
            await communicate.save(out_path)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
                return True
        except Exception as e:
            if attempt == max_retries:
                print(f"[ERROR] Failed synthesizing {out_path}: {e}")
                raise
            await asyncio.sleep(1.0 * attempt)
    return False

def run_async(coro):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()

def synthesize_clip(text, out_path, max_retries=5):
    return run_async(_synthesize_clip_async(text, out_path, max_retries))

# 1. Synthesize XL-Sum articles
print(f"--- Synthesizing {len(xlsum_articles)} XL-Sum articles ---")
for idx, art in enumerate(xlsum_articles):
    aid = f"xlsum_{idx:03d}_{art['id']}"
    wav_p = f"audio_tts/xlsum/{aid}.mp3"
    art['audio_path'] = wav_p
    if aid not in tts_manifest or not os.path.exists(wav_p) or os.path.getsize(wav_p) < 500:
        synthesize_clip(art['text'], wav_p)
        try:
            dur = sf.info(wav_p).duration
        except Exception:
            dur = 0.0
        tts_manifest[aid] = {
            "type": "xlsum",
            "id": art['id'],
            "audio_path": wav_p,
            "duration_s": round(dur, 2),
            "voice": VOICE,
            "text_len": len(art['text'])
        }
    if (idx + 1) % 10 == 0 or (idx + 1) == len(xlsum_articles):
        print(f"  XL-Sum articles synthesized: {idx + 1}/{len(xlsum_articles)}")

# 2. Synthesize ARCD passages
print(f"--- Synthesizing {len(passages)} ARCD passages ---")
for idx, p in enumerate(passages):
    pid = p['id']
    wav_p = f"audio_tts/arcd_passages/{pid}.mp3"
    p['audio_path'] = wav_p
    if pid not in tts_manifest or not os.path.exists(wav_p) or os.path.getsize(wav_p) < 500:
        synthesize_clip(p['text'], wav_p)
        try:
            dur = sf.info(wav_p).duration
        except Exception:
            dur = 0.0
        tts_manifest[pid] = {
            "type": "arcd_passage",
            "id": pid,
            "audio_path": wav_p,
            "duration_s": round(dur, 2),
            "voice": VOICE,
            "text_len": len(p['text'])
        }
    if (idx + 1) % 10 == 0 or (idx + 1) == len(passages):
        print(f"  ARCD passages synthesized: {idx + 1}/{len(passages)}")

# 3. Synthesize ARCD queries
print(f"--- Synthesizing {len(queries)} ARCD queries ---")
for idx, q in enumerate(queries):
    qid = f"query_{idx:03d}"
    wav_p = f"audio_tts/arcd_queries/{qid}.mp3"
    q['audio_path'] = wav_p
    if qid not in tts_manifest or not os.path.exists(wav_p) or os.path.getsize(wav_p) < 500:
        synthesize_clip(q['query'], wav_p)
        try:
            dur = sf.info(wav_p).duration
        except Exception:
            dur = 0.0
        tts_manifest[qid] = {
            "type": "arcd_query",
            "id": qid,
            "target_id": q['target_id'],
            "audio_path": wav_p,
            "duration_s": round(dur, 2),
            "voice": VOICE,
            "text_len": len(q['query'])
        }
    if (idx + 1) % 10 == 0 or (idx + 1) == len(queries):
        print(f"  ARCD queries synthesized: {idx + 1}/{len(queries)}")

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(tts_manifest, f, indent=2, ensure_ascii=False)

total_audio_sec = sum(item["duration_s"] for item in tts_manifest.values())
print()
print(f"[OK] Neural TTS synthesis complete! Total clips: {len(tts_manifest)} ({total_audio_sec:.1f} s / {total_audio_sec / 60:.1f} min)")
print(f"Manifest written to: {manifest_path}")