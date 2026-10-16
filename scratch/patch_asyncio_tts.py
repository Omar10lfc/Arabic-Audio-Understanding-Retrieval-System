with open("scripts/build_all_final_notebooks.py", "r", encoding="utf-8") as f:
    text = f.read()

old_block = """async def synthesize_clip(text, out_path, max_retries=5):
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

# 1. Synthesize XL-Sum articles
print(f"--- Synthesizing {len(xlsum_articles)} XL-Sum articles ---")
for idx, art in enumerate(xlsum_articles):
    aid = f"xlsum_{idx:03d}_{art['id']}"
    wav_p = f"audio_tts/xlsum/{aid}.mp3"
    art['audio_path'] = wav_p
    if aid not in tts_manifest or not os.path.exists(wav_p) or os.path.getsize(wav_p) < 500:
        asyncio.run(synthesize_clip(art['text'], wav_p))
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
        asyncio.run(synthesize_clip(p['text'], wav_p))
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
        asyncio.run(synthesize_clip(q['query'], wav_p))"""

new_block = """async def _synthesize_clip_async(text, out_path, max_retries=5):
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
    import concurrent.futures
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
        synthesize_clip(q['query'], wav_p)"""

assert old_block in text, "old_block not found in scripts/build_all_final_notebooks.py"
updated = text.replace(old_block, new_block)
with open("scripts/build_all_final_notebooks.py", "w", encoding="utf-8") as f:
    f.write(updated)
print("Successfully updated scripts/build_all_final_notebooks.py")
