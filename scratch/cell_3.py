# ── CELL 4: MULTI-MODEL WHISPER ASR TRANSCRIPTION (4 ARCHITECTURES) ──────
import time, gc, torch, numpy as np
from faster_whisper import WhisperModel
from evalkit.normalize import normalize_arabic_legacy, normalize_arabic_v2
from evalkit.wer import compute_dual_wer_record

# Recursion-proof PyAV compatibility patch for newer PyAV versions in Kaggle
try:
    import av
    import av.container.core
    _c_open = av.container.core.open
    def _safe_av_open(*args, **kwargs):
        kwargs.pop('metadata_errors', None)
        kwargs.pop('metadata_encoding', None)
        return _c_open(*args, **kwargs)
    av.container.core.open = _safe_av_open
    av.open = _safe_av_open
except Exception:
    pass

def load_audio_np(audio_path, target_sr=16000):
    if isinstance(audio_path, np.ndarray):
        return audio_path
    # 1. Try soundfile
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            import scipy.signal
            num_samples = int(len(audio) * target_sr / sr)
            audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
        return audio.astype(np.float32)
    except Exception:
        pass
    # 2. Try PyAV container decode with resampler (bypassing metadata_errors)
    try:
        import av
        with av.open(audio_path, mode="r") as container:
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
            chunks = []
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().flatten())
            if chunks:
                return np.concatenate(chunks).astype(np.float32)
    except Exception:
        pass
    # 3. Try torchaudio
    try:
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            import torchaudio.transforms as T
            waveform = T.Resample(sr, target_sr)(waveform)
        return waveform.squeeze().cpu().numpy().astype(np.float32)
    except Exception:
        pass
    return audio_path

# Monkeypatch faster_whisper decode_audio
try:
    import faster_whisper.audio
    import faster_whisper.transcribe
    def _fw_safe_decode(input_file, sampling_rate=16000, split_stereo=False):
        if isinstance(input_file, np.ndarray):
            return input_file
        return load_audio_np(input_file, target_sr=sampling_rate)
    faster_whisper.audio.decode_audio = _fw_safe_decode
    faster_whisper.transcribe.decode_audio = _fw_safe_decode
except Exception:
    pass

asr_out_file = os.path.join(OUT_DIR, "tts_asr_propagation_results.json")
arcd_trans_file = os.path.join(OUT_DIR, "spoken_arcd_transcripts.json")

ct2_models = [
    {
        "name": "Whisper-Large-v3",
        "key": "large_v3",
        "path": os.path.join(MERGED_MODELS_DIR, "ct2_int8/whisper-large-v3-ct2-int8")
    },
    {
        "name": "Whisper-Large-v3-Turbo",
        "key": "turbo",
        "path": os.path.join(MERGED_MODELS_DIR, "ct2_int8/whisper-large-v3-turbo-ct2-int8")
    },
    {
        "name": "Whisper-Medium",
        "key": "medium",
        "path": os.path.join(MERGED_MODELS_DIR, "ct2_int8/whisper-medium-ct2-int8")
    },
    {
        "name": "Whisper-Small",
        "key": "small",
        "path": os.path.join(MERGED_MODELS_DIR, "ct2_int8/whisper-small-ct2-int8")
    }
]

propagation_results = {}
if os.path.exists(asr_out_file):
    with open(asr_out_file, "r", encoding="utf-8") as f:
        propagation_results = json.load(f)

arcd_transcripts = {
    "passages": {p['id']: {"clean_text": p['text'], "models": {}} for p in passages},
    "queries": {f"query_{i:03d}": {"clean_query": q['query'], "target_id": q['target_id'], "models": {}} for i, q in enumerate(queries)}
}
if os.path.exists(arcd_trans_file):
    with open(arcd_trans_file, "r", encoding="utf-8") as f:
        arcd_transcripts = json.load(f)

def transcribe_clip_ct2(model, audio_path):
    t0 = time.time()
    audio_inp = load_audio_np(audio_path)
    segments, info = model.transcribe(
        audio_inp,
        language="ar",
        task="transcribe",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False
    )
    raw_text = " ".join([seg.text for seg in segments]).strip()
    compute_sec = time.time() - t0
    dur_sec = info.duration if hasattr(info, 'duration') and info.duration else 0.0
    if dur_sec == 0.0 and isinstance(audio_inp, np.ndarray):
        dur_sec = float(len(audio_inp) / 16000.0)
    return raw_text, dur_sec, compute_sec

summary_records = []

for m_info in ct2_models:
    m_name = m_info["name"]
    m_key = m_info["key"]
    m_path = m_info["path"]
    
    print()
    print("=======================================================")
    print(f"Transcribing with: {m_name}")
    print(f"Path: {m_path}")
    print("=======================================================")
    
    t_load = time.time()
    whisper_model = WhisperModel(m_path, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")
    from faster_whisper.feature_extractor import FeatureExtractor
    n_mels = 128 if ("large" in m_name.lower() or "turbo" in m_name.lower()) else 80
    whisper_model.feature_extractor = FeatureExtractor(feature_size=n_mels)
    print(f"[OK] Model loaded in {time.time() - t_load:.1f}s (feature_size={n_mels})")
    
    if m_key not in propagation_results:
        propagation_results[m_key] = {"model_name": m_name, "articles": []}
    
    already_done_xl = len(propagation_results[m_key]["articles"])
    print(f"Transcribing XL-Sum articles ({already_done_xl}/{len(xlsum_articles)} already done)...")
    
    total_audio_xl = 0.0
    total_comp_xl = 0.0
    
    for idx, art in enumerate(xlsum_articles):
        if idx < already_done_xl:
            continue
        raw_t, dur, comp = transcribe_clip_ct2(whisper_model, art['audio_path'])
        norm_leg = normalize_arabic_legacy(raw_t)
        norm_v2 = normalize_arabic_v2(raw_t)
        rec = compute_dual_wer_record(art['text'], raw_t)
        
        propagation_results[m_key]["articles"].append({
            "id": art['id'],
            "audio_path": art['audio_path'],
            "clean_text": art['text'],
            "reference_summary": art['summary'],
            "raw_transcript": raw_t,
            "norm_legacy_transcript": norm_leg,
            "norm_v2_transcript": norm_v2,
            "legacy_wer": rec["legacy_wer"],
            "v2_wer": rec["v2_wer"],
            "duration_s": dur,
            "compute_s": comp
        })
        total_audio_xl += dur
        total_comp_xl += comp
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(xlsum_articles):
            avg_w = np.mean([a["legacy_wer"] for a in propagation_results[m_key]["articles"]])
            print(f"  XL-Sum: {idx + 1}/{len(xlsum_articles)} | Avg Legacy WER: {avg_w * 100.0:.2f}%")
            with open(asr_out_file, "w", encoding="utf-8") as f:
                json.dump(propagation_results, f, indent=2, ensure_ascii=False)
    
    # 2. Transcribe ARCD Passages
    print(f"Transcribing ARCD passages ({len(passages)} passages)...")
    total_audio_arcd = 0.0
    total_comp_arcd = 0.0
    
    for idx, p in enumerate(passages):
        pid = p['id']
        if pid in arcd_transcripts["passages"] and m_key in arcd_transcripts["passages"][pid].get("models", {}):
            continue
        raw_t, dur, comp = transcribe_clip_ct2(whisper_model, p['audio_path'])
        norm_leg = normalize_arabic_legacy(raw_t)
        norm_v2 = normalize_arabic_v2(raw_t)
        rec = compute_dual_wer_record(p['text'], raw_t)
        
        if pid not in arcd_transcripts["passages"]:
            arcd_transcripts["passages"][pid] = {"clean_text": p['text'], "models": {}}
        arcd_transcripts["passages"][pid]["models"][m_key] = {
            "raw": raw_t,
            "norm_legacy": norm_leg,
            "norm_v2": norm_v2,
            "legacy_wer": rec["legacy_wer"],
            "v2_wer": rec["v2_wer"],
            "duration_s": dur,
            "compute_s": comp
        }
        total_audio_arcd += dur
        total_comp_arcd += comp
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(passages):
            print(f"  ARCD Passages: {idx + 1}/{len(passages)}")
            with open(arcd_trans_file, "w", encoding="utf-8") as f:
                json.dump(arcd_transcripts, f, indent=2, ensure_ascii=False)
    
    # 3. Transcribe ARCD Queries
    print(f"Transcribing ARCD queries ({len(queries)} queries)...")
    for idx, q in enumerate(queries):
        qid = f"query_{idx:03d}"
        if qid in arcd_transcripts["queries"] and m_key in arcd_transcripts["queries"][qid].get("models", {}):
            continue
        raw_t, dur, comp = transcribe_clip_ct2(whisper_model, q['audio_path'])
        norm_leg = normalize_arabic_legacy(raw_t)
        norm_v2 = normalize_arabic_v2(raw_t)
        rec = compute_dual_wer_record(q['query'], raw_t)
        
        if qid not in arcd_transcripts["queries"]:
            arcd_transcripts["queries"][qid] = {"clean_query": q['query'], "target_id": q['target_id'], "models": {}}
        arcd_transcripts["queries"][qid]["models"][m_key] = {
            "raw": raw_t,
            "norm_legacy": norm_leg,
            "norm_v2": norm_v2,
            "legacy_wer": rec["legacy_wer"],
            "v2_wer": rec["v2_wer"],
            "duration_s": dur,
            "compute_s": comp
        }
        if (idx + 1) % 10 == 0 or (idx + 1) == len(queries):
            print(f"  ARCD Queries: {idx + 1}/{len(queries)}")
            with open(arcd_trans_file, "w", encoding="utf-8") as f:
                json.dump(arcd_transcripts, f, indent=2, ensure_ascii=False)
    
    xl_wer = np.mean([a["legacy_wer"] for a in propagation_results[m_key]["articles"]]) * 100.0
    p_wer = np.mean([arcd_transcripts["passages"][p['id']]["models"][m_key]["legacy_wer"] for p in passages]) * 100.0
    tot_dur = total_audio_xl + total_audio_arcd
    tot_comp = total_comp_xl + total_comp_arcd
    th = tot_dur / tot_comp if tot_comp > 0 else 0.0
    
    summary_records.append({
        "Model": m_name,
        "XL-Sum WER (%)": round(xl_wer, 2),
        "ARCD Passages WER (%)": round(p_wer, 2),
        "Throughput (xRT)": round(th, 2)
    })
    print(f"--> {m_name}: XL-Sum WER = {xl_wer:.2f}% | ARCD WER = {p_wer:.2f}% | Throughput = {th:.2f}x")
    
    del whisper_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

with open(asr_out_file, "w", encoding="utf-8") as f:
    json.dump(propagation_results, f, indent=2, ensure_ascii=False)
with open(arcd_trans_file, "w", encoding="utf-8") as f:
    json.dump(arcd_transcripts, f, indent=2, ensure_ascii=False)

print()
print(f"[OK] Outputs saved to: {asr_out_file} and {arcd_trans_file}")