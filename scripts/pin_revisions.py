import os
import json

def pin_revisions():
    protocol_path = "configs/final_protocol.json"
    with open(protocol_path, "r", encoding="utf-8") as f:
        protocol = json.load(f)

    # 1. Models & Revisions
    protocol["models"] = {
        "summarizer": {
            "id": "Omar10lfc/arabart-xlsum-arabic",
            "revision": "c43dad77a8f6ac6319536f296bd0c260118a9efd"
        },
        "dense_bi_encoder": {
            "id": "CAMeL-Lab/bert-base-arabic-camelbert-msa",
            "revision": "277069fd3645fedb22b746caf38d111aadee0241",
            "pooling": "mean",
            "max_length": 128
        },
        "cross_encoder_reranker": {
            "id": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "revision": "1427fd652930e4ba29e8149678df786c240d8825",
            "max_length": 512
        },
        "whisper_small_arabic": {
            "id": "Omar10lfc/whisper-small-arabic",
            "revision": "85dd7085dde37003058f530a6bfd072c4e3e0add"
        },
        "whisper_large_v3_base": {
            "id": "openai/whisper-large-v3",
            "revision": "06f233fe06e710322aca913c1bc4249a0d71fce1"
        },
        "whisper_large_v3_turbo_base": {
            "id": "openai/whisper-large-v3-turbo",
            "revision": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
        },
        "whisper_medium_base": {
            "id": "openai/whisper-medium",
            "revision": "abdf7c39ab9d0397620ccaea8974cc764cd0953e"
        },
        "whisper_small_base": {
            "id": "openai/whisper-small",
            "revision": "973afd24965f72e36ca33b3055d56a652f456b4d"
        }
    }

    # 2. Normalizers
    protocol["normalizer"] = {
        "primary": "legacy",
        "secondary": "v2"
    }

    # 3. ASR Decoding Configurations
    protocol["asr"]["primary_greedy"] = {
        "language": "ar",
        "task": "transcribe",
        "beam_size": 1,
        "best_of": 1,
        "temperature": 0.0,
        "vad_filter": False,
        "condition_on_previous_text": False
    }
    protocol["asr"]["ct2_beam5_vad"] = {
        "language": "ar",
        "task": "transcribe",
        "beam_size": 5,
        "best_of": 5,
        "temperature": 0.0,
        "vad_filter": True,
        "condition_on_previous_text": False
    }

    # 4. Summarization
    protocol["summarization"]["batch_size"] = 8
    protocol["summarization"]["model_id"] = protocol["models"]["summarizer"]["id"]
    protocol["summarization"]["model_revision"] = protocol["models"]["summarizer"]["revision"]

    # 5. Retrieval
    protocol["retrieval"]["chunk_words"] = 50
    protocol["retrieval"]["chunk_overlap"] = 0
    protocol["retrieval"]["normalize"] = False
    protocol["retrieval"]["bi_encoder_id"] = protocol["models"]["dense_bi_encoder"]["id"]
    protocol["retrieval"]["bi_encoder_revision"] = protocol["models"]["dense_bi_encoder"]["revision"]
    protocol["retrieval"]["cross_encoder_id"] = protocol["models"]["cross_encoder_reranker"]["id"]
    protocol["retrieval"]["cross_encoder_revision"] = protocol["models"]["cross_encoder_reranker"]["revision"]

    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    print(f"[OK] Revisions and protocol configuration updated in {protocol_path}")

if __name__ == "__main__":
    pin_revisions()
