# ── CELL 1: ENVIRONMENT & EVALKIT-SRC VERIFICATION ────────────────────
import os, sys, subprocess, json, hashlib, zipfile, shutil

for pkg in ["ctranslate2", "faster-whisper", "edge-tts", "soundfile"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
            print(f"[OK] Installed {pkg}.")
        except Exception as e:
            print(f"[ERROR] Could not install {pkg}: {e}")
            print("Please toggle Internet: ON in Kaggle Notebook Settings (right sidebar) to allow pip install.")
            raise

RUN_MODE = os.environ.get("RUN_MODE", "full")
OUT_DIR = "Results/smoke" if RUN_MODE == "smoke" else "Results/final"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("Results/env", exist_ok=True)
os.makedirs("audio_tts/xlsum", exist_ok=True)
os.makedirs("audio_tts/arcd_passages", exist_ok=True)
os.makedirs("audio_tts/arcd_queries", exist_ok=True)

EXPECTED_EVALKIT_SHA = "555c1e5703a62487e76d6c4ef3367e2c97d318e71d6d2d51ebd686e73281e110"
print(f"Internet: ON | RUN_MODE: {RUN_MODE} | Target: {OUT_DIR} | Expected evalkit SHA: {EXPECTED_EVALKIT_SHA}")

evalkit_ready = False
if os.path.exists("/kaggle/working/evalkit") and os.path.exists("/kaggle/working/configs"):
    evalkit_ready = True
elif os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in sorted(files):
            if "evalkit" in f.lower() and f.endswith(".zip"):
                zip_path = os.path.join(root, f)
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall("/kaggle/working")
                evalkit_ready = True
                break
        if evalkit_ready:
            break
        if "evalkit" in dirs and "configs" in dirs:
            if root != "/kaggle/working":
                if not os.path.exists("/kaggle/working/evalkit"):
                    shutil.copytree(os.path.join(root, "evalkit"), "/kaggle/working/evalkit")
                if not os.path.exists("/kaggle/working/configs"):
                    shutil.copytree(os.path.join(root, "configs"), "/kaggle/working/configs")
            evalkit_ready = True
            break

if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

assert evalkit_ready or RUN_MODE == "smoke", "FATAL: evalkit-src not found in /kaggle/input!"
print("[OK] evalkit unpacked and loaded to sys.path.")

env_path = 'Results/env/final_02_tts_asr_env.txt'
out_lines = ["Python Version: " + str(sys.version)]
try:
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True)
    out_lines.append("GPU Info:" + chr(10) + str(gpu_info).strip())
except Exception as e:
    out_lines.append("GPU Info: None (" + str(e) + ")")

out_lines.append("Package Versions:")
for p in ['torch', 'transformers', 'ctranslate2', 'faster_whisper', 'edge_tts', 'soundfile']:
    try:
        mod = __import__(p.replace("-", "_"))
        out_lines.append(f"  {p}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as e:
        out_lines.append(f"  {p}: NOT INSTALLED ({e})")

with open(env_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(out_lines) + "\n")
print(f"[OK] Environment logged to: {env_path}")