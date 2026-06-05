"""Download BAAI/bge-base-en-v1.5 on Windows via proxy, then copy to WSL cache."""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Proxy settings
PROXY = "127.0.0.1:11304"
os.environ["HTTP_PROXY"] = f"http://{PROXY}"
os.environ["HTTPS_PROXY"] = f"http://{PROXY}"
os.environ["http_proxy"] = f"http://{PROXY}"
os.environ["https_proxy"] = f"http://{PROXY}"
# Use mirror for China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Target
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Windows cache dir
WIN_CACHE = Path(os.environ["USERPROFILE"]) / ".cache" / "huggingface" / "hub"
# WSL cache path mapping
WSL_CACHE = Path(r"\\wsl.localhost\Ubuntu-22.04\home\ztt\.cache\huggingface\hub")

def check_proxy():
    """Verify proxy is working."""
    import urllib.request
    try:
        proxy_handler = urllib.request.ProxyHandler({
            "http": f"http://{PROXY}",
            "https": f"http://{PROXY}"
        })
        opener = urllib.request.build_opener(proxy_handler)
        resp = opener.open("https://hf-mirror.com/api/models/BAAI/bge-base-en-v1.5", timeout=15)
        data = resp.read().decode()
        import json
        j = json.loads(data)
        print(f"Proxy OK. Model: {j.get('modelId', 'unknown')}")
        return True
    except Exception as e:
        print(f"Proxy check FAILED: {e}")
        # Try direct without proxy
        try:
            resp = urllib.request.urlopen("https://hf-mirror.com/api/models/BAAI/bge-base-en-v1.5", timeout=15)
            data = resp.read().decode()
            import json
            j = json.loads(data)
            print(f"No proxy needed. Model: {j.get('modelId', 'unknown')}")
            return True
        except Exception as e2:
            print(f"Direct also FAILED: {e2}")
            return False

def download_via_transformers():
    """Download using transformers library."""
    try:
        from transformers import AutoTokenizer, AutoModel
        print(f"Downloading {MODEL_NAME} via transformers...")
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Model downloaded! Config: {model.config}")
        return True
    except Exception as e:
        print(f"Transformers download failed: {e}")
        return False

def download_via_hub():
    """Download using huggingface_hub."""
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading {MODEL_NAME} via huggingface_hub...")
        path = snapshot_download(
            repo_id=MODEL_NAME,
            ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
            local_files_only=False,
        )
        print(f"Downloaded to: {path}")
        return True
    except Exception as e:
        print(f"HuggingFace hub download failed: {e}")
        return False

def download_via_pip():
    """Try using huggingface-cli."""
    try:
        result = subprocess.run(
            ["huggingface-cli", "download", MODEL_NAME],
            capture_output=True, text=True, timeout=300,
            env={**os.environ}
        )
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"huggingface-cli failed: {e}")
        return False

def check_cache():
    """Check current cache state."""
    cache_dir = WIN_CACHE / f"models--BAAI--bge-base-en-v1.5"
    if cache_dir.exists():
        blobs_dir = cache_dir / "blobs"
        snapshots_dir = cache_dir / "snapshots"
        n_blobs = len(list(blobs_dir.glob("*"))) if blobs_dir.exists() else 0
        n_complete = len(list(blobs_dir.glob("[!.]*"))) if blobs_dir.exists() else 0
        n_incomplete = len(list(blobs_dir.glob("*.incomplete"))) if blobs_dir.exists() else 0
        n_snapshots = len(list(snapshots_dir.glob("*"))) if snapshots_dir.exists() else 0
        
        total_bytes = sum(f.stat().st_size for f in blobs_dir.glob("*") if f.is_file()) if blobs_dir.exists() else 0
        
        print(f"Cache: {cache_dir}")
        print(f"  Blobs: {n_blobs} total ({n_complete} complete, {n_incomplete} incomplete)")
        print(f"  Snapshots: {n_snapshots}")
        print(f"  Total size: {total_bytes / 1024 / 1024:.1f} MB")
        
        if snapshots_dir.exists():
            for snap in snapshots_dir.iterdir():
                snap_files = list(snap.iterdir()) if snap.is_dir() else []
                print(f"  Snapshot {snap.name}: {len(snap_files)} files")
                for f in snap_files:
                    print(f"    {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    else:
        print(f"No cache directory found at {cache_dir}")
        return False

def copy_to_wsl():
    """Copy completed cache to WSL."""
    cache_dir = WIN_CACHE / f"models--BAAI--bge-base-en-v1.5"
    wsl_target = WSL_CACHE / f"models--BAAI--bge-base-en-v1.5"
    
    # Check if WSL path is accessible
    if not WSL_CACHE.exists():
        print(f"WSL cache path not accessible: {WSL_CACHE}")
        print("Trying alternative: copy to a temp dir on Windows, then copy via WSL")
        return False
    
    print(f"Copying to WSL: {cache_dir} -> {wsl_target}")
    
    # Use robocopy for better reliability
    try:
        subprocess.run([
            "robocopy", str(cache_dir), str(wsl_target),
            "/E", "/NP", "/NJH", "/NJS"
        ], check=True, timeout=120)
        print("Copy complete!")
        return True
    except Exception as e:
        print(f"Robocopy failed: {e}")
        # Fallback to xcopy
        try:
            subprocess.run([
                "xcopy", str(cache_dir), str(wsl_target),
                "/E", "/I", "/Y"
            ], check=True, timeout=120)
            print("xcopy complete!")
            return True
        except Exception as e2:
            print(f"xcopy also failed: {e2}")
            return False

if __name__ == "__main__":
    print("=== Step 1: Check current cache ===")
    check_cache()
    
    print("\n=== Step 2: Check proxy ===")
    proxy_ok = check_proxy()
    
    if not proxy_ok:
        print("\nProxy not working. Trying direct download (may work if mirror is accessible)...")
    
    print("\n=== Step 3: Download model ===")
    # Try multiple methods
    success = False
    for method_name, method_fn in [
        ("transformers", download_via_transformers),
        ("huggingface_hub", download_via_hub),
        ("huggingface-cli", download_via_pip),
    ]:
        print(f"\n--- Trying {method_name} ---")
        if method_fn():
            success = True
            break
        print(f"{method_name} failed, trying next...")
    
    print(f"\n=== Result: {'SUCCESS' if success else 'FAILED'} ===")
    
    if success:
        print("\n=== Step 4: Check cache after download ===")
        cache_ok = check_cache()
        
        print("\n=== Step 5: Copy to WSL ===")
        copy_to_wsl()
    
    print("\n=== Final cache state ===")
    check_cache()