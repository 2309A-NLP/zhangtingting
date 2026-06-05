#!/usr/bin/env python3
"""Fix huggingface hub cache: compute blob hashes and create proper symlinks."""
import os, sys, hashlib, shutil
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-base-en-v1.5"
BLOBS_DIR = MODEL_DIR / "blobs"
SNAPS_DIR = MODEL_DIR / "snapshots"
REFS_DIR = MODEL_DIR / "refs"

print(f"Model cache: {MODEL_DIR}")
print(f"  blobs/: {BLOBS_DIR} (exists: {BLOBS_DIR.exists()})")
print(f"  snapshots/: {SNAPS_DIR} (exists: {SNAPS_DIR.exists()})")

if not SNAPS_DIR.exists():
    print("ERROR: No snapshots directory")
    sys.exit(1)

snaps = sorted(SNAPS_DIR.iterdir())
if not snaps:
    print("ERROR: No snapshots")
    sys.exit(1)

snap = snaps[-1]
print(f"\nTarget snapshot: {snap.name}")
print()

# Clean stale incomplete blobs first
for f in BLOBS_DIR.glob("*.incomplete"):
    print(f"  Cleaning stale: {f.name}")
    f.unlink()

fixed = 0
for f in sorted(snap.iterdir()):
    if f.is_symlink() or f.is_dir():
        continue
    
    sz = f.stat().st_size
    print(f"  {f.name} ({sz/1024/1024:.1f} MB)" if sz > 1_000_000 else f"  {f.name} ({sz} bytes)")
    sys.stdout.flush()
    
    # Compute sha256
    sha = hashlib.sha256()
    with open(f, "rb") as fp:
        if sz > 10_000_000:
            for chunk in iter(lambda: fp.read(8*1024*1024), b""):
                sha.update(chunk)
        else:
            sha.update(fp.read())
    blob_hash = sha.hexdigest()
    print(f"    hash: {blob_hash[:32]}...")
    
    # Copy to blobs/ if not already there
    blob_path = BLOBS_DIR / blob_hash
    if not blob_path.exists():
        print(f"    copying to blobs/...")
        shutil.copy2(f, blob_path)
    
    # Replace with symlink: relative from snap/ -> ../../blobs/<hash>
    rel = os.path.relpath(blob_path, start=snap)
    f.unlink()
    os.symlink(rel, f)
    print(f"    symlink: {f.name} -> {rel}")
    fixed += 1

print(f"\nFixed {fixed} files")

# Ensure refs/main is correct
ref_main = REFS_DIR / "main"
if not ref_main.exists():
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    ref_main.write_text(snap.name + "\n")
    print(f"Wrote refs/main -> {snap.name}")

print(f"\nBlobs: {len(list(BLOBS_DIR.iterdir()))} files")
all_symlinks = all(f.is_symlink() for f in snap.iterdir() if not f.is_dir())
print(f"All snapshot files are symlinks: {all_symlinks}")

# Test
print("\n=== Testing load by name (offline) ===")
import sentence_transformers
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
model = sentence_transformers.SentenceTransformer("BAAI/bge-base-en-v1.5")
print(f"Loaded! Dim: {model.get_embedding_dimension()}, Max seq: {model.max_seq_length}")
print("OK - cache fixed!")
