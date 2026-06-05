import os, sys, time
from pathlib import Path

# Local snapshot path
SNAP = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-base-en-v1.5" / "snapshots"
snap_dirs = list(SNAP.glob("*")) if SNAP.exists() else []
print(f"Snapshot dirs: {[d.name for d in snap_dirs]}")

if snap_dirs:
    snap_path = str(snap_dirs[0])
    print(f"Loading from: {snap_path}")
    sys.stdout.flush()
    
    start = time.time()
    # Load via SentenceTransformer from local path
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(snap_path)
    elapsed = time.time() - start
    print(f"Loaded in {elapsed:.1f}s")
    print(f"Max seq length: {model.max_seq_length}")
    print(f"Embedding dim: {model.get_sentence_embedding_dimension()}")
    
    # Test encode
    emb = model.encode("Hello world, this is a test sentence.")
    print(f"Test embedding shape: {emb.shape}")
    print(f"First 5 values: {emb[:5].tolist()}")
    print("SUCCESS - model works!")
else:
    print("No snapshot directory found!")
    sys.exit(1)
