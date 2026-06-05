from sentence_transformers.losses import BatchHardTripletLoss
from sentence_transformers.losses.BatchHardTripletLoss import TripletDistanceMetric

print(f"COSINE value: {TripletDistanceMetric.COSINE}")
print(f"Type: {type(TripletDistanceMetric.COSINE)}")

# Check if it's a callable
cosine_val = TripletDistanceMetric.COSINE
print(f"Callable? {callable(cosine_val)}")

# Try different instantiation
print()
print("Trying with distance_metric='cosine' (string)...")
try:
    from sentence_transformers import SentenceTransformer
    import torch
    
    # Create a minimal mock since we don't have a model here
    # Actually let's just check the signature
    import inspect
    sig = inspect.signature(BatchHardTripletLoss.__init__)
    print(f"Signature: {sig}")
    for name, param in sig.parameters.items():
        print(f"  {name}: default={param.default}")
except Exception as e:
    print(f"Error: {e}")

# Check what TripletDistanceMetric enum members are
print()
print("All members:")
for name in dir(TripletDistanceMetric):
    if not name.startswith('_'):
        val = getattr(TripletDistanceMetric, name)
        if not callable(val):
            print(f"  {name}: {val}")
