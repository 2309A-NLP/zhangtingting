import sentence_transformers
print(f"sentence-transformers: {sentence_transformers.__version__}")

# Check available losses
from sentence_transformers import losses
print("\nAvailable losses:")
for name in dir(losses):
    if 'Loss' in name:
        print(f"  {name}")

# Try the loss that failed
print("\nTesting BatchHardTripletLoss...")
try:
    from sentence_transformers.losses import BatchHardTripletLoss
    loss = BatchHardTripletLoss(model=None, distance_metric="cosine")
    print("  BatchHardTripletLoss OK")
except Exception as e:
    print(f"  Failed: {e}")
    # Try with different args
    try:
        from sentence_transformers.losses import BatchHardTripletLoss
        from sentence_transformers.losses.BatchHardTripletLoss import TripletDistanceMetric
        loss = BatchHardTripletLoss(model=None, distance_metric=TripletDistanceMetric.COSINE)
        print(f"  With TripletDistanceMetric: OK")
    except Exception as e2:
        print(f"  Also failed: {e2}")

# Test MNR loss
print("\nTesting MultipleNegativesRankingLoss...")
try:
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    loss = MultipleNegativesRankingLoss(model=None)
    print("  MNR loss OK")
except Exception as e:
    print(f"  Failed: {e}")

# Test CoSENT
print("\nTesting CoSENTLoss...")
try:
    from sentence_transformers.losses import CoSENTLoss
    loss = CoSENTLoss(model=None)
    print("  CoSENT loss OK")
except Exception as e:
    print(f"  Failed: {e}")
