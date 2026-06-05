import os, glob

blobs = '/home/ztt/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/blobs'

# Remove incomplete blobs
removed = 0
for f in glob.glob(os.path.join(blobs, '*.incomplete')):
    os.remove(f)
    print(f'Removed: {os.path.basename(f)}')
    removed += 1

if removed == 0:
    print('No incomplete blobs to remove')

# List remaining
print('\nRemaining blobs:')
for f in os.listdir(blobs):
    sz = os.path.getsize(os.path.join(blobs, f))
    print(f'  {f}: {sz/1024/1024:.1f} MB')

print('\nDone')
