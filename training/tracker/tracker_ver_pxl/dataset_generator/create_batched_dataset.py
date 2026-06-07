import os
import glob
import argparse
import pickle
import random
import gc
import tqdm

def main():
    parser = argparse.ArgumentParser(description="Memory-efficiently shuffle compiled samples using a chunk-gathering sliding window.")
    parser.add_argument("--compiled_dir", default="compiled", help="Directory with compiled flight PKL files.")
    parser.add_argument("--output_dir", default="dataset", help="Target directory to save batched PKL files.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of samples per batched PKL file.")
    args = parser.parse_args()

    if not os.path.exists(args.compiled_dir):
        print(f"Error: Compiled directory '{args.compiled_dir}' does not exist.")
        return

    # Find all compile PKL files
    compiled_files = sorted(glob.glob(os.path.join(args.compiled_dir, "train_*.pkl")))
    if not compiled_files:
        print(f"No compiled files (train_*.pkl) found in '{args.compiled_dir}'. Run dataset_compiler.py first.")
        return

    # 1. Scanning and indexing compiled files to map available indices
    print(f"Scanning and indexing {len(compiled_files)} compiled files...")
    available_indices = {}
    total_samples = 0
    
    for fpath in tqdm.tqdm(compiled_files, desc="Indexing"):
        with open(fpath, 'rb') as f:
            try:
                samples = pickle.load(f)
                available_indices[fpath] = list(range(len(samples)))
                total_samples += len(samples)
                del samples
                gc.collect()
            except Exception as e:
                print(f"\nWarning: Failed to scan {fpath}: {e}")

    print(f"Successfully indexed {total_samples} samples.")

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Remove existing batch files in the output directory
    old_batches = glob.glob(os.path.join(args.output_dir, "batch_*.pkl"))
    if old_batches:
        print(f"Cleaning {len(old_batches)} existing batch files in '{args.output_dir}'...")
        for old_b in old_batches:
            try:
                os.remove(old_b)
            except Exception:
                pass

    # 2. Gather all samples into memory and separate by positive status
    pos_pool = []
    neg_pool = []
    
    print("Gathering and separating samples into positive and negative pools...")
    for fpath in tqdm.tqdm(compiled_files, desc="Loading Pool"):
        with open(fpath, 'rb') as f:
            try:
                samples = pickle.load(f)
                for s in samples:
                    is_pos = s.get('metadata', {}).get('is_positive', 1)
                    if is_pos != 0:
                        pos_pool.append(s)
                    else:
                        neg_pool.append(s)
                del samples
                gc.collect()
            except Exception as e:
                print(f"\nWarning: Failed to load {fpath}: {e}")

    print(f"Pool Stats - Positives: {len(pos_pool)} | Negatives: {len(neg_pool)}")

    # Shuffle both pools globally
    random.shuffle(pos_pool)
    random.shuffle(neg_pool)

    batch_size = args.batch_size
    total_samples = len(pos_pool) + len(neg_pool)
    
    if total_samples == 0:
        print("Error: No samples found.")
        return

    # Calculate optimal positive/negative split per batch based on global ratio
    num_pos_per_batch = int(round(batch_size * (len(pos_pool) / total_samples)))
    num_pos_per_batch = max(0, min(batch_size, num_pos_per_batch))
    num_neg_per_batch = batch_size - num_pos_per_batch

    print(f"Batch configuration: {num_pos_per_batch} positives and {num_neg_per_batch} negatives per batch (size {batch_size}).")

    # Determine maximum number of full batches we can make
    num_batches = 0
    if num_pos_per_batch > 0 and num_neg_per_batch > 0:
        num_batches = min(len(pos_pool) // num_pos_per_batch, len(neg_pool) // num_neg_per_batch)
    elif num_pos_per_batch > 0:
        num_batches = len(pos_pool) // batch_size
    elif num_neg_per_batch > 0:
        num_batches = len(neg_pool) // batch_size

    print(f"Generating {num_batches} homogeneous batches...")
    
    for i in tqdm.tqdm(range(num_batches), desc="Creating Batches"):
        batch_samples = []
        if num_pos_per_batch > 0:
            batch_samples.extend(pos_pool[i * num_pos_per_batch : (i + 1) * num_pos_per_batch])
        if num_neg_per_batch > 0:
            batch_samples.extend(neg_pool[i * num_neg_per_batch : (i + 1) * num_neg_per_batch])
        
        # Shuffle internally to mix positive/negative positions within the batch
        random.shuffle(batch_samples)
        
        batch_path = os.path.join(args.output_dir, f"batch_{i:04d}.pkl")
        with open(batch_path, 'wb') as f:
            pickle.dump(batch_samples, f)
            
    print(f"Done! {num_batches} homogeneous batched files (batch_*.pkl) successfully created.")

if __name__ == "__main__":
    main()
