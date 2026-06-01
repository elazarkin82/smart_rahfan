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
    
    # Remove existing batch files in the output directory to prevent old batch leftovers
    old_batches = glob.glob(os.path.join(args.output_dir, "batch_*.pkl"))
    if old_batches:
        print(f"Cleaning {len(old_batches)} existing batch files in '{args.output_dir}'...")
        for old_b in old_batches:
            try:
                os.remove(old_b)
            except Exception:
                pass

    # 2. Chunk-based gathering and shuffling loop
    temporary_samples = []
    batch_idx = 0
    batch_size = args.batch_size
    
    print("Starting chunk-based gathering and shuffling loop...")
    pbar = tqdm.tqdm(total=total_samples, desc="Processing Samples")
    
    while True:
        # Get list of files that still have unused samples
        active_files = [f for f, indices in available_indices.items() if len(indices) > 0]
        if not active_files:
            break
            
        # Interleave files randomly in each pass
        random.shuffle(active_files)
        
        # Load one file at a time, randomly pick exactly 10 samples, and add to the temporary pool
        for fpath in active_files:
            indices = available_indices[fpath]
            k_to_pick = min(10, len(indices))
            
            # Randomly select indices without replacement
            chosen_indices = random.sample(indices, k_to_pick)
            
            # Remove from available pool
            for idx in chosen_indices:
                indices.remove(idx)
                
            # Read compiled file to fetch samples
            with open(fpath, 'rb') as f:
                samples = pickle.load(f)
                
            for idx in chosen_indices:
                temporary_samples.append(samples[idx])
                pbar.update(1)
                
            del samples
            gc.collect()
            
        # Globally shuffle the gathered pool
        random.shuffle(temporary_samples)
        
        # Slice the shuffled pool into complete batches of size batch_size
        num_batches_to_write = len(temporary_samples) // batch_size
        
        for _ in range(num_batches_to_write):
            batch_samples = temporary_samples[:batch_size]
            temporary_samples = temporary_samples[batch_size:]
            
            batch_path = os.path.join(args.output_dir, f"batch_{batch_idx:04d}.pkl")
            with open(batch_path, 'wb') as f:
                pickle.dump(batch_samples, f)
            batch_idx += 1
            
            del batch_samples
            gc.collect()

    # 3. Save any final remainder at the very end as is
    if temporary_samples:
        batch_path = os.path.join(args.output_dir, f"batch_{batch_idx:04d}.pkl")
        with open(batch_path, 'wb') as f:
            pickle.dump(temporary_samples, f)
        batch_idx += 1
        
        del temporary_samples
        gc.collect()

    pbar.close()
    print(f"Done! {batch_idx} batched files successfully created in '{args.output_dir}'.")

if __name__ == "__main__":
    main()
