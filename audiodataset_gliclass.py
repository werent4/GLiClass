import os
if os.path.exists("/mnt/werent4-storage"):
    os.environ['HF_HOME'] = '/mnt/werent4-storage/huggingface_cache'
    os.environ['TRANSFORMERS_CACHE'] = '/mnt/werent4-storage/huggingface_cache'
    os.environ['HF_DATASETS_CACHE'] = '/mnt/werent4-storage/huggingface_cache'
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from datasets import load_dataset, Audio
import json
import random
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import numpy as np
from tqdm import tqdm
import hashlib
import pickle

audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

def extract_and_save_features(dataset, audio_dir):   
    audio_paths = []
    srs = []
    for example in tqdm(dataset, desc= "Saving audio arrays"):
        idx = example["index"]

        audio_array = example['audio']['array']
        sampling_rate = example['audio']['sampling_rate']
        srs.append(sampling_rate)

        if len(audio_array) == 0:
            print(f"Warning: Empty audio array found for index {idx}. Creating a zero array.")
            audio_array = np.zeros(16000) 

        if isinstance(audio_array, np.ndarray):
            audio_array = torch.from_numpy(audio_array)
        elif isinstance(audio_array, list):
            audio_array = torch.tensor(audio_array)
        elif isinstance(audio_array, torch.Tensor):
            pass  
        else:
            print(f"Warning: Unknown audio_array type {type(audio_array)} for index {idx}")
            audio_array = torch.tensor(audio_array)


        if isinstance(audio_array, torch.Tensor):
            audio_hash = hashlib.md5(audio_array.numpy().tobytes()).hexdigest()
        else:
            audio_hash = hashlib.md5(audio_array.tobytes()).hexdigest()

        audio_filename = f"{audio_hash}-{sampling_rate}.pt"
        audio_path = os.path.join(audio_dir, audio_filename)
        
        torch.save(audio_array, audio_path)
        audio_paths.append(audio_path)
    
    dataset = dataset.add_column("audio_path", audio_paths)
    dataset = dataset.add_column("sampling_rate", srs)
    return dataset

def get_label_mapping_from_dataset(dataset, audio_column = "audio"):   
    try:
        if audio_column in dataset.features and hasattr(dataset.features['label'], 'names'):
            label_names = dataset.features['label'].names
            label_mapping = {i: name for i, name in enumerate(label_names)}
            print("Labels mapping:")
            print(label_mapping)
            return label_mapping
        else:
            print("No label mapping found in the dataset features.")
            if 'label' in dataset.features:
                unique_labels = set(dataset['label'])
                label_mapping = {i: str(i) for i in unique_labels}
                print("Generated label mapping from data:")
                print(label_mapping)
                return label_mapping
    except AttributeError:
        print("Failed to access dataset features. Ensure the dataset is loaded correctly.")
    
    return {0: "unknown"}

def add_index(example, idx):
    example["index"] = idx
    return example

def main(args):
    audio_dir = os.path.join(os.path.dirname(args.save_path), "audio_features")
    os.makedirs(audio_dir, exist_ok=True)
    print(f"Audio features will be saved to: {audio_dir}")
    
    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.map(add_index, with_indices=True)
    dataset = dataset.shuffle(seed=42)
    dataset = extract_and_save_features(dataset, audio_dir)
    
    id2label = get_label_mapping_from_dataset(dataset)
    
    metadata = {
        "dataset_name": args.dataset_name,
        "processing_date": str(np.datetime64('now')),
        "num_samples": len(dataset),
        "label_mapping": id2label,
        "audio_dir": audio_dir,
        "dataset_file_name": os.path.basename(args.save_path).split('/')[-1],
    }
    
    metadata_path = os.path.join(os.path.dirname(args.save_path), "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    dataset_list = []
    for example in tqdm(dataset, desc="Processing audio files", total=len(dataset)):
        idx = example["index"]   
        audio_path = example["audio_path"]
        sample_rate = example["sampling_rate"]
        label = id2label[example['label']] if example['label'] in id2label else "unknown"
        
        all_labels = list(id2label.values())
        random.shuffle(all_labels) # Dont forget to shuffle the labels, to help the model to better generalize
        row = {
            "id": idx,
            "audio_path": audio_path,
            "sample_rate" : sample_rate,
            "all_labels": all_labels,
            "true_labels": [label],
        }
        
        dataset_list.append(row)                

    with open(args.save_path, "w") as f:
        json.dump(dataset_list, f, indent=4)

    print(f"Processed dataset saved to {args.save_path}")
    print(f"Audio features saved to {audio_dir}")
    print(f"Metadata saved to {metadata_path}")
    
    dataset_size = os.path.getsize(args.save_path) / (1024 * 1024)
    print(f"Dataset JSON file size: {dataset_size:.2f} MB")
    
    audio_dir_size = sum(os.path.getsize(os.path.join(audio_dir, f)) for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f)))
    audio_dir_size = audio_dir_size / (1024 * 1024)
    print(f"Audio features directory size: {audio_dir_size:.2f} MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="Hemg/Emotion-audio-Dataset")
    parser.add_argument('--audio_column', type=str, default="audio")
    parser.add_argument('--save_path', type=str, default="./datasets/processed_dataset.json")
    parser.add_argument('--clean_temp', action='store_true', default=True, help="Clean temporary files after processing.")
    args = parser.parse_args()
    main(args)