import argparse
import hashlib
import json
import os
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from datasets import Audio, load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"

audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)


def extract_and_save_features(example, audio_dir):
    audio_array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]

    if len(audio_array) == 0:
        print("Warning: Empty audio array found. Creating a zero array.")
        audio_array = np.zeros(16000)

    max_length = 16000 * 5
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    elif len(audio_array) < max_length:
        audio_array = np.pad(
            audio_array, (0, max_length - len(audio_array)), mode="constant"
        )

    inputs = audio_feature_extractor(
        audio_array, sampling_rate=sampling_rate, return_tensors="pt"
    )["input_values"].squeeze(0)

    audio_hash = hashlib.md5(audio_array.tobytes()).hexdigest()
    audio_filename = f"{audio_hash}.pt"
    audio_path = os.path.join(audio_dir, audio_filename)

    torch.save(inputs, audio_path)

    return audio_path


def get_label_mapping_from_dataset(dataset, audio_column="audio"):
    try:
        if audio_column in dataset.features and hasattr(
            dataset.features["label"], "names"
        ):
            label_names = dataset.features["label"].names
            label_mapping = dict(enumerate(label_names))
            print("Labels mapping:")
            print(label_mapping)
            return label_mapping
        else:
            print("No label mapping found in the dataset features.")
            if "label" in dataset.features:
                unique_labels = set(dataset["label"])
                label_mapping = {i: str(i) for i in unique_labels}
                print("Generated label mapping from data:")
                print(label_mapping)
                return label_mapping
    except AttributeError:
        print(
            "Failed to access dataset features. Ensure the dataset is loaded correctly."
        )

    return {0: "unknown"}


def main(args):
    audio_dir = os.path.join(os.path.dirname(args.save_path), "audio_features")
    os.makedirs(audio_dir, exist_ok=True)
    print(f"Audio features will be saved to: {audio_dir}")

    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print("Dataset loaded...")
    print(dataset)
    print(dataset[0])

    dataset = dataset.shuffle(seed=42)

    id2label = get_label_mapping_from_dataset(dataset, args.audio_column)

    metadata = {
        "dataset_name": args.dataset_name,
        "processing_date": str(np.datetime64("now")),
        "num_samples": len(dataset),
        "label_mapping": id2label,
        "audio_dir": audio_dir,
        "dataset_file": os.path.basename(args.save_path).split("/")[-1],
    }

    metadata_path = os.path.join(os.path.dirname(args.save_path), "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    dataset_list = []
    classes_ = {}
    for idx, example in enumerate(
        tqdm(dataset, desc="Processing audio files", total=len(dataset))
    ):
        try:
            audio_path = extract_and_save_features(example, os.path.abspath(audio_dir))

            label = (
                id2label[example["label"]]
                if example["label"] in id2label
                else "unknown"
            )
            if label not in classes_:
                classes_[label] = 0
            if classes_[label] >= 15:
                print(f"Skipping label '{label}' as it has reached the limit of 15 MB.")
                continue
            classes_[label] += 1
            all_labels = list(id2label.values())
            random.shuffle(
                all_labels
            )  # Dont forget to shuffle the labels, to help the model to better generalize
            row = {
                "id": idx,
                "audio_features_path": audio_path,
                "all_labels": all_labels,
                "true_labels": [label],
            }

            dataset_list.append(row)

            if idx > 0 and idx % 5000 == 0:
                temp_save_path = f"{args.save_path}.temp_{idx}"
                with open(temp_save_path, "w") as f:
                    json.dump(dataset_list, f)
                print(f"Saved temporary dataset with {idx} samples to {temp_save_path}")

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue

    with open(args.save_path, "w") as f:
        json.dump(dataset_list, f, indent=4)

    print(f"Processed dataset saved to {args.save_path}")
    print(f"Audio features saved to {audio_dir}")
    print(f"Metadata saved to {metadata_path}")

    dataset_size = os.path.getsize(args.save_path) / (1024 * 1024)
    print(f"Dataset JSON file size: {dataset_size:.2f} MB")

    audio_dir_size = sum(
        os.path.getsize(os.path.join(audio_dir, f))
        for f in os.listdir(audio_dir)
        if os.path.isfile(os.path.join(audio_dir, f))
    )
    audio_dir_size = audio_dir_size / (1024 * 1024)
    print(f"Audio features directory size: {audio_dir_size:.2f} MB")

    if args.clean_temp:
        print("Cleaning temporary files...")
        import re

        base_filename = os.path.basename(args.save_path)
        temp_pattern = re.compile(re.escape(base_filename) + r"\.temp_\d+$")
        dir_path = os.path.dirname(args.save_path)

        temp_files = []
        for f in os.listdir(dir_path):
            if temp_pattern.match(f):
                temp_files.append(f)

        for temp_file in temp_files:
            os.remove(os.path.join(dir_path, temp_file))
        print("Successfully cleaned temporary files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="Hemg/Emotion-audio-Dataset"
    )
    parser.add_argument("--audio_column", type=str, default="audio")
    parser.add_argument(
        "--save_path", type=str, default="./datasets1/processed_dataset.json"
    )
    parser.add_argument(
        "--clean_temp",
        action="store_true",
        default=True,
        help="Clean temporary files after processing.",
    )
    args = parser.parse_args()
    main(args)
