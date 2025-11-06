from gliclass.audio_data_processing import GCSManager
import json
from google.cloud import storage
from tqdm import tqdm
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoFeatureExtractor
from gliclass.audio_data_processing import GLiClassAudioDataset
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from dotenv import load_dotenv
import boto3
import json
import numpy as np
import random
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer, AutoConfig, AutoFeatureExtractor,
)
from transformers.models.clap.configuration_clap import ClapConfig
from gliclass import GLiClassModelConfig, GLiClassModel
from gliclass.training import TrainingArguments, Trainer
from gliclass.data_processing import DataCollatorWithPadding
from gliclass.audio_data_processing import GLiClassAudioDataset, S3Manager


class Args:
    model_name = None
    encoder_model_name = "microsoft/deberta-v3-base"
    audio_model_name = "facebook/hubert-large-ls960-ft"
    save_path = "./models/gliclass-hu-audio-base"
    data_path = "/home/aleksandrlukasov/multi_gpu_gliclass_audio/sampled.jsonl"
    problem_type = "multi_label_classification"
    pooler_type = "first"
    scorer_type = "audio-token-dot"
    architecture_type = "audio-encoder"
    normalize_features = True
    extract_text_features = False
    prompt_first = True
    use_lstm = False
    squeeze_layers = False
    shuffle_labels = True
    num_epochs = 1
    batch_size = 1
    gradient_accumulation_steps = 1
    encoder_lr = 1e-5
    audio_lr = 1e-5
    others_lr = 1e-5
    encoder_weight_decay = 0.015
    audio_weight_decay = 0.015
    others_weight_decay = 0.015
    warmup_ratio = 0.008
    lr_scheduler_type = "cosine"
    focal_loss_alpha = 0.6
    focal_loss_gamma = 2
    contrastive_loss_coef = 0.
    max_length = 2048
    sampling_rate = 16000
    max_duration_s = 15
    save_steps = 5000
    save_total_limit = 15
    num_workers = 1
    fp16 = False
    bf16 = True

args = Args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)

if not os.path.exists(args.data_path):
    raise FileNotFoundError(f"Dataset file not found: {args.data_path}")
print(f"✓ Dataset file exists: {args.data_path}")

client = storage.Client()
gcs_manager = GCSManager(
    gcs_client=client,
    local_cache_dir="./cache",
    preload_size=5,
    max_load_workers=10,
    remaining_preloaded_threshold=1,
    max_cache_size_mb=100,
)

tokenizer = AutoTokenizer.from_pretrained(
    args.encoder_model_name,
    use_fast=False,
    trust_remote_code=True
)

encoder_config = AutoConfig.from_pretrained(args.encoder_model_name)
audio_config = AutoConfig.from_pretrained(args.audio_model_name)
audio_feature_extractor = AutoFeatureExtractor.from_pretrained(args.audio_model_name)

glicalss_config = GLiClassModelConfig(
    encoder_config=encoder_config,
    encoder_model=args.encoder_model_name,
    audio_model_name=args.audio_model_name,
    audio_model_config=audio_config,
    class_token_index=len(tokenizer),
    text_token_index=len(tokenizer) + 1,
    audio_token_index=len(tokenizer) + 2,
    pooling_strategy=args.pooler_type,
    scorer_type=args.scorer_type,
    use_lstm=args.use_lstm,
    focal_loss_alpha=args.focal_loss_alpha,
    focal_loss_gamma=args.focal_loss_gamma,
    contrastive_loss_coef=args.contrastive_loss_coef,
    normalize_features=args.normalize_features,
    extract_text_features=args.extract_text_features,
    architecture_type=args.architecture_type,
    prompt_first=args.prompt_first,
    squeeze_layers=args.squeeze_layers,
    shuffle_labels=args.shuffle_labels,
)

model = GLiClassModel(glicalss_config, from_pretrained=True)
model.to(device)
model.config.problem_type = args.problem_type

new_words = ["<<LABEL>>", "<<SEP>>", "<<AUDIO>>"]
tokenizer.add_tokens(new_words, special_tokens=True)
model.resize_token_embeddings(len(tokenizer))

print("Creating dataset...")
train_dataset = GLiClassAudioDataset(
    dataset_path=args.data_path,
    cloud_manager=gcs_manager,
    tokenizer=tokenizer,
    audio_features_extractor=audio_feature_extractor,
    max_length=args.max_length,
    problem_type=args.problem_type,
    architecture_type=args.architecture_type,
    sampling_rate=args.sampling_rate,
    max_duration_s=args.max_duration_s,
    validate_json_file=True,
    buffer_size=8192,
)

data_collator = DataCollatorWithPadding(device=device)


def compute_metrics(p):
    predictions, labels = p
    labels = labels.reshape(-1)
    
    if args.problem_type == "single_label_classification":
        preds = np.argmax(predictions, axis=1)
    else:
        predictions = predictions.reshape(-1)
        preds = (predictions > 0.5).astype(int)
        labels = np.where(labels > 0.5, 1, 0)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


world_size = int(os.environ.get("WORLD_SIZE", 1))
total_examples = train_dataset.get_num_examples()

print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)
print(f"Dataset path: {args.data_path}")
print(f"Total examples: {total_examples}")
print(f"World size (GPUs): {world_size}")
print(f"Batch size per device: {args.batch_size}")
print(f"Gradient accumulation: {args.gradient_accumulation_steps}")

if total_examples == 0:
    raise ValueError("Dataset returned 0 examples! Check your JSONL file format.")

effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
steps_per_epoch = total_examples // max(effective_batch_size, 1)

if steps_per_epoch == 0:
    print(f"WARNING: steps_per_epoch is 0. Setting to 1.")
    steps_per_epoch = 1

max_steps = steps_per_epoch * args.num_epochs

print(f"Effective batch size: {effective_batch_size}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Num epochs: {args.num_epochs}")
print(f"Max steps: {max_steps}")
print("="*60 + "\n")

if max_steps == 0:
    raise ValueError("max_steps is 0! Cannot train.")

training_args = TrainingArguments(
    dataloader_pin_memory=False,
    output_dir=args.save_path,
    learning_rate=args.encoder_lr,
    weight_decay=args.encoder_weight_decay,
    audio_lr=args.audio_lr,
    audio_weight_decay=args.audio_weight_decay,
    others_lr=args.others_lr,
    others_weight_decay=args.others_weight_decay,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_ratio=args.warmup_ratio,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    max_steps=max_steps,
    num_train_epochs=args.num_epochs,
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    dataloader_num_workers=args.num_workers,
    logging_steps=1,
    use_cpu=not torch.cuda.is_available(),
    report_to="none",
    fp16=args.fp16,
    bf16=args.bf16,
    accelerator_config={
        "dispatch_batches": False,
        "split_batches": False,
        "even_batches": True,
    },
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()