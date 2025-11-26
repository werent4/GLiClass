from gliclass.audio_data_processing import GCSManager
import json
from google.cloud import storage
from tqdm import tqdm
import time
from pathlib import Path
from transformers import TrainerCallback
from transformers import AutoTokenizer, AutoFeatureExtractor
from gliclass.training import AnalysisTrainer
from gliclass.audio_data_processing import GLiClassAudioDataset
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from dotenv import load_dotenv
import boto3
import json
import numpy as np
import random
from huggingface_hub import HfApi
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer, AutoConfig, AutoFeatureExtractor,
)
from transformers.models.clap.configuration_clap import ClapConfig
from gliclass import GLiClassModelConfig, GLiClassModel
from gliclass.training import TrainingArguments, Trainer
from gliclass.data_processing import DataCollatorWithPadding
from gliclass.audio_data_processing import GLiClassAudioDataset, JSONLManager
from model_builder import ModelBuilder


class Args:
    model_name = None
    encoder_model_name = "microsoft/deberta-v3-base"
    audio_model_name = "facebook/hubert-large-ls960-ft"
    data_path = "/home/alexandrlukashov/gliclass-audio-experiments/GLiClass/data/gliclass-audio-1M-shuffled-shuffled-labels.jsonl"
    problem_type = "multi_label_classification"
    pooler_type = "avg"
    scorer_type = "audio-token-dot"
    architecture_type = "audio-encoder"
    pooling_strategy = 'avg'
    normalize_features = True
    extract_text_features = False
    prompt_first = True
    use_lstm = False
    squeeze_layers = False
    shuffle_labels = True
    num_epochs = 1
    batch_size = 4
    gradient_accumulation_steps = 4
    encoder_lr = 3e-5
    audio_lr = 5e-4
    others_lr = 5e-4
    encoder_weight_decay = 0.001
    audio_weight_decay = 0.001
    others_weight_decay = 0.001
    warmup_ratio = 0.01
    lr_scheduler_type = "cosine"
    focal_loss_alpha = 0.0
    focal_loss_gamma = 0
    audio_text_contrastive_coef: float = 2.0
    audio_text_temperature: float = 0.03
    contrastive_loss_coef = 0.
    max_length = 2048
    sampling_rate = 16000
    max_duration_s = 15
    save_steps = 1000
    save_total_limit = 5
    use_stable_adam = False
    num_workers = 1
    fp16 = False
    bf16 = True
    save_path = f"./models_exp/gliclass-hu-audio-base-fp16-{fp16}-bf16-{bf16}"


    preload_size = 2000  # 500 × 0.91 MB ≈ 1820 MB avg
    max_load_workers = 10
    remaining_preloaded_threshold = 400  # 400/5it/s = 80 sek buffer (max cache size should be preload size + threshold)
    total_cache_size_mb = 32768  # 32gb total; 4GB × 8 GPUs


class HuggingFacePushCallback(TrainerCallback):
    def __init__(self, args_to_save, username="alexandrlukashov"):
        self.username = username
        self.api = HfApi()
        self.args_to_save = args_to_save

    def on_save(self, args, state, control, **kwargs):
        checkpoint_num = state.global_step
        checkpoint_path = f"{args.output_dir}/checkpoint-{checkpoint_num}"
        if os.path.exists(checkpoint_path):
            args_dict = {
                key: getattr(self.args_to_save, key)
                for key in dir(self.args_to_save)
                if not key.startswith('_') and not callable(getattr(self.args_to_save, key))
            }
            with open(os.path.join(checkpoint_path, "training_args.json"), 'w') as f:
                json.dump(args_dict, f, indent=4, default=str)
        repo_id = f"{self.username}/gliclass-1M-experiments-{checkpoint_num}"
        try:
            self.api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            self.api.upload_folder(folder_path=checkpoint_path, repo_id=repo_id, repo_type="model")
            print('pushed')
        except Exception as e:
            print(e)
        return control



args = Args()
save_args_callback = HuggingFacePushCallback(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)

if not os.path.exists(args.data_path):
    raise FileNotFoundError(f"Dataset file not found: {args.data_path}")
print(f"Dataset file exists: {args.data_path}")

tokenizer = AutoTokenizer.from_pretrained(
    args.encoder_model_name,
    use_fast=False,
    trust_remote_code=True
)
audio_feature_extractor = AutoFeatureExtractor.from_pretrained(args.audio_model_name)

# glicalss_config = GLiClassModelConfig(
#     encoder_config=encoder_config,
#     encoder_model=args.encoder_model_name,
#     audio_model_name=args.audio_model_name,
#     audio_model_config=audio_config,
#     class_token_index=len(tokenizer),
#     text_token_index=len(tokenizer) + 1,
#     audio_token_index=len(tokenizer) + 2,
#     pooling_strategy=args.pooler_type,
#     scorer_type=args.scorer_type,
#     use_lstm=args.use_lstm,
#     focal_loss_alpha=args.focal_loss_alpha,
#     focal_loss_gamma=args.focal_loss_gamma,
#     contrastive_loss_coef=args.contrastive_loss_coef,
#     normalize_features=args.normalize_features,
#     extract_text_features=args.extract_text_features,
#     architecture_type=args.architecture_type,
#     prompt_first=args.prompt_first,
#     squeeze_layers=args.squeeze_layers,
#     shuffle_labels=args.shuffle_labels,
# )

# model = GLiClassModel(glicalss_config, from_pretrained=True)
model_builder = ModelBuilder(device, 'knowledgator/gliclass-base-v1.0-lw', tokenizer, args)
model = model_builder.swap_layers()
for param in model.model.encoder_model.parameters():
    param.requires_grad = False

# model.to(device)
# model.config.problem_type = args.problem_type

# new_words = ["<<LABEL>>", "<<SEP>>", "<<AUDIO>>"]
# tokenizer.add_tokens(new_words, special_tokens=True)
# model.resize_token_embeddings(len(tokenizer))

world_size = int(os.environ.get("WORLD_SIZE", 1))
client = storage.Client()
print("Creating dataset...")
if world_size > 1:
    train_dataset = {
        "dataset_path": args.data_path,
        "gcs_manager_params": {
                    "gcs_client": client,
                    "local_cache_dir_template": "./cache/worker_{rank}",
                    "preload_size": args.preload_size,
                    "max_load_workers": args.max_load_workers,
                    "remaining_preloaded_threshold": args.remaining_preloaded_threshold,
                    "total_cache_size_mb": args.total_cache_size_mb,
                    "world_size": world_size,
                },
        "tokenizer": tokenizer,
        "audio_features_extractor": audio_feature_extractor,
        "max_length": args.max_length,
        "problem_type": args.problem_type,
        "architecture_type": args.architecture_type,
        "sampling_rate": args.sampling_rate,
        "max_duration_s": args.max_duration_s,
        "validate_json_file": True,
        "buffer_size": 8192
    }
    total_examples = JSONLManager(args.data_path, True).count_examples()
else:
    gcs_manager = GCSManager(
        gcs_client=client,
        local_cache_dir="./cache",
        preload_size=args.preload_size,
        max_load_workers=args.max_load_workers,
        remaining_preloaded_threshold=args.remaining_preloaded_threshold,
        max_cache_size_mb=args.total_cache_size_mb,
    )
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
    total_examples = train_dataset.get_num_examples()

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
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    dataloader_num_workers=args.num_workers,
    logging_steps=100,
    use_cpu=not torch.cuda.is_available(),
    use_stable_adam=args.use_stable_adam,
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
    callbacks=[save_args_callback]
)

print("Starting training...")
trainer.train()