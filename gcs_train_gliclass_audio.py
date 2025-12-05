from gliclass.audio_data_processing import GCSManager
import json
from google.cloud import storage
from tqdm import tqdm
import time
from torch import nn
from pathlib import Path
from transformers import TrainerCallback
from transformers import AutoTokenizer, AutoFeatureExtractor
from gliclass.audio_data_processing import GLiClassAudioDataset
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import random
from huggingface_hub import HfApi
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoConfig, AutoFeatureExtractor
from gliclass import GLiClassModelConfig, GLiClassModel
from gliclass.training import TrainingArguments, Trainer
from gliclass.data_processing import DataCollatorWithPadding
from gliclass.audio_data_processing import GLiClassAudioDataset, JSONLManager
from model_builder import ModelBuilder


class Args:
    model_name = None
    encoder_model_name = "microsoft/deberta-v3-base"
    audio_model_name = "facebook/hubert-large-ls960-ft"
    data_path = "/home/alexandrlukashov/gliclass-audio-experiments/GLiClass/train_next.jsonl"
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
    batch_size = 8
    gradient_accumulation_steps = 4
    
    encoder_lr = 1e-6
    audio_lr = 1e-6
    others_lr = 1e-6
    encoder_weight_decay = 0.001
    audio_weight_decay = 0.001
    others_weight_decay = 0.001
    warmup_ratio = 0.05
    lr_scheduler_type = "cosine"
    
    focal_loss_alpha = -1
    focal_loss_gamma = 0.0
    
    audio_text_contrastive_coef = 0.1
    audio_text_temperature = 0.07
    contrastive_loss_coef = 0.
    
    max_length = 2048
    sampling_rate = 16000
    max_duration_s = 15
    save_steps = 100
    save_total_limit = 5
    use_stable_adam = False
    num_workers = 1
    fp16 = False
    bf16 = True
    save_path = "./models_exp/gliclass-audio-focal"
    
    preload_size = 2000
    max_load_workers = 10
    remaining_preloaded_threshold = 400
    total_cache_size_mb = 32768


class DebugCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model
        self.initial_params = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.initial_params[name] = param.data.clone().cpu()
                if len(self.initial_params) >= 5:
                    break
        
        print(f"Tracking {len(self.initial_params)} params: {list(self.initial_params.keys())}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 50 == 0 and state.global_step > 0:
            print(f"\n=== Step {state.global_step} param check ===")
            for name, initial in self.initial_params.items():
                for n, p in self.model.named_parameters():
                    if n == name:
                        diff = (p.data.cpu() - initial).abs().mean().item()
                        norm = p.data.cpu().norm().item()
                        grad_norm = p.grad.norm().item() if p.grad is not None else 0
                        print(f"{name[:50]}: diff={diff:.6f}, norm={norm:.4f}, grad={grad_norm:.6f}")
                        break
            print("=" * 50 + "\n")
        return control

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
        repo_id = f"{self.username}/gliclass-focal-loss-experiments-{checkpoint_num}"
        try:
            self.api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            self.api.upload_folder(folder_path=checkpoint_path, repo_id=repo_id, repo_type="model")
            print(f"pushed to {repo_id}")
        except Exception as e:
            print(e)
        return control


args = Args()
save_args_callback = HuggingFacePushCallback(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)

if not os.path.exists(args.data_path):
    raise FileNotFoundError(f"Dataset file not found: {args.data_path}")
print(f"dataset file exists: {args.data_path}")

tokenizer = AutoTokenizer.from_pretrained(
    args.encoder_model_name,
    use_fast=False,
    trust_remote_code=True
)
audio_feature_extractor = AutoFeatureExtractor.from_pretrained(args.audio_model_name)

# model_builder = ModelBuilder(device, 'knowledgator/gliclass-base-v1.0-lw', tokenizer, args)
# model = model_builder.swap_layers()
model = GLiClassModel.from_pretrained('alexandrlukashov/gliclass-focal-loss-7000')
new_words = ["<<LABEL>>", "<<SEP>>", "<<AUDIO>>"]
tokenizer.add_tokens(new_words, special_tokens=True)
model.resize_token_embeddings(len(tokenizer))
model.model.logit_scale.data.fill_(0.0)      
# freeze text encoder
for param in model.parameters():
    param.requires_grad = False

# unfreeze projectors
for param in model.model.classes_projector.parameters():
    param.requires_grad = True

for param in model.model.audio_projector.parameters():
    param.requires_grad = True

if hasattr(model.model, 'logit_scale'):
    model.model.logit_scale.requires_grad = True
    print(f"logit_scale unfrozen, current value: {model.model.logit_scale.item():.4f}")

# for name, param in model.model.audio_encoder.named_parameters():
#     if any(f"encoder.layers.{i}" in name for i in range(20, 24)):
#         param.requires_grad = True
#     if "encoder.layer_norm" in name or "final_layer_norm" in name:
#         param.requires_grad = True


model.config.audio_text_contrastive_coef = args.audio_text_contrastive_coef
model.config.audio_text_temperature = args.audio_text_temperature
model.config.focal_loss_alpha = args.focal_loss_alpha
model.config.focal_loss_gamma = args.focal_loss_gamma

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"trainable: {trainable:,} ({trainable/(trainable+frozen)*100:.1f}%)")
print(f"frozen: {frozen:,} ({frozen/(trainable+frozen)*100:.1f}%)")

world_size = int(os.environ.get("WORLD_SIZE", 1))
client = storage.Client()
print("creating dataset...")

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
debug_callback = DebugCallback(model)

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


effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
steps_per_epoch = total_examples // max(effective_batch_size, 1)
if steps_per_epoch == 0:
    steps_per_epoch = 1
max_steps = steps_per_epoch * args.num_epochs

print(f"total examples: {total_examples}")
print(f"effective batch size: {effective_batch_size}")
print(f"steps per epoch: {steps_per_epoch}")
print(f"max steps: {max_steps}")

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
    #warmup_ratio=args.warmup_ratio,
    warmup_steps=200,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    max_steps=max_steps,
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    dataloader_num_workers=args.num_workers,
    logging_steps=10,
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
    callbacks=[save_args_callback, debug_callback]
)

print("starting training...")
trainer.train()