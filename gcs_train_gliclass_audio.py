import sys
sys.path.insert(0, '/home/alexandrlukashov/gliclass-audio-experiments/perception_models')

import gc
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch._dynamo
torch._dynamo.disable()

import json
import random
import numpy as np
import torch
from pathlib import Path
from google.cloud import storage
from huggingface_hub import HfApi
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import TrainerCallback
import pandas as pd
from tqdm import tqdm

from core.audio_visual_encoder import PEAudioVisualTransform
from gliclass.audio_data_processing import GCSManager, GLiClassAudioDataset, JSONLManager
from gliclass import GLiClassModelConfig
from gliclass.model import GLiClassAudio
from gliclass.training import TrainingArguments, Trainer
from gliclass.data_processing import DataCollatorWithPadding


class Args:
    pe_model_name = "pe-av-small"
    pe_pretrained = True
    
    data_path = "/home/alexandrlukashov/gliclass-audio-experiments/GLiClass/data/gliclass-audio-1M-shuffled-shuffled-labels.jsonl"
    problem_type = "multi_label_classification"
    architecture_type = "audio-encoder"
    
    normalize_features = False
    hidden_dropout_prob = 0.1
    logit_scale_init_value = 2.0
    
    sampling_rate = 48000
    num_epochs = 1
    batch_size = 1
    gradient_accumulation_steps = 64
    
    model_lr = 5e-6
    model_weight_decay = 0.05
    others_lr = 5e-4
    others_weight_decay = 0.01
    
    warmup_ratio = 0.001
    lr_scheduler_type = "cosine"
    
    focal_loss_alpha = 0.25
    focal_loss_gamma = 2.0
    
    shuffle_labels = True
    save_steps = 100
    save_total_limit = 1
    use_stable_adam = False
    num_workers = 1
    fp16 = False
    bf16 = True
    save_path = "./models_exp/gliclass-audio-pe-zeroshot"
    
    gradient_checkpointing = True
    
    preload_size = 2000
    max_load_workers = 10
    remaining_preloaded_threshold = 400
    total_cache_size_mb = 32768


class ESCEvalCallback(TrainerCallback):
    def __init__(self, dataset_path: str, audio_dir: str, transform: PEAudioVisualTransform):
        self.dataset_path = dataset_path
        self.audio_dir = Path(audio_dir)
        self.transform = transform
        self.dataset_meta = pd.read_csv(dataset_path)
        self.all_categories = sorted(self.dataset_meta['category'].unique().tolist())
        print(f"ESCEvalCallback: {len(self.dataset_meta)} samples, {len(self.all_categories)} categories")
    
    @torch.no_grad()
    def run_evaluation(self, model, device):
        model.eval()
        
        if hasattr(model, 'logit_scale'):
            logit_scale = model.logit_scale.exp().item()
        else:
            logit_scale = 1.0
        
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        total = len(self.dataset_meta)
        
        for _, row in tqdm(self.dataset_meta.iterrows(), total=total, desc="ESC-50 Eval"):
            audio_path = str(self.audio_dir / row['filename'])
            true_label = row['category']
            
            inputs = self.transform(text=self.all_categories, audio=[audio_path]).to(device)
            
            with torch.autocast(device.type, dtype=torch.bfloat16):
                outputs = model(**inputs)
            
            logits = outputs.logits.squeeze(0)
            scores = (logits / logit_scale).float().cpu().numpy()
            
            top_indices = np.argsort(scores)[::-1]
            top1_pred = self.all_categories[top_indices[0]]
            top3_preds = [self.all_categories[i] for i in top_indices[:3]]
            top5_preds = [self.all_categories[i] for i in top_indices[:5]]
            
            if top1_pred == true_label:
                correct_top1 += 1
            if true_label in top3_preds:
                correct_top3 += 1
            if true_label in top5_preds:
                correct_top5 += 1
        
        model.train()
        
        return {
            'top1_acc': correct_top1 / total,
            'top3_acc': correct_top3 / total,
            'top5_acc': correct_top5 / total,
        }
    
    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        
        device = next(model.parameters()).device
        
        print(f"\n{'='*60}")
        print(f"ESC-50 EVALUATION @ Step {state.global_step}")
        print(f"{'='*60}")
        
        results = self.run_evaluation(model, device)
        
        print(f"Top-1 Accuracy: {results['top1_acc']:.4f} ({results['top1_acc']*100:.2f}%)")
        print(f"Top-3 Accuracy: {results['top3_acc']:.4f} ({results['top3_acc']*100:.2f}%)")
        print(f"Top-5 Accuracy: {results['top5_acc']:.4f} ({results['top5_acc']*100:.2f}%)")
        print(f"{'='*60}\n")
        
        return control


class HuggingFacePushCallback(TrainerCallback):
    def __init__(self, args_to_save, username="alexandrlukashov", repo_prefix="gliclass-pe-audio-zeroshot"):
        self.username = username
        self.repo_prefix = repo_prefix
        self.api = HfApi()
        self.args_to_save = args_to_save

    def on_save(self, args, state, control, model=None, **kwargs):
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
        
        repo_id = f"{self.username}/{self.repo_prefix}-{checkpoint_num}"
        try:
            self.api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            self.api.upload_folder(folder_path=checkpoint_path, repo_id=repo_id, repo_type="model")
            print(f"Pushed to {repo_id}")
        except Exception as e:
            print(f"Error pushing to HF: {e}")
        return control


class DebugCallback(TrainerCallback):
    def __init__(self, model, top_k=5):
        self.model = model
        self.top_k = top_k
        self.prev_params = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.prev_params[name] = param.data.clone().cpu()
        
        print(f"Tracking {len(self.prev_params)} trainable params")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 1 == 0 and state.global_step > 0:
            diffs = []
            for name, prev in self.prev_params.items():
                for n, p in self.model.named_parameters():
                    if n == name:
                        diff = (p.data.cpu() - prev).abs().mean().item()
                        short_name = name.split('.')[-3] + '.' + name.split('.')[-2] + '.' + name.split('.')[-1]
                        diffs.append((short_name, diff))
                        self.prev_params[name] = p.data.clone().cpu()
                        break
            
            top = sorted(diffs, key=lambda x: x[1], reverse=True)[:self.top_k]
            top_str = ' | '.join([f"{n}: {d:.6f}" for n, d in top])
            print(f"{{step {state.global_step}}} {{{top_str}}}")
        
        return control


def enable_gradient_checkpointing(model):
    if hasattr(model.model, 'audio_visual_model'):
        audio_transformer = model.model.audio_visual_model.audio_model.audio_transformer
        if hasattr(audio_transformer, 'gradient_checkpointing_enable'):
            audio_transformer.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for audio_transformer")
        else:
            audio_transformer.gradient_checkpointing = True
            print("Set gradient_checkpointing=True for audio_transformer")
    
    if hasattr(model.model, 'text_model'):
        if hasattr(model.model.text_model, 'gradient_checkpointing_enable'):
            model.model.text_model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for text_model")
    
    return model


def compute_metrics(p):
    predictions, labels = p
    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)
    preds = (predictions > 0.5).astype(int)
    labels = np.where(labels > 0.5, 1, 0)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main():
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset file not found: {args.data_path}")
    print(f"Dataset file exists: {args.data_path}")

    transform = PEAudioVisualTransform.from_config(args.pe_model_name)
    print(f"Loaded transform: {args.pe_model_name}")

    config = GLiClassModelConfig(
        architecture_type=args.architecture_type,
        problem_type=args.problem_type,
        pe_model_name=args.pe_model_name,
        pe_pretrained=args.pe_pretrained,
        normalize_features=args.normalize_features,
        hidden_dropout_prob=args.hidden_dropout_prob,
        logit_scale_init_value=args.logit_scale_init_value,
        focal_loss_alpha=args.focal_loss_alpha,
        focal_loss_gamma=args.focal_loss_gamma,
    )

    model = GLiClassAudio(config, from_pretrained=args.pe_pretrained)
    
    if args.gradient_checkpointing:
        model = enable_gradient_checkpointing(model)
    
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    print(f"\nTotal params: {total:,}")
    print(f"Trainable: {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"Frozen: {frozen:,} ({frozen/total*100:.2f}%)")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    client = storage.Client()
    print("\nCreating dataset...")

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
            "transform": transform,
            "problem_type": args.problem_type,
            "shuffle_labels": args.shuffle_labels,
            "sampling_rate": args.sampling_rate, 
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
            transform=transform,
            problem_type=args.problem_type,
            shuffle_labels=args.shuffle_labels,
            validate_json_file=True,
            buffer_size=8192,
        )
        total_examples = train_dataset.get_num_examples()

    data_collator = DataCollatorWithPadding(device=device)

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
    steps_per_epoch = total_examples // max(effective_batch_size, 1)
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    max_steps = steps_per_epoch * args.num_epochs

    print(f"Total examples: {total_examples}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Max steps: {max_steps}")

    training_args = TrainingArguments(
        dataloader_pin_memory=False,
        output_dir=args.save_path,
        learning_rate=args.model_lr,
        weight_decay=args.model_weight_decay,
        model_lr=args.model_lr,
        model_weight_decay=args.model_weight_decay,
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
        logging_steps=1,
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

    esc_eval_callback = ESCEvalCallback(
        dataset_path='/home/alexandrlukashov/gliclass-audio-experiments/GLiClass/ESC-50-master/meta/esc50.csv',
        audio_dir='/home/alexandrlukashov/gliclass-audio-experiments/GLiClass/ESC-50-master/audio',
        transform=transform,
    )
    save_args_callback = HuggingFacePushCallback(args, repo_prefix="gliclass-pe-audio-zeroshot")
    debug_callback = DebugCallback(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[esc_eval_callback, save_args_callback, debug_callback]
    )

    print("\nStarting training...")
    trainer.train()


if __name__ == "__main__":
    main()