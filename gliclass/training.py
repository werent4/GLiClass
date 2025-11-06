from typing import Optional, Tuple, Dict, List, Union, Any, Callable
from tqdm import tqdm
import numpy as np
import os

from dataclasses import dataclass, field
import torch
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)
import transformers
from transformers import ZeroShotClassificationPipeline as TransformersClassificationPipeline
from .utils import default_f1_reward
from .pipeline import ZeroShotClassificationPipeline
from collections import defaultdict
from lion_pytorch import Lion
import torch.distributed as dist

def get_component_name(param_name):
    if 'audio_encoder' in param_name:
        return 'audio_encoder'
    elif 'audio_projector' in param_name:
        return 'audio_projector'
    elif 'text_projector' in param_name:
        return 'text_projector'
    elif 'encoder_model' in param_name or ('encoder' in param_name and 'audio_encoder' not in param_name):
        return 'text_encoder'
    elif 'scorer' in param_name:
        return 'scorer'
    else:
        return 'others'

def check_frozen_layers(model):
    print("=== FROZEN LAYERS CHECK ===")
    
    component_stats = {
        'audio_encoder': {'total': 0, 'trainable': 0, 'frozen': 0},
        'text_encoder': {'total': 0, 'trainable': 0, 'frozen': 0},
        'audio_projector': {'total': 0, 'trainable': 0, 'frozen': 0},
        'text_projector': {'total': 0, 'trainable': 0, 'frozen': 0},
        'others': {'total': 0, 'trainable': 0, 'frozen': 0}
    }
    
    for name, param in model.named_parameters():
        component = get_component_name(name)
        component_stats[component]['total'] += param.numel()
        
        if param.requires_grad:
            component_stats[component]['trainable'] += param.numel()
            print(f"GOOD {name}: {param.numel():,} params, requires_grad=True")
        else:
            component_stats[component]['frozen'] += param.numel()
            print(f"BADBADBAD {name}: {param.numel():,} params, requires_grad=FALSE")
    
    for comp, stats in component_stats.items():
        if stats['total'] > 0:
            pct = stats['trainable'] / stats['total'] * 100
            print(f"{comp}: {stats['trainable']:,}/{stats['total']:,} ({pct:.1f}%) trainable")

def check_gradient_flow(model):
    print("=== GRADIENT FLOW CHECK ===")

    no_grad_layers = []
    small_grad_layers = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                no_grad_layers.append(name)
                print(f"NO GRADIENT: {name}")
            elif torch.norm(param.grad) < 1e-8:
                small_grad_layers.append(name)
                print(f"TINY GRADIENT: {name}, norm={torch.norm(param.grad):.2e}")
            else:
                print(f"OK GRADIENT: {name}, norm={torch.norm(param.grad):.2e}")
    
    print(f"\nSummary:")
    print(f"Layers with NO gradients: {len(no_grad_layers)}")
    print(f"Layers with TINY gradients: {len(small_grad_layers)}")

def check_detached_branches(model):
    print("=== DETACHED BRANCHES CHECK ===")
    
    def hook_fn(module, input, output):
        if hasattr(output, 'requires_grad'):
            print(f"{module.__class__.__name__}: requires_grad={output.requires_grad}")
        elif isinstance(output, tuple):
            for i, out in enumerate(output):
                if hasattr(out, 'requires_grad'):
                    print(f"{module.__class__.__name__}[{i}]: requires_grad={out.requires_grad}")
    
    for name, module in model.named_modules():
        if any(x in name for x in ['audio_encoder', 'text_encoder', 'projector']):
            module.register_forward_hook(hook_fn)

def check_suspicious_areas(model):
    print("=== SUSPICIOUS AREAS CHECK ===")
    
    for name, param in model.named_parameters():
        if 'embedding' in name.lower():
            print(f"Embedding {name}: requires_grad={param.requires_grad}")

    for name, module in model.named_modules():
        if any(x in name.lower() for x in ['norm', 'batch', 'layer']):
            for pname, param in module.named_parameters():
                print(f"Norm {name}.{pname}: requires_grad={param.requires_grad}")
    
    for name, param in model.named_parameters():
        if any(x in name for x in ['classifier', 'head', 'scorer']):
            print(f"Head {name}: requires_grad={param.requires_grad}")

def check_feature_normalization(model):
    print("=== FEATURE NORMALIZATION CHECK ===")
    
    def feature_hook(name):
        def hook(module, input, output):
            if hasattr(output, 'data'):
                data = output.data
                print(f"{name} output:")
                print(f"    Mean: {data.mean():.4f}, Std: {data.std():.4f}")
                print(f"    Range: [{data.min():.2f}, {data.max():.2f}]")
                print(f"    Shape: {data.shape}")
                
        return hook
    
    key_modules = ['audio_projector', 'text_projector', 'scorer']
    for name, module in model.named_modules():
        if any(key in name for key in key_modules):
            module.register_forward_hook(feature_hook(name))

def check_activation_saturation(model):
    print("=== ACTIVATION SATURATION CHECK ===")
    
    saturation_hooks = []
    
    def activation_hook(name, activation_type):
        def hook(module, input, output):
            if isinstance(input, tuple):
                pre_activation = input[0]
            else:
                pre_activation = input
                
            if hasattr(pre_activation, 'data'):
                data = pre_activation.data
                
                if activation_type == 'sigmoid':
                    saturated = torch.abs(data) > 5.0
                    saturated_pct = saturated.float().mean().item() * 100
                    
                elif activation_type == 'tanh':  
                    saturated = torch.abs(data) > 3.0
                    saturated_pct = saturated.float().mean().item() * 100
                    
                elif activation_type == 'relu':
                    dead = data < 0
                    saturated_pct = dead.float().mean().item() * 100
                    
                print(f"{name} ({activation_type}): {saturated_pct:.1f}% saturated")
                print(f"    Pre-activation range: [{data.min():.2f}, {data.max():.2f}]")
                
        return hook
    
    for name, module in model.named_modules():
        if 'sigmoid' in str(type(module)).lower():
            hook = module.register_forward_hook(activation_hook(name, 'sigmoid'))
            saturation_hooks.append(hook)
        elif 'tanh' in str(type(module)).lower():
            hook = module.register_forward_hook(activation_hook(name, 'tanh'))
            saturation_hooks.append(hook)
        elif isinstance(module, torch.nn.ReLU):
            hook = module.register_forward_hook(activation_hook(name, 'relu'))
            saturation_hooks.append(hook)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    others_lr: Optional[float] = None
    others_weight_decay: Optional[float] = 0.0
    audio_lr: Optional[float] = None 
    audio_weight_decay: Optional[float] = 0.0

class Trainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids("<<AUDIO>>") if hasattr(self, 'tokenizer') else None

    def training_step(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        model.train()
        loss = None
        error_flag = torch.tensor(0, device=self.args.device)
        try:
            if "labels_text" in inputs:
                            labels_text = inputs.pop('labels_text')
            if "input_texts" in inputs:
                            input_texts = inputs.pop('input_texts')
            if "original_seq_len" in inputs:
                            original_seq_len = inputs.pop('original_seq_len')
            print(f"original_seq_len: {original_seq_len}")
            inputs = self._prepare_inputs(inputs)
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)  
        except Exception as e:
            print(f"Skipping iteration due to error: {e}")
            error_flag = torch.tensor(1, device=self.args.device)
        if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(error_flag, op=torch.distributed.ReduceOp.MAX)
        if error_flag.item() > 0:
                    model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    return torch.tensor(0.0, device=self.args.device)
        if hasattr(self, '_step_counter'):
            self._step_counter += 1
        else:
            self._step_counter = 1
        del inputs
        torch.cuda.empty_cache()
        kwargs = {}
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)
        return loss.detach() / self.args.gradient_accumulation_steps
        
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on model using inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (nn.Module):
                The model to evaluate.
            inputs (Dict[str, Union[torch.Tensor, Any]]):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument labels. Check your model's documentation for all accepted arguments.
            prediction_loss_only (bool):
                Whether or not to return the loss only.
            ignore_keys (List[str], *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        try:
            with torch.no_grad():
                if "labels_text" in inputs:
                    labels_text = inputs.pop('labels_text')
                if "input_texts" in inputs:
                    input_texts = inputs.pop('input_texts')
                loss = None
                with self.compute_loss_context_manager():
                    try:
                        outputs = model(**inputs)
                    except Exception as e:
                        raise RuntimeError(f"Error during model forward pass: {str(e)}")

                if not hasattr(outputs, 'loss'):
                    raise AttributeError("Model output does not contain 'loss' attribute")
                loss = outputs.loss

                if not hasattr(outputs, 'logits'):
                    raise AttributeError("Model output does not contain 'logits' attribute")
                logits = outputs.logits

                if 'labels' not in inputs:
                    raise KeyError("'labels' not found in input dictionary")
                labels = inputs['labels']

            if prediction_loss_only:
                return (loss, None, None)
            return (loss, logits, labels)

        except Exception as e:
            print(f"An error occurred during prediction step: {str(e)}")
            return (None, None, None)
        
    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            audio_encoder_parameters = [name for name, _ in opt_model.named_parameters() if "audio_encoder" in name]
            text_encoder_parameters = [name for name, _ in opt_model.named_parameters() if "encoder" in name and "audio_encoder" not in name]
            optimizer_grouped_parameters = []
            # text encoder
            optimizer_grouped_parameters.extend([
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in text_encoder_parameters and n not in audio_encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in text_encoder_parameters and n not in audio_encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
            ])
            
            # audio encoder
            if self.args.audio_lr is not None:
                 optimizer_grouped_parameters.extend([{
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in text_encoder_parameters and n in audio_encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.others_weight_decay,
                        "lr": self.args.audio_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in text_encoder_parameters and n in audio_encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.audio_lr,
                    }
                ])
            
            # Others
            if self.args.others_lr is not None:
                optimizer_grouped_parameters.extend([
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in text_encoder_parameters and n not in audio_encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.others_weight_decay,
                        "lr": self.args.others_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in text_encoder_parameters and n not in audio_encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.others_lr,
                    },
                ])
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # lion_kwargs = {}
            # valid_lion_params = {'lr', 'betas', 'weight_decay'}
            # for key, value in optimizer_kwargs.items():
            #     if key in valid_lion_params:
            #         lion_kwargs[key] = value
            
            # if 'betas' not in lion_kwargs:
            #     lion_kwargs['betas'] = (0.9, 0.99) 

            # self.optimizer = Lion(optimizer_grouped_parameters, **lion_kwargs)

        return self.optimizer

@dataclass
class RLTrainerConfig(TrainingArguments):
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Clip range."},
    )
    num_rl_iters: int = field(
        default=3,
        metadata={"help": "Number of RL iterations."},
    )
    gamma: float = field(
        default=-1,
        metadata={"help": "Focal loss gamma."},
    )
    alpha: float = field(
        default=-1,
        metadata={"help": "Focal loss alpha."},
    )
    labels_smoothing: float = field(
        default=-1,
        metadata={"help": "Labels smoothing factor."}
    )
    entropy_beta: float = field(
        default=-1,
        metadata={"help": "Coeficient of entropy factor."}
    )
    kl_beta: float = field(
        default=-1,
        metadata={"help": "Coeficient of KL-divergence factor."}
    )
    get_actions: str = field(
        default="bernoulli",
        metadata={"help": "How to get actions of a model, default is `bernoulli`, another option is `threshold`"},
    )
    threshold: float = field(
        default=0.5,
        metadata={"help": "Threshold value for predictions."},
    )

class RLTrainer(Trainer):
    def __init__(
        self,
        value_model: Optional[torch.nn.Module] = None,
        reference_model: Optional[Union[ZeroShotClassificationPipeline|TransformersClassificationPipeline]] = None,
        reward_components: Optional[List[Tuple[str, Callable]]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if value_model is not None:
            self.value_model = value_model.to(self.model.device)
        self.reference_model = reference_model
        if reward_components is None:
            reward_components = [('f1', default_f1_reward)]
        self.reward_components = reward_components
        self._init_metrics()

    def _init_metrics(self):
        self.metrics = {
            'total_loss': [],
            'advantages': [],
        }
        # Initialize metrics for each reward component
        for name, _ in self.reward_components.items():
            self.metrics[f'reward_{name}'] = []

    def compute_rewards(
        self,
        probs: torch.Tensor,
        actions: torch.Tensor,
        original_targets: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        rewards = {}
        total_reward = 0.0
        for name, reward_fn in self.reward_components.items():
            component = reward_fn(probs, actions, original_targets, valid_mask)
            rewards[name] = component
            total_reward += component
        rewards['total_reward'] = total_reward
        return rewards
    
    def get_reference_scores(self, input_texts, labels_text):
        if input_texts is None or labels_text is None:
            return None
        all_scores = []
        with torch.no_grad():
            if isinstance(self.reference_model, ZeroShotClassificationPipeline):
                results = self.reference_model(input_texts, labels_text, threshold=0.)
                for id, result in enumerate(results):
                    label2score = {item['label']: item['score'] for item in result}
                    label_scores = [label2score[label] for label in labels_text[id]]
                    all_scores.append(label_scores)
            elif isinstance(self.reference_model, TransformersClassificationPipeline):
                for text, labels in zip(input_texts, labels_text):
                    result = self.reference_model(text, labels)
                    label2score = {label:score for label, score in zip(result['labels'], result['scores'])}
                    label_scores = [label2score[label] for label in labels_text[id]]
                    all_scores.append(label_scores)
            else:
                raise NotImplementedError("This classification pipelines is not supported as a reference model.")
        max_length = max(len(seq) for seq in all_scores)
        all_scores = torch.FloatTensor([seq + [0] * (max_length - len(seq)) 
                                            for seq in all_scores]).to(self.model.device)
        return all_scores
    
    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        log_prob_prev: Optional[torch.Tensor] = None,
        value_outputs: Optional[torch.Tensor] = None,
        reference_probs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_mask = targets != -100
        original_targets = targets.clone()

        probs = torch.sigmoid(inputs)

        if self.args.get_actions == 'bernoulli':
            actions = torch.bernoulli(probs).detach()
        else:
            actions = (probs > self.args.threshold).float().detach()

        with torch.no_grad():
            metrics = self.compute_rewards(probs, actions, original_targets, valid_mask)

        reward = metrics['total_reward']

        if value_outputs is not None:
            state_values = value_outputs.logits[:, 0].unsqueeze(-1)  # Using first token logits as value prediction
            value_loss = torch.nn.functional.mse_loss(state_values, reward.detach())
        else:
            state_values = reward.mean()
            value_loss = torch.tensor(0.0).to(inputs.device)

        advantages = (reward - state_values).detach()
        self.metrics['advantages'].append(advantages.mean().item())

        for name, _ in self.reward_components.items():
            key = f'reward_{name}'
            self.metrics[key].append(metrics[name].mean().item())

        if self.args.label_smoothing_factor > 0:
            smoothed_actions = actions * (1 - self.args.label_smoothing_factor) + 0.5 * self.args.label_smoothing_factor
            log_prob_current = (
                smoothed_actions * torch.log(probs + 1e-8) +
                (1 - smoothed_actions) * torch.log(1 - probs + 1e-8)
            )
        else:
            log_prob_current = (
                actions * torch.log(probs + 1e-8) +
                (1 - actions) * torch.log(1 - probs + 1e-8)
            )

        if log_prob_prev is None:
            log_prob_prev = log_prob_current.detach()

        log_probs_diff = log_prob_current - log_prob_prev
        ratio = torch.exp(log_probs_diff)

        cliprange = self.args.cliprange
        per_label_loss1 = ratio * advantages
        per_label_loss2 = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages
        loss_elements = -torch.min(per_label_loss1, per_label_loss2)

        loss_elements = loss_elements * valid_mask
        self.metrics['total_loss'].append(loss_elements.mean().item())

        if self.args.gamma > 0:
            p_t = probs * original_targets + (1 - probs) * (1 - original_targets)
            loss_elements = loss_elements * (p_t ** self.args.gamma)

        if self.args.alpha >= 0:
            alpha_t = self.args.alpha * original_targets + (1 - self.args.alpha) * (1 - original_targets)
            loss_elements = alpha_t * loss_elements

        loss = loss_elements.sum() / valid_mask.shape[0] + value_loss

        if reference_probs is not None:
            ref_per_token_logps = torch.log(reference_probs + 1e-8)
            per_token_logps = log_prob_current  
            per_label_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_label_kl = per_label_kl * valid_mask
            kl_loss = self.args.kl_beta * per_label_kl.mean()
            loss = loss + kl_loss

        if self.args.entropy_beta:
            entropy = - (probs * torch.log(probs + 1e-8) +
                        (1 - probs) * torch.log(1 - probs + 1e-8))
            loss = loss + self.args.entropy_beta * entropy.mean()

        return loss, log_prob_current


    def _inner_training_loop(self, *args, **kwargs):
        self.create_optimizer()
        if self.value_model is not None:
            value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=self.args.learning_rate)
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        dataloader = self.get_train_dataloader()
        device = accelerator.device

        num_local_steps = len(dataloader)
        num_iters = args.num_train_epochs*num_local_steps
        pbar = tqdm(total=num_iters, desc="Training iterations")
        self._init_metrics()

        for epoch in range(args.num_train_epochs):
            self._init_metrics()
            model.train()
            if self.value_model is not None:
                self.value_model.train()

            for step, inputs in enumerate(dataloader):
                global_step = step+epoch*num_local_steps

                inputs = self._prepare_inputs(inputs)
                labels = inputs.pop('labels').to(device)
                if "labels_text" in inputs:
                    labels_text = inputs.pop('labels_text')
                else:
                    labels_text = None
                if "input_texts" in inputs:
                    input_texts = inputs.pop('input_texts')
                else:
                    input_texts = None
                prev_logps = None
                for iter in range(args.num_rl_iters):
                    try:
                        outputs = model(**inputs)
                        logits = outputs.logits
                        if self.value_model is not None:
                            value_outputs = self.value_model(**inputs)
                        else:
                            value_outputs = None
                        if self.reference_model is not None:
                            reference_probs = self.get_reference_scores(input_texts, labels_text)
                        else:
                            reference_probs = None
                        loss, current_logps = self.compute_loss(logits, labels, log_prob_prev=prev_logps, 
                                                                                value_outputs=value_outputs,
                                                                                reference_probs=reference_probs)
                    except Exception as e:
                        print(f"An error occurred during training step: {str(e)}")
                        del inputs
                        model.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        break

                    accelerator.backward(loss)
                    if self.args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                        if self.value_model is not None:
                            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()
                    if self.value_model is not None:
                        value_optimizer.step()
                        value_optimizer.zero_grad()

                    prev_logps = current_logps.detach()

                if global_step % args.logging_steps == 0:
                    self.log_metrics()

                if args.save_steps is not None and global_step % args.save_steps == 0:
                    self._save_checkpoint(model, step=global_step)

                pbar.set_postfix(epoch=epoch, step=step)
                pbar.update(1)

            if args.evaluation_strategy == "epoch":
                self.evaluate()

    def log_metrics(self):
        logged_metrics = {
            'loss': np.mean(self.metrics['total_loss']),
            'advantages': np.mean(self.metrics['advantages']),
        }
        # Add user reward components
        for name, _ in self.reward_components.items():
            key = f'reward_{name}'
            logged_metrics[key] = np.mean(self.metrics[key])
        self.log(logged_metrics)
        self._init_metrics()

    def _save_checkpoint(self, model, step=None):
        checkpoint_dir = f"checkpoint-{step}" if step else "final_model"
        output_dir = os.path.join(self.args.output_dir, checkpoint_dir)
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        print(f"Checkpoint saved to {output_dir}")

class AnalysisTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # torch.autograd.set_detect_anomaly(True)
        self.current_step_gradients = {}
        self.hooks = []
        self.step_count = 0
        self.component_param_counts = {}
        self.analyze_model_parameters()

    def analyze_model_parameters(self):
        print("Model Parameter Analysis:")
        print("=" * 60)
        
        component_stats = {}
        total_params = 0
        total_trainable = 0
        total_frozen = 0
        
        for name, param in self.model.named_parameters():
            component = self._get_component_name(name)
            
            if component not in component_stats:
                component_stats[component] = {
                    'total': 0,
                    'trainable': 0, 
                    'frozen': 0,
                    'param_names': []
                }

            param_count = param.numel()
            component_stats[component]['total'] += param_count
            component_stats[component]['param_names'].append(name)
            
            if param.requires_grad:
                component_stats[component]['trainable'] += param_count
                total_trainable += param_count
            else:
                component_stats[component]['frozen'] += param_count
                total_frozen += param_count
                
            total_params += param_count
        
        print(f"TOTAL PARAMETERS: {total_params:,}")
        print(f"Trainable: {total_trainable:,} ({total_trainable/total_params*100:.1f}%)")
        print(f"Frozen: {total_frozen:,} ({total_frozen/total_params*100:.1f}%)")
        print()

    def setup_hooks(self):
        self._count_component_parameters()
        
        def param_gradient_hook(component_name):
            def hook(grad):
                if grad is not None:
                    grad_norm = torch.norm(grad).item()
                    grad_mean = torch.mean(grad).item()
                    if grad.numel() > 1:
                        grad_std = torch.std(grad, unbiased=False).item()
                    else:
                        grad_std = 0.0

                    if component_name not in self.current_step_gradients:
                        self.current_step_gradients[component_name] = {
                            'norms': [], 'means': [], 'stds': []
                        }
                    
                    self.current_step_gradients[component_name]['norms'].append(grad_norm)
                    self.current_step_gradients[component_name]['means'].append(grad_mean)
                    self.current_step_gradients[component_name]['stds'].append(grad_std)
                return grad
            return hook

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                component = self._get_component_name(name)
                handle = param.register_hook(param_gradient_hook(component))
                self.hooks.append(handle)

    def _get_component_name(self, param_name):
        if 'audio_encoder' in param_name:
            return 'audio_encoder'
        elif 'audio_projector' in param_name:
            return 'audio_projector'
        elif 'text_projector' in param_name:
            return 'text_projector'
        elif 'encoder_model' in param_name or ('encoder' in param_name and 'audio_encoder' not in param_name):
            return 'text_encoder'
        elif 'scorer' in param_name:
            return 'scorer'
        else:
            return 'others'

    def _count_component_parameters(self):
        self.component_param_counts = {
            'audio_encoder': 0,
            'audio_projector': 0, 
            'text_projector': 0,
            'text_encoder': 0,
            'scorer': 0,
            'others': 0
        }
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                component = self._get_component_name(name)
                self.component_param_counts[component] += param.numel()

        print("Component parameter counts:")
        for component, count in self.component_param_counts.items():
            if count > 0:
                print(f"  {component}: {count:,} parameters")
                self.log({f"params/{component}_count": count})
        
    def training_step(self, model, inputs, *args, **kwargs):
        self.current_step_gradients.clear()
        
        loss = super().training_step(model, inputs, *args, **kwargs)
        self.step_count += 1

        if (self.step_count % self.args.gradient_accumulation_steps == 0 and 
            (self.step_count // self.args.gradient_accumulation_steps) % self.args.logging_steps == 0):
            self.log_gradient_stats()
        return loss

    def log_gradient_stats(self):
        print(f"Step {self.step_count}:")
        
        for component, stats in self.current_step_gradients.items():
            if stats['norms']:
                total_norm = sum(stats['norms'])
                avg_mean = sum(stats['means']) / len(stats['means'])
                avg_std = sum(stats['stds']) / len(stats['stds'])
                
                self.log({f"grad/{component}_norm": total_norm})
                self.log({f"grad/{component}_mean": avg_mean})
                self.log({f"grad/{component}_std": avg_std})
                
                param_count = self.component_param_counts.get(component, 1)
                if param_count > 0:
                    normalized_norm = total_norm / param_count
                    self.log({f"grad/{component}_norm_per_param": normalized_norm})

    def cleanup_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        self.cleanup_hooks()