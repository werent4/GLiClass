import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
            dist.all_gather(output, input)
            return torch.cat(output, dim=0)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            batch_size = input.shape[0]
            return grad_output[rank * batch_size:(rank + 1) * batch_size]
        return grad_output


# def audio_text_contrastive_loss(audio_embeds, text_embeds, labels, margin=0.3, collapse_coef=0.5, align_coef=0.5):
#     batch_size, num_classes, dim = text_embeds.shape
#     dtype = audio_embeds.dtype
    
#     if audio_embeds.dim() == 3:
#         audio_embeds = audio_embeds.mean(dim=1)
    
#     audio_normed = F.normalize(audio_embeds.float(), p=2, dim=-1).to(dtype)
#     text_normed = F.normalize(text_embeds.float(), p=2, dim=-1).to(dtype)
    
#     labels_float = labels.float().to(dtype)
#     neg_mask = 1 - labels_float
    
#     all_sim = torch.bmm(audio_normed.unsqueeze(1), text_normed.transpose(1, 2)).squeeze(1)
    
#     pos_sim = (all_sim * labels_float).sum(dim=-1) / labels_float.sum(dim=-1).clamp(min=1)
#     neg_sim = (all_sim * neg_mask).sum(dim=-1) / neg_mask.sum(dim=-1).clamp(min=1)
    
#     margin_loss = F.relu(neg_sim - pos_sim + margin).mean()
    
#     align_loss = (1 - pos_sim).mean()
    
#     audio_sim = audio_normed @ audio_normed.T
#     audio_mask = ~torch.eye(batch_size, dtype=bool, device=audio_sim.device)
#     audio_collapse = audio_sim[audio_mask].pow(2).mean()
    
#     total_loss = margin_loss + align_coef * align_loss + collapse_coef * audio_collapse
    
#     gap = (pos_sim - neg_sim).mean().item()
#     print(f"pos: {pos_sim.mean().item():.4f}, neg: {neg_sim.mean().item():.4f}, gap: {gap:.4f}, align: {align_loss.item():.4f}, collapse: {audio_collapse.item():.4f}")
    
#     return total_loss


class InfoNCELoss(nn.Module):
    def __init__(self, init_temperature=0.07, margin=0.5, focal_alpha=0.25, 
                 focal_gamma=2.0, focal_coef=0.1, accumulation_steps=4):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        self.margin = margin
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_coef = focal_coef
        self.accumulation_steps = accumulation_steps
        
        self.accumulated_audio = []
        self.current_step = 0
    
    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(min=0.01, max=100.0)
    
    def focal_loss(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
        if self.focal_gamma > 0:
            p_t = p * targets + (1 - p) * (1 - targets)
            ce_loss = ce_loss * ((1 - p_t) ** self.focal_gamma)
        
        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            ce_loss = alpha_t * ce_loss
        
        return ce_loss.mean()
    
    def reset_accumulation(self):
        self.accumulated_audio = []
        self.current_step = 0
    
    def forward(self, audio_embeds, text_embeds, labels):
        self.accumulated_audio.append(audio_embeds.detach())
        self.current_step += 1
        
        all_sim = torch.bmm(audio_embeds.unsqueeze(1), text_embeds.transpose(1, 2)).squeeze(1)
        all_sim_scaled = all_sim / self.temperature
        
        labels_float = labels.float()
        neg_mask = 1 - labels_float
        
        log_softmax_a2t = F.log_softmax(all_sim_scaled, dim=-1)
        loss_a2t = -(log_softmax_a2t * labels_float).sum() / labels_float.sum().clamp(min=1)
        
        if len(self.accumulated_audio) > 1:
            all_audio = torch.cat(self.accumulated_audio, dim=0)
            audio_sim = all_audio @ all_audio.T / self.temperature
            batch_size = all_audio.shape[0]
            audio_labels = torch.arange(batch_size, device=all_audio.device)
            loss_t2a = F.cross_entropy(audio_sim, audio_labels)
        else:
            loss_t2a = torch.tensor(0.0, device=audio_embeds.device)
        
        pos_sim = (all_sim * labels_float).sum(dim=-1) / labels_float.sum(dim=-1).clamp(min=1)
        neg_sim = (all_sim * neg_mask).sum(dim=-1) / neg_mask.sum(dim=-1).clamp(min=1)
        margin_loss = F.relu(neg_sim - pos_sim + self.margin).mean()
        
        focal_loss = self.focal_loss(all_sim, labels_float)
        
        loss = loss_a2t + 0.5 * loss_t2a + margin_loss + self.focal_coef * focal_loss
        
        if self.current_step >= self.accumulation_steps:
            self.reset_accumulation()
        
        with torch.no_grad():
            gap = (pos_sim - neg_sim).mean().item()
            pos_mean = pos_sim.mean().item()
            neg_mean = neg_sim.mean().item()
            mid = (pos_mean + neg_mean) / 2
            
            best_f1, best_thresh = 0, mid
            for thresh in [mid - 0.15, mid - 0.1, mid - 0.05, mid, mid + 0.05, mid + 0.1, mid + 0.15]:
                preds = (all_sim > thresh).float()
                tp = (preds * labels_float).sum()
                fp = (preds * neg_mask).sum()
                fn = ((1 - preds) * labels_float).sum()
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                if f1 > best_f1:
                    best_f1 = f1.item()
                    best_thresh = thresh
            
            preds = (all_sim > best_thresh).float()
            tp = (preds * labels_float).sum()
            fp = (preds * neg_mask).sum()
            fn = ((1 - preds) * labels_float).sum()
            tn = ((1 - preds) * neg_mask).sum()
            
            precision = (tp / (tp + fp + 1e-8)).item()
            recall = (tp / (tp + fn + 1e-8)).item()
            f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
            acc = ((tp + tn) / (tp + tn + fp + fn + 1e-8)).item()
            
            top1_preds = all_sim.argmax(dim=-1)
            top1_targets = labels_float.argmax(dim=-1)
            top1_acc = (top1_preds == top1_targets).float().mean().item()
            
            print(f"pos: {pos_mean:.4f}, neg: {neg_mean:.4f}, gap: {gap:.4f}, temp: {self.temperature.item():.4f}, accum: {len(self.accumulated_audio)}/{self.accumulation_steps}")
            print(f"loss_a2t: {loss_a2t.item():.4f}, loss_audio_contr: {loss_t2a.item():.4f}, margin: {margin_loss.item():.4f}, focal: {focal_loss.item():.4f}")
            print(f"acc: {acc:.4f}, prec: {precision:.4f}, rec: {recall:.4f}, f1: {f1:.4f}, top1: {top1_acc:.4f}, thresh: {best_thresh:.2f}")
        
        return loss


# def audio_text_contrastive_loss(audio_embeds, text_embeds, labels, temperature=0.1):
#     batch_size, num_classes, dim = text_embeds.shape
    
#     all_sim = torch.bmm(audio_embeds.unsqueeze(1), text_embeds.transpose(1, 2)).squeeze(1)
#     all_sim = all_sim / temperature
    
#     target_idx = labels.argmax(dim=-1)  
    
#     loss = F.cross_entropy(all_sim, target_idx)
    
#     with torch.no_grad():
#         preds = all_sim.argmax(dim=-1)
#         acc = (preds == target_idx).float().mean().item()
#         print(f"loss: {loss.item():.4f}, acc: {acc:.4f}")
    
#     return loss

# def audio_text_contrastive_loss(audio_embeds, text_embeds, labels, margin=0.2, collapse_coef=1):
#     batch_size, num_classes, dim = text_embeds.shape
#     dtype = audio_embeds.dtype

#     if audio_embeds.dim() == 3:
#         audio_embeds = audio_embeds.mean(dim=1)

#     audio_normed = F.normalize(audio_embeds.float(), p=2, dim=-1).to(dtype)
#     text_normed = F.normalize(text_embeds.float(), p=2, dim=-1).to(dtype)

#     labels_float = labels.float().to(dtype)

#     all_sim = torch.bmm(audio_normed.unsqueeze(1), text_normed.transpose(1, 2)).squeeze(1)

#     pos_sim = (all_sim * labels_float).sum(dim=-1) / labels_float.sum(dim=-1).clamp(min=1)
#     neg_sim_hard = (all_sim - 1e9 * labels_float).max(dim=-1).values

#     attract_loss = F.relu(0.5 - pos_sim).mean()
#     repel_loss = F.relu(neg_sim_hard + 0.2).mean()
#     margin_loss = F.relu(neg_sim_hard - pos_sim + margin).mean()

#     audio_sim = audio_normed @ audio_normed.T
#     audio_mask = ~torch.eye(batch_size, dtype=bool, device=audio_sim.device)
#     audio_collapse = audio_sim[audio_mask].pow(2).mean()

#     text_mean = text_normed.mean(dim=0) 
#     text_sim = text_mean @ text_mean.T
#     text_mask = ~torch.eye(num_classes, dtype=bool, device=text_sim.device)
#     text_collapse = text_sim[text_mask].pow(2).mean()
# #attract_loss + 5 * repel_loss + margin_loss +
#     total_loss = collapse_coef * (2.5 * audio_collapse + text_collapse)

#     gap = (pos_sim - neg_sim_hard).mean().item()
#     print(f"pos: {pos_sim.mean().item():.4f}, neg: {neg_sim_hard.mean().item():.4f}, gap: {gap:.4f}, a_col: {audio_sim[audio_mask].mean().item():.4f}, t_col: {text_sim[text_mask].mean().item():.4f}")

#     return total_loss

def sequence_contrastive_loss(embeddings, mask):
    # embeddings shape: (B, L, D)
    # mask shape: (B, L)
    B, L, D = embeddings.shape

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=-1)

    # Compute similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.transpose(1, 2)) #/ self.temperature
    
    # Create labels for cross entropy (diagonal indices)
    labels = torch.arange(L, device=embeddings.device).unsqueeze(0).expand(B, -1)
    
    # Compute loss for each element in the batch
    loss = F.cross_entropy(sim_matrix.reshape(B*L, L), labels.reshape(-1), reduction='none')
    
    # Apply mask to loss
    loss = loss.view(B, L) * mask

    # Compute mean loss over non-padded elements
    loss = loss.sum() / mask.sum()

    return loss


def focal_loss_with_logits(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "sum",
        label_smoothing: float = 0.0,
        ignore_index: int = -100  # default value for ignored index
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
        label_smoothing (float): Specifies the amount of smoothing when computing the loss, 
                                                                where 0.0 means no smoothing.
        ignore_index (int): Specifies a target value that is ignored and does not contribute
                            to the input gradient. Default: ``-100``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Create a mask to ignore specified index
    valid_mask = targets != ignore_index
    
    # Apply label smoothing if needed
    if label_smoothing != 0:
        with torch.no_grad():
            targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing

    # Apply sigmoid activation to inputs
    p = torch.sigmoid(inputs)

    # Compute the binary cross-entropy loss without reduction
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # Apply the valid mask to the loss
    loss = loss * valid_mask

    # Apply focal loss modulation if gamma is greater than 0
    if gamma > 0:
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = loss * ((1 - p_t) ** gamma)

    # Apply alpha weighting if alpha is specified
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Apply reduction method
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.sum() / valid_mask.sum()  # Normalize by the number of valid (non-ignored) elements
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(
            f"Invalid value for argument 'reduction': '{reduction}'. "
            f"Supported reduction modes: 'none', 'mean', 'sum'"
        )