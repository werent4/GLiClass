import math
import torch.nn as nn

class CosineDropoutScheduler(nn.Module):
    def __init__(self, p_max=0.5, p_min=0.1, total_steps=1000, osc_freq=4, osc_strength=0.1, mode="+"):
        super().__init__()
        self.p_max = p_max
        self.p_min = p_min
        self.total_steps = total_steps
        self.osc_freq = osc_freq
        self.osc_strength = osc_strength
        self.step_num = 0
        self.dropout = nn.Dropout(p_max)
        self.mode = mode

    def step(self):
      t = min(self.step_num, self.total_steps)

      base = self.p_min + 0.5 * (self.p_max - self.p_min) * (1 + math.cos(math.pi * t / self.total_steps))

      decay = 0.5 * (1 + math.cos(math.pi * t / self.total_steps))

      oscillation = self.osc_strength * decay * math.cos(self.osc_freq * math.pi * t / self.total_steps)

      if self.mode == "+":
        p = base + oscillation
      elif self.mode == "-":
        p = base - oscillation 
      self.dropout.p = max(0.0, min(1.0, p))
      self.step_num += 1

      if self.step_num % 50 == 0:
        print(f"Dropout p: {p:.4f}")
        
    def forward(self, x):
        return self.dropout(x)

    def get_current_p(self):
        return self.dropout.p