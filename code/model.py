import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import ModelOutput
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig

@dataclass
class ClassificationOutput(ModelOutput):
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class MoralEmotionVLClassifier(nn.Module):
    def __init__(self, model_id, num_labels=1, device="auto", label_names=None):
        super().__init__()

        self.device = device
        self.model_id = model_id

        # Bits and bytes config for model quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load base model (vision-to-text)
        self.base_model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            device_map="auto" if device == "auto" else {"": device},
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )

        self.config = self.base_model.config
        self.config.num_labels = num_labels
        self.gradient_checkpointing_enable = self.base_model.gradient_checkpointing_enable
        
        # Modify the final classification head (lm_head)
        original_lm_head = self.base_model.lm_head
        hidden_size = original_lm_head.in_features
        head_device = original_lm_head.weight.device
        head_dtype = original_lm_head.weight.dtype

        # Change to classification head for the number of labels required
        self.base_model.lm_head = nn.Linear(
            hidden_size, 
            num_labels,
            device=head_device,
            dtype=head_dtype
        )

        # label mapping
        self.num_labels = num_labels
        self.label_names = label_names if label_names is not None else []
        self.label2id = {label: i for i, label in enumerate(self.label_names)}
        self.id2label = {i: label for i, label in enumerate(self.label_names)}

    def forward(self, **kwargs):
        outputs = self.base_model(**kwargs)        
        logits = outputs.logits        
        classification_logits = logits[:, -1, :]
        
        return ClassificationOutput(
            logits=classification_logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        )