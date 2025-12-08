from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from .verb import ManualVerbalizer
from .utils import *
from .loggers import *
from .base import *
from dataloader.collate_fns import llama_collate_fn_w_truncation

import re
import re
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import numpy as np
from abc import *
from pathlib import Path

import bitsandbytes as bnb
from transformers.trainer import *
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


def compute_metrics_for_ks(ks, verbalizer):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = torch.tensor(logits)
        labels = torch.tensor(labels).view(-1)
        scores = verbalizer.process_logits(logits)
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels, ks)
        return metrics
    return compute_metrics


class LLMTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            train_loader,
            val_loader,
            test_loader,
            tokenizer,
            export_root,
            use_wandb,
            **kwargs
        ):
        self.original_args = args
        self.export_root = export_root
        self.use_wandb = use_wandb
        self.llm_max_text_len = args.llm_max_text_len
        self.rerank_metric_ks = args.rerank_metric_ks
        self.verbalizer = ManualVerbalizer(
            tokenizer=tokenizer,
            prefix='',
            post_log_softmax=False,
            classes=list(range(args.llm_negative_sample_size+1)),
            label_words={i: chr(ord('A')+i) for i in range(args.llm_negative_sample_size+1)},
        )

        eval_strategy = args.lora_eval_strategy if hasattr(args, 'lora_eval_strategy') else "no"
        
        # Detect number of GPUs
        num_gpus = torch.cuda.device_count()
        
        # Ensure batch size is valid
        micro_batch_size = getattr(args, 'lora_micro_batch_size', None)
        if micro_batch_size is None or micro_batch_size <= 0:
            micro_batch_size = 4  # Safe default
            print(f"⚠️  Warning: lora_micro_batch_size was {args.lora_micro_batch_size}, using default {micro_batch_size}")
        
        # Calculate gradient accumulation steps safely
        total_batch = getattr(args, 'train_batch_size', 16)
        grad_accum_steps = max(1, total_batch // micro_batch_size // max(1, num_gpus))
        
        hf_args = TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=grad_accum_steps,  # Safe calculation
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.lora_num_epochs,
            learning_rate=args.lora_lr,
            fp16=True,  # Use FP16 mixed precision
            logging_steps=10,
            optim="paged_adamw_8bit",  # 8-bit optimizer for memory efficiency
            eval_strategy=eval_strategy,
            save_strategy="steps",
            eval_steps=args.lora_val_iterations,
            save_steps=args.lora_val_iterations,
            output_dir=export_root,
            save_total_limit=3,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if num_gpus > 1 else None,  # Enable DDP for multi-GPU
            group_by_length=False,
            report_to="none",  # Completely disable wandb
            run_name=None,
            metric_for_best_model=args.rerank_best_metric,
            greater_is_better=True,
            # Multi-GPU optimization
            dataloader_num_workers=args.num_workers if num_gpus > 1 else 0,
            dataloader_pin_memory=True if num_gpus > 1 else False,
            gradient_checkpointing=True,  # Save memory
        )
        
        callbacks = []
        if eval_strategy != "no":
            callbacks.append(EarlyStoppingCallback(args.lora_early_stopping_patience))
        
        super().__init__(
            model=model,
            args=hf_args,
            train_dataset=None,
            eval_dataset=None,
            data_collator=None,
            callbacks=callbacks,
            **kwargs)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.processing_class = tokenizer  # Use new attribute name instead of deprecated tokenizer
        self.tokenizer = tokenizer  # Keep for backward compatibility
        
        self.train_loader.collate_fn = llama_collate_fn_w_truncation(self.llm_max_text_len, eval=False)
        self.val_loader.collate_fn = llama_collate_fn_w_truncation(self.llm_max_text_len, eval=True)
        self.test_loader.collate_fn = llama_collate_fn_w_truncation(self.llm_max_text_len, eval=True)
        self.compute_metrics = compute_metrics_for_ks(self.rerank_metric_ks, self.verbalizer)

        if len(self.label_names) == 0:
            self.label_names = ['labels']  # for some reason label name is not set
    
    def test(self, test_retrieval):
        average_metrics = self.predict(test_dataset=None).metrics
        print('Ranking Performance on Subset:', average_metrics)
        print('************************************************************')
        with open(os.path.join(self.export_root, 'subset_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)

        print('Original Performance:', test_retrieval['original_metrics'])
        print('************************************************************')
        original_size = test_retrieval['original_size']
        retrieval_size = test_retrieval['retrieval_size']
        
        overall_metrics = {}
        for key in test_retrieval['non_retrieval_metrics'].keys():
            if 'test_' + key in average_metrics:
                overall_metrics['test_' + key] = (average_metrics['test_' + key] * retrieval_size  + \
                    test_retrieval['non_retrieval_metrics'][key] * (original_size - retrieval_size)) / original_size
        print('Overall Performance of Our Framework:', overall_metrics)
        with open(os.path.join(self.export_root, 'overall_metrics.json'), 'w') as f:
                json.dump(overall_metrics, f, indent=4)
        
        return average_metrics

    def get_train_dataloader(self):
        return self.train_loader
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        return self.val_loader
    
    def get_test_dataloader(self, test_dataset: Optional[Dataset] = None) -> DataLoader:
        return self.test_loader
    
    def training_step(self, *args, **kwargs):
        # Call the original training_step method
        loss = super().training_step(*args, **kwargs)

        # Add custom logging with batch size and GPU info
        if self.state.global_step % 10 == 0:
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            gpu_id = local_rank if local_rank != -1 else torch.cuda.current_device()
            
            # Get batch size from first arg (model inputs)
            batch_size = 0
            if len(args) > 1 and isinstance(args[1], dict):
                inputs = args[1]
                if 'input_ids' in inputs:
                    batch_size = inputs['input_ids'].size(0)
            
            # Calculate per-GPU loss BEFORE synchronization
            local_loss = loss.item() if not isinstance(loss, float) else loss
            
            # Show BOTH local loss and global loss
            print(f"[GPU {gpu_id}] Step {self.state.global_step}, Batch: {batch_size}, Local Loss: {local_loss:.4f}")

        return loss