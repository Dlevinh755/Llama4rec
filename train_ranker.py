import os
import warnings
import sys
import torch
import traceback

# Suppress most warnings but keep errors visible
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

# Environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_CONSOLE'] = 'off'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_DISABLED'] = 'true'
os.environ['TORCHELASTIC_ERROR_FILE'] = '/tmp/error.json'  # Enable traceback

# Multi-GPU setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
from llama_datasets import DATASETS
from config import *
from model import *
from dataloader import *
from trainer import *
# from unsloth import FastLanguageModel  # Removed - using native HuggingFace

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from pytorch_lightning import seed_everything


try:
    os.environ['WANDB_PROJECT'] = PROJECT_NAME
except:
    print('WANDB_PROJECT not available, please set it in config.py')


def main(args, export_root=None):
    seed_everything(args.seed)
    if export_root == None:
        export_root = EXPERIMENT_ROOT + '/' + args.llm_base_model.split('/')[-1] + '/' + args.dataset_code

    train_loader, val_loader, test_loader, tokenizer, test_retrieval = dataloader_factory(args)
    
    # ✅ USE PURE BF16 MIXED PRECISION FOR TRUE DATA PARALLELISM
    # BitsAndBytes forces Model Parallelism (splits model across GPUs)
    # BF16 enables Data Parallelism (each GPU processes different batch)
    
    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if local_rank in [-1, 0]:
        print(f"🚀 Using {num_gpus} GPU(s) with FP16 Data Parallelism")
        print(f"   Mode: Each GPU processes DIFFERENT batches (2x speedup)")
    
    # Load model in FP16 for memory efficiency (Kaggle supports FP16, not BF16)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_base_model,
        torch_dtype=torch.float16,  # FP16 for Kaggle GPU compatibility
        cache_dir=args.llm_cache_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Move to GPU (DDP will handle distribution)
    if local_rank != -1:
        model = model.to(f'cuda:{local_rank}')
    else:
        model = model.to('cuda:0')
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_base_tokenizer if hasattr(args, 'llm_base_tokenizer') else args.llm_base_model,
        cache_dir=args.llm_cache_dir,
        trust_remote_code=True,
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Verify BF16 mixed precision
    if local_rank in [-1, 0]:  # Only print on main process
        print("\n" + "="*50)
        print("🔍 MODEL PRECISION CHECK")
        print("="*50)
        
        # Check model dtype
        for name, param in model.named_parameters():
            if 'embed' in name or 'lm_head' in name or 'q_proj' in name:
                print(f"Layer: {name}")
                print(f"   dtype: {param.dtype}")
                print(f"   device: {param.device}")
                if name.endswith('q_proj.weight'):
                    break
        
        print(f"\n✅ Model loaded in {next(model.parameters()).dtype}")
        print("✅ Using FP16 Mixed Precision for 2x speedup!")
        print("="*50 + "\n")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False  # Disable cache for training
    model.config.use_cache = False
    
    trainer = LLMTrainer(args, model, train_loader, val_loader, test_loader, tokenizer, export_root, args.use_wandb)
    
    trainer.train()
    trainer.test(test_retrieval)


if __name__ == "__main__":
    try:
        # Import args (already parsed in config.py)
        args.model_code = 'llm'
        
        # Set template to configure batch sizes based on GPU count
        set_template(args)
        
        # Debug: Print batch size configuration
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank in [-1, 0]:
            print(f"📊 Batch Configuration:")
            print(f"   lora_micro_batch_size: {args.lora_micro_batch_size}")
            print(f"   train_batch_size: {args.train_batch_size}")
        
        main(args, export_root=None)
    except Exception as e:
        print(f"\n❌ ERROR in train_ranker.py:")
        print(f"   {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise
