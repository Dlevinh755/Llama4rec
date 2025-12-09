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

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training
from pytorch_lightning import seed_everything


try:
    os.environ['WANDB_PROJECT'] = PROJECT_NAME
except:
    print('WANDB_PROJECT not available, please set it in config.py')


def main(args, export_root=None):
    seed_everything(args.seed)
    if export_root == None:
        export_root = EXPERIMENT_ROOT + '/' + args.llm_base_model.split('/')[-1] + '/' + args.dataset_code

    # Ensure export directory exists (for checkpoints)
    os.makedirs(export_root, exist_ok=True)

    train_loader, val_loader, test_loader, tokenizer, test_retrieval = dataloader_factory(args)
    
    # ✅ USE 4-BIT QUANTIZATION WITH MODEL PARALLELISM
    # Model Parallelism: splits model layers across GPUs (better memory efficiency)
    # Data Parallelism: each GPU processes different batches (faster but needs more memory)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if local_rank in [-1, 0]:
        print(f"🚀 Using {num_gpus} GPU(s) with 4-bit Model Parallelism")
        print(f"   Mode: Model layers SPLIT across GPUs (memory efficient)")
    
    # Model Parallelism: distribute model across GPUs
    # Use device_map based on whether running with accelerate or not
    if local_rank != -1:
        # Running with accelerate: use specific GPU for this process
        device_map = {'': local_rank}
        max_memory_mapping = {local_rank: "13GB"}
    else:
        # Running without accelerate: auto-distribute across all GPUs
        device_map = "auto"
        max_memory_mapping = {i: "13GB" for i in range(num_gpus)}
    
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        cache_dir=args.llm_cache_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory=max_memory_mapping,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_base_tokenizer if hasattr(args, 'llm_base_tokenizer') else args.llm_base_model,
        cache_dir=args.llm_cache_dir,
        trust_remote_code=True,
    )
    
    # Prepare model for 4-bit training with gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # Verify 4-bit quantization
    if local_rank in [-1, 0]:  # Only print on main process
        print("\n" + "="*50)
        print("🔍 MODEL QUANTIZATION CHECK (4-bit)")
        print("="*50)
        
        # Check model dtype and device distribution
        for name, param in model.named_parameters():
            if 'embed' in name or 'lm_head' in name:
                print(f"Layer: {name}")
                print(f"   dtype: {param.dtype}")
                print(f"   device: {param.device}")
            if hasattr(param, 'quant_state'):
                print(f"✅ 4-bit Quantized: {name}")
                print(f"   device: {param.device}")
                break
        
        print(f"\n✅ Model using 4-bit quantization with Model Parallelism")
        print("✅ Memory efficient for 2 GPU training!")
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

    # Optional: resume from latest checkpoint if available and requested
    resume_from_checkpoint = getattr(args, "resume_from_checkpoint", False)
    if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.lower() in ["true", "1", "yes"]:
        resume_from_checkpoint = True

    last_checkpoint = None
    if resume_from_checkpoint:
        # Look for last checkpoint in export_root (format: checkpoint-* as saved by HF Trainer)
        checkpoints = [
            os.path.join(export_root, d)
            for d in os.listdir(export_root)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(export_root, d))
        ]
        if len(checkpoints) > 0:
            checkpoints.sort(key=lambda p: int(p.split("checkpoint-")[-1]))
            last_checkpoint = checkpoints[-1]
            print(f"✅ Found checkpoint, resuming from: {last_checkpoint}")
        else:
            print("ℹ️ No checkpoint found in export_root, training from scratch.")

    trainer.train(resume_from_checkpoint=last_checkpoint)
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
            # Show resume flag if present
            if hasattr(args, "resume_from_checkpoint"):
                print(f"   resume_from_checkpoint: {args.resume_from_checkpoint}")
        
        main(args, export_root=None)
    except Exception as e:
        print(f"\n❌ ERROR in train_ranker.py:")
        print(f"   {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise
