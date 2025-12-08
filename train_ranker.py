import os
import warnings
import sys
import torch

# Suppress ALL warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
sys.stderr = open(os.devnull, 'w')  # Suppress stderr warnings

# Environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_CONSOLE'] = 'off'
os.environ['WANDB_MODE'] = 'disabled'  # Completely disable wandb
os.environ['WANDB_DISABLED'] = 'true'

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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pytorch_lightning import seed_everything
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)


try:
    os.environ['WANDB_PROJECT'] = PROJECT_NAME
except:
    print('WANDB_PROJECT not available, please set it in config.py')


def main(args, export_root=None):
    seed_everything(args.seed)
    if export_root == None:
        export_root = EXPERIMENT_ROOT + '/' + args.llm_base_model.split('/')[-1] + '/' + args.dataset_code

    train_loader, val_loader, test_loader, tokenizer, test_retrieval = dataloader_factory(args)
    
    # Modern HuggingFace approach with BitsAndBytes 8-bit quantization
    # 8-bit supports DDP (Data Parallelism) better than 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Changed from 4bit to 8bit for multi-GPU support
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    # Multi-GPU support
    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if local_rank in [-1, 0]:
        print(f"🚀 Using {num_gpus} GPU(s) for training")
    
    # Device mapping for each process
    if local_rank != -1:
        # DDP mode: Each process gets full model on its own GPU
        device_map = {'': local_rank}
        max_memory_mapping = {local_rank: "14GB"}  # 8-bit uses less memory than float32
    else:
        # Single GPU mode
        device_map = 'auto'
        max_memory_mapping = {0: "14GB"}
    
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_base_model,
        quantization_config=bnb_config,  # 4-bit config
        device_map=device_map,
        cache_dir=args.llm_cache_dir,
        trust_remote_code=True,
        # ❌ DON'T set torch_dtype - conflicts with quantization_config
        low_cpu_mem_usage=True,
        max_memory=max_memory_mapping,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_base_tokenizer if hasattr(args, 'llm_base_tokenizer') else args.llm_base_model,
        cache_dir=args.llm_cache_dir,
        trust_remote_code=True,
    )
    
    # Prepare model for k-bit training with gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # Verify 8-bit quantization
    if local_rank in [-1, 0]:  # Only print on main process
        print("\n" + "="*50)
        print("🔍 MODEL QUANTIZATION CHECK (8-bit)")
        print("="*50)
        
        # Check if model is quantized (8-bit has different attributes)
        is_quantized = False
        for name, param in model.named_parameters():
            if hasattr(param, 'CB') or hasattr(param, 'SCB'):  # 8-bit quantization attributes
                is_quantized = True
                print(f"✅ 8-bit Quantized Layer: {name}")
                if hasattr(param, 'CB'):
                    print(f"   CB attribute present (8-bit)")
                break
            elif hasattr(param, 'quant_state'):  # 4-bit would have this
                is_quantized = True
                print(f"✅ 4-bit Quantized Layer: {name}")
                break
            elif 'embed' in name or 'lm_head' in name:
                print(f"Layer: {name}")
                print(f"   dtype: {param.dtype}")
                print(f"   device: {param.device}")
        
        if is_quantized:
            print("✅ Model is using quantization for multi-GPU DDP!")
        else:
            print("⚠️  Model is NOT quantized (using full precision)")
        
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
    args.model_code = 'llm'
    set_template(args)
    main(args, export_root=None)
