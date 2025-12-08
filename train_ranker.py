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
    
    # Modern HuggingFace approach with BitsAndBytes 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for better stability
    )
    
    # Multi-GPU support
    num_gpus = torch.cuda.device_count()
    print(f"🚀 Using {num_gpus} GPU(s) for training")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # For multi-GPU, distribute model across GPUs
    if num_gpus > 1:
        max_memory_mapping = {i: "13GB" for i in range(num_gpus)}  # Equal split
    else:
        max_memory_mapping = {0: "13GB"}
    
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_base_model,
        quantization_config=bnb_config,
        device_map={'': local_rank},
        cache_dir=args.llm_cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
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
    
    # Verify 4-bit quantization
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank in [-1, 0]:  # Only print on main process
        print("\n" + "="*50)
        print("🔍 MODEL QUANTIZATION CHECK")
        print("="*50)
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                if 'lm_head' in name or 'embed' in name:  # Check key layers
                    print(f"Layer: {name}")
                    print(f"  dtype: {module.weight.dtype}")
                    print(f"  shape: {module.weight.shape}")
                    if hasattr(module.weight, 'quant_state'):
                        print(f"  ✅ Quantized: {module.weight.quant_state}")
                    break
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
