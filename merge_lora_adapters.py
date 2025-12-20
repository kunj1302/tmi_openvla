#!/usr/bin/env python3
"""
Script to merge LoRA adapter weights with base model.

Usage:
    # Using HuggingFace Hub model:
    python merge_lora_adapters.py \
        --adapter_path /scratch1/kgolwala/openvla_adapter_tmp/<exp_id> \
        --base_model_path openvla/openvla-7b-finetuned-libero-object \
        --output_path /scratch1/kgolwala/openvla_runs/<exp_id>
    
    # Using local cache:
    python merge_lora_adapters.py \
        --adapter_path /scratch1/kgolwala/openvla_adapter_tmp/<exp_id> \
        --base_model_path openvla/openvla-7b \
        --cache_dir /scratch1/kgolwala/hf_cache/models--openvla--openvla-7b \
        --output_path /scratch1/kgolwala/openvla_runs/<exp_id>
"""

import argparse
import os
import glob
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from transformers import AutoConfig, AutoImageProcessor


def find_cache_snapshot(cache_dir: str):
    """Find the snapshot directory in HuggingFace cache."""
    snapshots_dir = os.path.join(cache_dir, "snapshots")
    if not os.path.exists(snapshots_dir):
        return None
    
    # Find all snapshot directories
    snapshots = glob.glob(os.path.join(snapshots_dir, "*"))
    snapshots = [s for s in snapshots if os.path.isdir(s)]
    
    if len(snapshots) == 0:
        return None
    elif len(snapshots) == 1:
        return snapshots[0]
    else:
        # If multiple snapshots, return the most recent one
        return max(snapshots, key=os.path.getmtime)


def merge_adapters(adapter_path: str, base_model_path: str, output_path: str, cache_dir: str = None):
    """Merge LoRA adapters with base model and save."""
    
    print(f"[*] Merging LoRA adapters from: {adapter_path}")
    print(f"[*] Base model: {base_model_path}")
    print(f"[*] Output path: {output_path}")
    
    # Register OpenVLA model to HF Auto Classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Handle cache directory - if cache_dir is provided, try to find snapshot
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    
    if cache_dir:
        snapshot_path = find_cache_snapshot(cache_dir)
        if snapshot_path:
            print(f"[*] Using cached model from: {snapshot_path}")
            base_model_path = snapshot_path
        else:
            # Set cache_dir for HuggingFace to use
            load_kwargs["cache_dir"] = cache_dir
            print(f"[*] Using cache directory: {cache_dir}")
    
    # Load base model
    print("\n[*] Loading base model...")
    base_vla = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        **load_kwargs
    )
    print("[+] Base model loaded")
    
    # Load LoRA adapters
    print("\n[*] Loading LoRA adapters...")
    merged_vla = PeftModel.from_pretrained(base_vla, adapter_path)
    print("[+] LoRA adapters loaded")
    
    # Merge adapters into base model
    print("\n[*] Merging adapters into base model...")
    merged_vla = merged_vla.merge_and_unload()
    print("[+] Merging complete")
    
    # Save merged model
    print(f"\n[*] Saving merged model to {output_path}...")
    merged_vla.save_pretrained(output_path)
    print("[+] Merged model saved")
    
    # Also copy processor if it exists in adapter directory
    processor_path = os.path.join(os.path.dirname(adapter_path), "..", os.path.basename(output_path))
    # Actually, processor should be in run_dir, let's check adapter_path's parent
    run_dir = os.path.dirname(adapter_path).replace("openvla_adapter_tmp", "openvla_runs")
    processor_src = os.path.join(run_dir, os.path.basename(adapter_path))
    
    if os.path.exists(processor_src):
        print(f"\n[*] Copying processor from {processor_src}...")
        processor = AutoProcessor.from_pretrained(processor_src, trust_remote_code=True)
        processor.save_pretrained(output_path)
        print("[+] Processor copied")
    else:
        print(f"\n[!] Processor not found at {processor_src}, loading from base model...")
        processor_kwargs = {"trust_remote_code": True}
        if cache_dir and not os.path.isdir(base_model_path):
            processor_kwargs["cache_dir"] = cache_dir
        processor = AutoProcessor.from_pretrained(base_model_path, **processor_kwargs)
        processor.save_pretrained(output_path)
        print("[+] Processor saved")
    
    print(f"\n[+] Merge complete! Merged model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with base model")
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapter weights directory"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="openvla/openvla-7b-finetuned-libero-object",
        help="Path to base model (HuggingFace Hub path or local path)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save merged model"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to HuggingFace cache directory (e.g., /scratch1/kgolwala/hf_cache/models--openvla--openvla-7b)"
    )
    
    args = parser.parse_args()
    
    merge_adapters(
        adapter_path=args.adapter_path,
        base_model_path=args.base_model_path,
        output_path=args.output_path,
        cache_dir=args.cache_dir
    )
