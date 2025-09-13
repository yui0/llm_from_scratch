# 924deb3c9886ee2605ef78894dc8b1baba62d32a
import argparse
import os
import glob
import wandb
import torch
import transformers
from datetime import datetime
from itertools import chain
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer
)

# ========================
# 引数設定（Colab対応）
# ========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (overrides epochs if >0)")
    parser.add_argument("--train_batch", type=int, default=8, help="Train batch size per device")
    parser.add_argument("--eval_batch", type=int, default=8, help="Eval batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to resume checkpoint (auto if not set)")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--streaming", action="store_true", help="Enable dataset streaming")
    parser.add_argument("--output_dir", type=str, default="gpt-2-warm-up/standard-gpt", help="Output directory for checkpoints")
    try:
        args = parser.parse_args()
    except SystemExit:  # Colabなどで引数が空のとき
        args = parser.parse_args([])
    return args

args = parse_args()

# ========================
# 最新チェックポイント自動検出
# ========================
def find_latest_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]

if args.resume_from_checkpoint is None:
    latest_ckpt = find_latest_checkpoint(args.output_dir)
    if latest_ckpt:
        print(f"[INFO] Latest checkpoint found: {latest_ckpt}")
        args.resume_from_checkpoint = latest_ckpt
    else:
        print("[INFO] No checkpoint found. Starting from scratch.")

# ========================
# wandb設定（run ID 管理）
# ========================
wandb_id_file = os.path.join(args.output_dir, "wandb_run_id.txt")
if args.resume_from_checkpoint and os.path.exists(wandb_id_file):
    with open(wandb_id_file, "r") as f:
        wandb_run_id = f.read().strip()
    print(f"[INFO] Resuming wandb run with ID: {wandb_run_id}")
else:
    wandb_run_id = wandb.util.generate_id()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(wandb_id_file, "w") as f:
        f.write(wandb_run_id)
    print(f"[INFO] Starting new wandb run with ID: {wandb_run_id}")

# ========================
# run名自動生成
# ========================
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_label = f"steps{args.max_steps}" if args.max_steps > 0 else f"epochs{args.epochs}"
run_name = f"gpt2-{timestamp}-{run_label}-bs{args.train_batch}-lr{args.learning_rate}"

# ========================
# GPU & モデル情報取得
# ========================
gpu_info = {}
if torch.cuda.is_available():
    gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
    gpu_info["gpu_memory_GB"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
else:
    gpu_info["gpu_name"] = "CPU"
    gpu_info["gpu_memory_GB"] = 0

temp_config = GPT2Config()
temp_model = GPT2LMHeadModel(temp_config)
model_size = sum(p.numel() for p in temp_model.parameters())
del temp_model

# ========================
# wandb初期化 + config記録
# ========================
wandb_config = vars(args).copy()
wandb_config.update({
    "model_size_params": model_size,
    **gpu_info
})
wandb.init(
    project="gpt2-training",
    id=wandb_run_id,
    resume="allow",
    name=run_name,
    config=wandb_config
)

# ========================
# データ読み込み
# ========================
if args.streaming:
    dataset = load_dataset(
        "bookcorpus/bookcorpus",
        split="train",
        streaming=True,
        trust_remote_code=True,
        cache_dir="/cache"
    )
else:
    dataset = load_dataset(
        "bookcorpus/bookcorpus",
        split="train",
        streaming=False,
        trust_remote_code=True,
        cache_dir="/cache"
    )
    dataset = dataset.select(range(3200))
    dataset = dataset.train_test_split(test_size=0.0015)

# ========================
# Tokenizer設定
# ========================
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    return tokenizer(text=example["text"])

if not args.streaming:
    tokenized_ds = dataset.map(tokenize_function, batched=True, remove_columns="text")
    def concat(examples):
        examples["input_ids"] = [list(chain.from_iterable(examples['input_ids']))]
        examples["attention_mask"] = [list(chain.from_iterable(examples['attention_mask']))]
        return examples
    concated_ds = tokenized_ds.map(concat, batched=True, batch_size=1000000, num_proc=8)
    def chunk(examples):
        chunk_size = 1024
        input_ids = examples["input_ids"][0]
        attention_mask = examples["attention_mask"][0]
        ids_trunc, attn_trunc = [], []
        for i in range(0, len(input_ids), chunk_size):
            c = input_ids[i:i+chunk_size]
            if len(c) == chunk_size:
                ids_trunc.append(c)
                attn_trunc.append(attention_mask[i:i+chunk_size])
        examples['input_ids'] = ids_trunc
        examples['attention_mask'] = attn_trunc
        return examples
    chunked_ds = concated_ds.map(chunk, batched=True, batch_size=2, num_proc=2)
else:
    chunked_ds = dataset

# ========================
# Data collator
# ========================
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ========================
# モデル読み込み
# ========================
if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
    print(f"[INFO] Resuming from checkpoint: {args.resume_from_checkpoint}")
    model = GPT2LMHeadModel.from_pretrained(args.resume_from_checkpoint)
else:
    print("[INFO] Initializing new model from scratch")
    configuration = GPT2Config()
    model = GPT2LMHeadModel(configuration)

# ========================
# TrainingArguments（バージョン互換）
# ========================
from transformers import TrainingArguments
ta_params = {}
if "evaluation_strategy" in transformers.TrainingArguments.__init__.__code__.co_varnames:
    ta_params["evaluation_strategy"] = "steps"
else:
    ta_params["do_eval"] = True

if "logging_strategy" in transformers.TrainingArguments.__init__.__code__.co_varnames:
    ta_params["logging_strategy"] = "steps"

training_args = TrainingArguments(
    output_dir=args.output_dir,
    **ta_params,
    eval_steps=args.eval_steps,
    num_train_epochs=args.epochs,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.train_batch,
    per_device_eval_batch_size=args.eval_batch,
    learning_rate=args.learning_rate,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    adam_beta1=0.9,
    adam_beta2=0.999,
    weight_decay=0.01,
    logging_steps=500,
    save_steps=args.save_steps,
    save_total_limit=10,
    fp16=args.fp16,
    dataloader_num_workers=4,
    report_to='wandb',
    run_name=run_name,
)

# ========================
# Trainer作成 & 学習開始
# ========================
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=None if args.streaming else chunked_ds["train"],
    eval_dataset=None if args.streaming else chunked_ds["test"],
    data_collator=data_collator
)
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
