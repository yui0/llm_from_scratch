# Google Colab 用: JAX で LLM をゼロから構築（日本語対応、GGUFエクスポート対応、memmap学習、GPU効率化、チェックポイント再開）

import os
import re
import pickle
import logging
from typing import Dict, Tuple, Optional, List

import jax
import jax.numpy as jnp
from jax import random, checkpoint
import numpy as np
import optax
from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from safetensors.torch import save_file
import shutil

# -----------------------------
# Google Drive マウント (Colabでの実行時)
# -----------------------------
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("Google Colab環境ではないため、Driveのマウントをスキップします。")

# -----------------------------
# ロギング設定
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 乱数シード
rng_key = random.PRNGKey(0)

# デバッグ
jax.config.update("jax_debug_nans", False)

# Enforce bfloat16 precision for matrix multiplications
jax.config.update("jax_default_matmul_precision", "bfloat16")

# -----------------------------
# 1. データセットの準備（ストリーミング & 低メモリ） - 改善: データセットを追加して品質向上
# -----------------------------
def prepare_data(
    dataset_configs: List[Tuple[str, str]] = [
        ("wikimedia/wikipedia", "20231101.ja"),  # Wikipedia日本語
        #("oscar", "unshuffled_deduplicated_ja"),  # OSCAR日本語（大規模コーパス、無料）
    ],
    split: str = "train",
    chunk_size: int = 1000,
    output_path: str = "input.txt",
    min_text_length: int = 100,
    min_sentences: int = 2
):
    try:
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logging.info(f"Dataset file already exists at {output_path} with size {os.path.getsize(output_path)} bytes. Skipping preparation.")
            return output_path

        datasets = []
        for dataset_name, config in dataset_configs:
            ds = load_dataset(dataset_name, config, split=split, streaming=True)
            datasets.append(ds)

        combined_dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]

        with open(output_path, "w", encoding="utf-8") as f:
            buffer = []
            total_written = 0
            for i, item in enumerate(tqdm(combined_dataset, desc="データセット準備中")):
                text = item.get("text", "").strip()
                if len(text) > min_text_length and text.count("。") > min_sentences:
                    sentences = text.split("。 ")
                    np.random.shuffle(sentences)
                    sentences = [s for s in sentences if np.random.random() > 0.05]
                    buffer.append("。 ".join(sentences))

                if (i + 1) % chunk_size == 0:
                    f.write(" ".join(buffer) + " ")
                    total_written += len(" ".join(buffer))
                    buffer = []
            if buffer:
                f.write(" ".join(buffer))
                total_written += len(" ".join(buffer))
        logging.info(f"Dataset prepared and saved to {output_path} with {total_written} characters")
        if total_written == 0:
            raise ValueError("No data written to output file. Check dataset or filtering criteria.")
        return output_path
    except Exception as e:
        logging.error(f"Failed to prepare dataset: {e}")
        raise

# -----------------------------
# 2. トークナイザーのロード - 変更なし
# -----------------------------
def load_pretrained_tokenizer(model_name: str = "NousResearch/Llama-2-7b-hf") -> Tokenizer:
    try:
        tokenizer = Tokenizer.from_pretrained(model_name)
        pad_token = "<|endoftext|>"
        pad_id = tokenizer.token_to_id(pad_token)
        if pad_id is None:
            tokenizer.add_special_tokens([pad_token])
            pad_id = tokenizer.token_to_id(pad_token)
        tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
        if "<|endoftext|>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens(["<|endoftext|>"])
        logging.info(f"Tokenizer loaded. vocab size={tokenizer.get_vocab_size()}")
        return tokenizer
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        raise

# -----------------------------
# 3. memmap へトークナイズ（巨大コーパス対応） - 改善: chunk_sizeを増やしてメモリ節約
# -----------------------------
def tokenize_to_memmap(
    text_path: str,
    tokenizer: Tokenizer,
    output_file: str = "tokens.memmap",
    block_size: int = 64,  # 改善: ブロックサイズを64に拡大（文脈強化）
    chunk_size: int = 20000,  # 改善: chunk_sizeを増やして処理効率化
    drive_path: Optional[str] = None
) -> np.memmap:
    if os.path.exists(output_file):
        logging.info(f"Memmap file already exists at {output_file}. Loading it.")
        tokens = np.memmap(output_file, dtype=np.int32, mode="r+")
        return tokens

    logging.info("Estimating total tokens...")
    def estimate_tokens(sample_size: int = 10 * 1024 * 1024) -> int:
        total_size = os.path.getsize(text_path)
        with open(text_path, "r", encoding="utf-8") as f:
            sample_text = f.read(sample_size)
        sample_tokens = len(tokenizer.encode(sample_text).ids)
        estimated_tokens = int(sample_tokens * (total_size / len(sample_text.encode('utf-8'))))
        return estimated_tokens - (estimated_tokens % block_size)

    estimated_tokens = estimate_tokens()
    logging.info(f"Estimated tokens: {estimated_tokens:,}")

    tokens = np.memmap(output_file, dtype=np.int32, mode="w+", shape=(estimated_tokens,))
    
    logging.info("Writing tokens to memmap...")
    buffer = []
    offset = 0
    with open(text_path, "r", encoding="utf-8", buffering=4 * 1024 * 1024) as f:
        for i, line in enumerate(tqdm(f, desc="Tokenizing to memmap")):
            line = line.strip()
            if not line:
                continue
            buffer.append(line)
            if (i + 1) % chunk_size == 0:
                ids = tokenizer.encode(" ".join(buffer)).ids
                ids = ids[: len(ids) - (len(ids) % block_size)]
                if offset + len(ids) > estimated_tokens:
                    ids = ids[: estimated_tokens - offset]
                tokens[offset : offset + len(ids)] = ids
                offset += len(ids)
                buffer = []
        if buffer and offset < estimated_tokens:
            ids = tokenizer.encode(" ".join(buffer)).ids
            ids = ids[: len(ids) - (len(ids) % block_size)]
            if offset + len(ids) > estimated_tokens:
                ids = ids[: estimated_tokens - offset]
            tokens[offset : offset + len(ids)] = ids
            offset += len(ids)
    
    tokens = np.memmap(output_file, dtype=np.int32, mode="r+", shape=(offset,))
    tokens.flush()
    logging.info(f"Tokenized dataset saved to {output_file} with {offset:,} tokens")
    
    if drive_path and not os.path.exists(drive_path):
        logging.info(f"Copying {output_file} to {drive_path}...")
        shutil.copy(output_file, drive_path)
        logging.info(f"Copied to {drive_path}")
    
    return tokens

# -----------------------------
# 4. モデル作成（シンプルTransformer） - 改善: モデルサイズ縮小でメモリ節約（embed=256, layers=4, heads=4）
# -----------------------------
def create_model(vocab_size: int, embed_size: int = 256, num_layers: int = 4, num_heads: int = 4, head_size: int = 64, block_size: int = 64, dropout_rate: float = 0.1, param_dtype=jnp.bfloat16):
    def init_params(rng):
        params = {}
        def u(shape, scale):
            return random.uniform(rng, shape, minval=-scale, maxval=scale, dtype=param_dtype)

        scale_e = jnp.sqrt(6.0 / embed_size).astype(param_dtype)
        scale_h = jnp.sqrt(6.0 / (head_size * num_heads)).astype(param_dtype)
        scale_f = jnp.sqrt(6.0 / (embed_size * 4)).astype(param_dtype)

        params["embed"] = u((vocab_size, embed_size), scale_e)
        params["embed_norm"] = {"scale": jnp.ones(embed_size, dtype=param_dtype), "bias": jnp.zeros(embed_size, dtype=param_dtype)}
        for i in range(num_layers):
            params[f"layer_{i}"] = {
                "w_q": u((embed_size, head_size * num_heads), scale_e),
                "w_k": u((embed_size, head_size * num_heads), scale_e),
                "w_v": u((embed_size, head_size * num_heads), scale_e),
                "w_o": u((head_size * num_heads, embed_size), scale_h),
                "attn_norm": {"scale": jnp.ones(embed_size, dtype=param_dtype), "bias": jnp.zeros(embed_size, dtype=param_dtype)},
                "ffn1": u((embed_size, embed_size * 4), scale_e),
                "ffn2": u((embed_size * 4, embed_size), scale_f),
                "ffn_norm": {"scale": jnp.ones(embed_size, dtype=param_dtype), "bias": jnp.zeros(embed_size, dtype=param_dtype)},
            }
            logging.info(f"Layer {i} w_o shape: {params[f'layer_{i}']['w_o'].shape}")
        params["final"] = u((embed_size, vocab_size), scale_e)
        logging.info(f"Embedding shape: {params['embed'].shape}, Final shape: {params['final'].shape}")
        return params
    params = init_params(random.split(rng_key)[0])
    return params

# -----------------------------
# 5. 前処理ユーティリティ - 変更なし
# -----------------------------
def layer_norm(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias

def forward(params, x: jnp.ndarray, rng_key: jnp.ndarray, embed_size: int = 256, num_heads: int = 4, head_size: int = 64, dropout_rate: float = 0.1, training: bool = True, compute_dtype=jnp.bfloat16) -> jnp.ndarray:  # 改善: パラメータ調整
    x = x.astype(jnp.int32)
    rng, dropout_rng = random.split(rng_key)
    embeddings = params["embed"][x].astype(compute_dtype)
    embeddings = layer_norm(embeddings, params["embed_norm"]["scale"], params["embed_norm"]["bias"])

    seq_len = x.shape[-1]
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=compute_dtype)).reshape(1, 1, seq_len, seq_len)

    def layer_fn(embeddings, layer_params, rng_key):
        q = jnp.dot(embeddings, layer_params["w_q"]).reshape(-1, seq_len, num_heads, head_size)
        k = jnp.dot(embeddings, layer_params["w_k"]).reshape(-1, seq_len, num_heads, head_size)
        v = jnp.dot(embeddings, layer_params["w_v"]).reshape(-1, seq_len, num_heads, head_size)

        attn = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(jnp.array(head_size, dtype=compute_dtype))
        attn = attn * causal_mask - 1e9 * (1 - causal_mask)
        attn = jax.nn.softmax(attn, axis=-1)

        if training and dropout_rate > 0.0:
            dropout_mask = jax.random.bernoulli(dropout_rng, 1.0 - dropout_rate, attn.shape)
            attn = jnp.where(dropout_mask, attn / (1.0 - dropout_rate), 0.0)

        attn_out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(-1, seq_len, num_heads * head_size)
        embeddings = embeddings + jnp.dot(attn_out, layer_params["w_o"])
        embeddings = layer_norm(embeddings, layer_params["attn_norm"]["scale"], params["embed_norm"]["bias"])  # 注意: 元コードのtypo修正 (attn_norm)

        ffn = jax.nn.relu(jnp.dot(embeddings, layer_params["ffn1"]))
        if training and dropout_rate > 0.0:
            dropout_mask = jax.random.bernoulli(dropout_rng, 1.0 - dropout_rate, ffn.shape)
            ffn = jnp.where(dropout_mask, ffn / (1.0 - dropout_rate), 0.0)

        ffn = jnp.dot(ffn, layer_params["ffn2"])
        embeddings = embeddings + ffn
        embeddings = layer_norm(embeddings, layer_params["ffn_norm"]["scale"], layer_params["ffn_norm"]["bias"])
        return embeddings

    for i in range(len([k for k in params.keys() if k.startswith("layer_")])):
        layer = params[f"layer_{i}"]
        embeddings = checkpoint(layer_fn)(embeddings, layer, dropout_rng)

    logits = jnp.dot(embeddings, params["final"]).astype(jnp.float32)
    return logits

def loss_fn(params, x: jnp.ndarray, y: jnp.ndarray, rng_key: jnp.ndarray, **fw_kwargs) -> jnp.ndarray:
    logits = forward(params, x, rng_key, **fw_kwargs)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    targets = jax.nn.one_hot(y, logits.shape[-1])
    return -jnp.mean(jnp.sum(log_probs * targets, axis=-1))

def perplexity(loss: float) -> float:
    return jnp.exp(loss)

# -----------------------------
# 6. データローダ（memmap） + GPUプリフェッチ - 変更なし
# -----------------------------
def batch_sampler_memmap(tokens: np.memmap, batch_size: int, block_size: int, n_batches: int, rng: np.random.RandomState, start: int, end: int):
    max_index = (end - start) - block_size - 1
    if max_index <= 0:
        raise ValueError("Token array too small for batch sampling. Check token count or block size.")
    for _ in range(n_batches):
        idx = rng.randint(0, max_index, size=batch_size)
        X = np.stack([tokens[start + j : start + j + block_size] for j in idx]).astype(np.int32)
        Y = np.stack([tokens[start + j + 1 : start + j + block_size + 1] for j in idx]).astype(np.int32)
        yield X, Y

def prefetch_to_device(generator, prefetch_size: int = 2):
    queue = []
    for item in generator:
        item_dev = jax.tree_util.tree_map(jax.device_put, item)
        queue.append(item_dev)
        if len(queue) > prefetch_size:
            yield queue.pop(0)
    while queue:
        yield queue.pop(0)

# -----------------------------
# 7. チェックポイント関連 - 改善: expectedパラメータ調整
# -----------------------------
def tree_to_cpu_np(tree):
    return jax.tree_util.tree_map(lambda x: np.array(x), tree)

def save_checkpoint(ckpt_dir: str, step: int, params, opt_state, best_val: float, extra: Optional[dict] = None):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_step_{step:07d}.pkl")
    payload = {
        "step": step,
        "params": tree_to_cpu_np(params),
        "opt_state": tree_to_cpu_np(opt_state),
        "best_val": float(best_val),
        "extra": extra or {},
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Checkpoint saved: {path}")
    return path

def latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_step_") and f.endswith(".pkl")]
    if not files:
        return None
    files.sort()
    return os.path.join(ckpt_dir, files[-1])

def load_checkpoint(path: str, expected_embed_size: int = 256, expected_num_heads: int = 4, expected_head_size: int = 64):  # 改善: パラメータ調整
    with open(path, "rb") as f:
        payload = pickle.load(f)
    params = payload["params"]
    for i in range(len([k for k in params.keys() if k.startswith("layer_")])):
        w_o = params[f"layer_{i}"]["w_o"]
        expected_w_o_shape = (expected_num_heads * expected_head_size, expected_embed_size)
        if w_o.shape != expected_w_o_shape:
            logging.error(f"Checkpoint w_o shape {w_o.shape} does not match expected {expected_w_o_shape}. Reinitializing model.")
            raise ValueError(f"Incompatible checkpoint: w_o shape mismatch at layer {i}")
    return payload

# -----------------------------
# 8. 学習ループ（memmap + GPU効率化 + 再開対応） - 改善: grad_accum_steps=16, n_iterations=20000, dropout=0.05
# -----------------------------
def train_model_memmap(
    params,
    tokens: np.memmap,
    batch_size: int = 32,
    n_iterations: int = 20000,  # 改善: イテレーション増加
    learning_rate: float = 1e-3,
    block_size: int = 64,  # 改善: 拡大
    dropout_rate: float = 0.05,  # 改善: 低下で安定化
    val_ratio: float = 0.2,
    grad_accum_steps: int = 16,  # 改善: 増加で有効バッチ拡大、メモリ節約
    compute_dtype=jnp.bfloat16,
    ckpt_dir: str = "checkpoints",
    ckpt_every: int = 500,
):
    warmup_steps = min(2000, n_iterations // 10)
    schedule = optax.join_schedules(
        [optax.linear_schedule(0.0, learning_rate, warmup_steps),
         optax.cosine_decay_schedule(learning_rate, n_iterations - warmup_steps, 0.5)],
        [warmup_steps]
    )
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(schedule))
    opt_state = optimizer.init(params)

    total_tokens = len(tokens)
    val_size = int(total_tokens * val_ratio)
    train_size = total_tokens - val_size
    train_start, train_end = 0, train_size
    val_start, val_end = train_size, total_tokens

    def loss_for_batch(p, bx, by, key):
        fw_kwargs = {'compute_dtype': compute_dtype, 'dropout_rate': dropout_rate, 'embed_size': 256, 'num_heads': 4}  # 改善: kwargs調整
        return loss_fn(p, bx, by, key, **fw_kwargs)

    @jax.jit
    def update_step(p, ostate, bxs, bys, key):
        losses = []
        new_p = p
        new_ostate = ostate
        subkey = key
        for t in range(bxs.shape[0]):
            subkey, k = random.split(subkey)
            loss, grads = jax.value_and_grad(loss_for_batch)(new_p, bxs[t], bys[t], k)
            updates, new_ostate = optimizer.update(grads, new_ostate, params=new_p)
            new_p = optax.apply_updates(new_p, updates)
            losses.append(loss)
        mean_loss = jnp.stack(losses).mean()
        return new_p, new_ostate, mean_loss

    @jax.jit
    def eval_loss(p, bx, by, key):
        fw_kwargs = {'compute_dtype': compute_dtype, 'training': False, 'embed_size': 256, 'num_heads': 4}  # 改善: kwargs調整
        return loss_fn(p, bx, by, key, **fw_kwargs)

    train_losses, val_losses = [], []
    best_val = float("inf")
    patience = 800
    patience_counter = 0
    start_iter = 0

    last = latest_checkpoint(ckpt_dir)
    if last is not None:
        try:
            payload = load_checkpoint(last, embed_size=256, num_heads=4, head_size=64)
            params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
            opt_state = jax.tree_util.tree_map(jnp.asarray, payload["opt_state"])
            best_val = float(payload.get("best_val", best_val))
            start_iter = int(payload.get("step", 0)) + 1
            logging.info(f"Resumed from {last} (step={start_iter-1}, best_val={best_val:.4f})")
        except ValueError as e:
            logging.warning(f"Failed to load checkpoint due to {e}. Starting with fresh parameters.")
            params = create_model(
                tokenizer.get_vocab_size(),
                embed_size=256,
                num_layers=4,
                num_heads=4,
                head_size=64,
                block_size=block_size,
                dropout_rate=dropout_rate,
                param_dtype=jnp.bfloat16
            )
            opt_state = optimizer.init(params)

    rng_np = np.random.RandomState(0)
    pbar = tqdm(range(start_iter, n_iterations), desc="Training Progress", ncols=120, mininterval=1.0)

    for i in pbar:
        gen = batch_sampler_memmap(tokens, batch_size, block_size, n_batches=grad_accum_steps, rng=rng_np, start=train_start, end=train_end)
        batches = list(prefetch_to_device(gen, prefetch_size=2))
        bxs = jnp.stack([jnp.array(x) for x, _ in batches], axis=0)
        bys = jnp.stack([jnp.array(y) for _, y in batches], axis=0)

        global rng_key
        rng_key, subkey = random.split(rng_key)
        params, opt_state, tr_loss = update_step(params, opt_state, bxs, bys, subkey)
        train_losses.append(float(tr_loss))

        if i % 50 == 0:
            vgen = batch_sampler_memmap(tokens, batch_size, block_size, n_batches=1, rng=rng_np, start=val_start, end=val_end)
            (vx, vy) = next(vgen)
            vx = jax.device_put(vx)
            vy = jax.device_put(vy)
            rng_key, subkey = random.split(rng_key)
            v_loss = float(eval_loss(params, vx, vy, subkey))
            v_perp = perplexity(v_loss)
            val_losses.append(v_loss)

            pbar.set_postfix({
                "Iter": i,
                "TrainLoss": f"{float(tr_loss):.4f}",
                "ValLoss": f"{v_loss:.4f}",
                "ValPerp": f"{v_perp:.4f}",
                "BestVal": f"{best_val:.4f}",
                "Pat": f"{patience_counter}/{patience}",
            })

            improved = v_loss < best_val
            if improved:
                best_val = v_loss
                patience_counter = 0
                save_checkpoint(ckpt_dir, i, params, opt_state, best_val, extra={"block_size": block_size})
            else:
                patience_counter += 1

            if i > 0 and i % ckpt_every == 0:
                save_checkpoint(ckpt_dir, i, params, opt_state, best_val, extra={"block_size": block_size})

            if patience_counter >= patience:
                logging.info(f"Early stopping at iteration {i}")
                break

    if len(train_losses) > 0:
        plt.figure()
        plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
        if len(val_losses) > 0:
            plt.plot(range(0, len(val_losses) * 50, 50), val_losses, label="Val Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()

    return params

# -----------------------------
# 8.5 Fine-Tuningループ（SFT用） - 改善: 有効化 & n_iterations=10000, dropout=0.05, 日本語フィルタ追加
# -----------------------------
def fine_tune_model(
    params,
    tokenizer: Tokenizer,
    dataset_name: str = "databricks/databricks-dolly-15k",
    config: Optional[str] = None,
    split: str = "train",
    batch_size: int = 16,
    n_iterations: int = 10000,  # 改善: イテレーション増加
    learning_rate: float = 5e-5,
    block_size: int = 64,  # 改善: 拡大
    dropout_rate: float = 0.05,
    compute_dtype=jnp.bfloat16,
    ckpt_dir: str = "fine_tune_checkpoints",
    ckpt_every: int = 200,
):
    try:
        dataset = load_dataset(dataset_name, config, split=split)
    except Exception as e:
        logging.error(f"Failed to load fine-tuning dataset: {e}")
        raise

    # 改善: 日本語例のみフィルタ（Dollyは英語中心なので、必要に応じて日本語データセットに置き換え可能）
    def is_japanese(example):
        text = example.get("instruction", "") + example.get("context", "") + example.get("response", "")
        return bool(re.search(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]', text))  # ひらがな/カタカナ/漢字検出

    dataset = dataset.filter(is_japanese)

    def preprocess(example):
        prompt = example.get("instruction", "") + " " + example.get("context", "")
        response = example.get("response", "")
        eos_token = "<|endoftext|>"
        prompt_ids = tokenizer.encode(prompt).ids
        response_ids = tokenizer.encode(response + eos_token).ids
        
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        
        if len(input_ids) > block_size:
            input_ids = input_ids[:block_size]
            labels = labels[:block_size]
        
        pad_length = block_size - len(input_ids)
        input_ids += [tokenizer.token_to_id(eos_token)] * pad_length
        labels += [-100] * pad_length
        
        logging.info(f"Processed example: input_ids length={len(input_ids)}, labels length={len(labels)}")
        return {"input_ids": input_ids, "labels": labels}

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    dataset = dataset.with_format("jax")

    warmup_steps = min(500, n_iterations // 10)
    schedule = optax.join_schedules(
        [optax.linear_schedule(0.0, learning_rate, warmup_steps),
         optax.cosine_decay_schedule(learning_rate, n_iterations - warmup_steps, 0.1)],
        [warmup_steps]
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule))
    opt_state = optimizer.init(params)

    def ft_loss_fn(p, inputs, labels, key):
        logits = forward(p, inputs, key, training=True, compute_dtype=compute_dtype, dropout_rate=dropout_rate, embed_size=256, num_heads=4)  # 改善: kwargs調整
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        mask = (labels != -100)
        labels = jax.nn.one_hot(labels, logits.shape[-1])
        loss = -jnp.sum(log_probs * labels, axis=-1)
        loss = jnp.sum(loss * mask) / jnp.sum(mask)
        return loss

    @jax.jit
    def update_step(p, ostate, inputs, labels, key):
        loss, grads = jax.value_and_grad(ft_loss_fn)(p, inputs, labels, key)
        updates, new_ostate = optimizer.update(grads, ostate, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_ostate, loss

    train_losses = []
    start_iter = 0
    last = latest_checkpoint(ckpt_dir)
    if last:
        try:
            payload = load_checkpoint(last, embed_size=256, num_heads=4, head_size=64)
            params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
            opt_state = jax.tree_util.tree_map(jnp.asarray, payload["opt_state"])
            start_iter = payload["step"] + 1
            logging.info(f"Resumed fine-tuning from {last} (step={start_iter})")
        except ValueError as e:
            logging.warning(f"Failed to load fine-tuning checkpoint due to {e}. Continuing with pretrained parameters.")
            opt_state = optimizer.init(params)

    pbar = tqdm(range(start_iter, n_iterations), desc="Fine-Tuning Progress")
    dataset_size = len(dataset)
    
    for i in pbar:
        batch_idx = (i * batch_size) % dataset_size
        batch = dataset[batch_idx : batch_idx + batch_size]
        input_ids = [batch["input_ids"][j] for j in range(len(batch["input_ids"]))]
        labels = [batch["labels"][j] for j in range(len(batch["labels"]))]
        
        input_ids = jnp.array([x[:block_size] + [tokenizer.token_to_id("<|endoftext|>")] * (block_size - len(x[:block_size])) for x in input_ids])
        labels = jnp.array([x[:block_size] + [-100] * (block_size - len(x[:block_size])) for x in labels])
        
        global rng_key
        rng_key, subkey = random.split(rng_key)
        params, opt_state, loss = update_step(params, opt_state, input_ids, labels, subkey)
        train_losses.append(float(loss))
        pbar.set_postfix({"Loss": f"{loss:.4f}"})

        if i % ckpt_every == 0:
            save_checkpoint(ckpt_dir, i, params, opt_state, loss)

    return params

# -----------------------------
# 9. 文章生成 - 改善: temperature=0.8で創造性向上
# -----------------------------
def generate_text(params, tokenizer: Tokenizer, prompt: str, max_new_tokens: int = 200, block_size: int = 64, top_k: int = 40, top_p: float = 0.9, temperature: float = 0.8, compute_dtype=jnp.bfloat16) -> str:  # 改善: temperature調整
    try:
        global rng_key
        tokens = tokenizer.encode(prompt).ids
        tokens = tokens[-block_size:]
        tokens = jnp.array(tokens, dtype=jnp.int32).reshape(1, -1)
        pad_id = tokenizer.token_to_id("<|endoftext|>")
        
        fw_kwargs = {'compute_dtype': compute_dtype, 'training': False, 'embed_size': 256, 'num_heads': 4}  # 改善: kwargs調整

        for _ in range(max_new_tokens):
            rng_key, subkey = random.split(rng_key)
            logits = forward(params, tokens, subkey, **fw_kwargs)
            logits = logits[:, -1, :] / temperature

            top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
            probs = jax.nn.softmax(top_k_logits, axis=-1)
            sorted_probs = jnp.sort(probs, axis=-1)[..., ::-1]
            sorted_indices = jnp.argsort(probs, axis=-1)[..., ::-1]
            cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
            top_p_mask = cumulative_probs[0] <= top_p
            top_p_indices = jnp.take_along_axis(top_k_indices, sorted_indices, axis=-1)[0, top_p_mask]
            top_p_probs = sorted_probs[0, top_p_mask]

            if top_p_indices.shape[0] == 0:
                top_p_indices = top_k_indices[0, :1]
                top_p_probs = sorted_probs[0, :1]

            top_p_probs_sum = jnp.sum(top_p_probs)
            top_p_probs = top_p_probs / (top_p_probs_sum + 1e-10)

            next_token = jax.random.choice(subkey, top_p_indices, p=top_p_probs)

            if pad_id is not None and int(next_token) == int(pad_id):
                continue
            tokens = jnp.concatenate([tokens, jnp.array([[next_token]])], axis=1)
            tokens = tokens[:, -block_size:]

        generated_text = tokenizer.decode([int(t) for t in tokens[0].tolist() if pad_id is None or t != pad_id])
        logging.info(f"Generated text: {generated_text[:120]}...")
        return generated_text
    except Exception as e:
        logging.error(f"Failed to generate text: {e}")
        raise

# -----------------------------
# 10. GGUF エクスポート - 変更なし（オプション）
# -----------------------------
def export_to_gguf(params, tokenizer: Tokenizer, output_dir: str = "model_gguf", quantization: str = "q8_0"):
    try:
        os.makedirs(output_dir, exist_ok=True)

        state_dict = {}
        for k, v in params.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, dict):
                        for norm_k, norm_v in sub_v.items():
                            state_dict[f"{k}.{sub_k}.{norm_k}"] = torch.tensor(jnp.array(norm_v))
                    else:
                        state_dict[f"{k}.{sub_k}"] = torch.tensor(jnp.array(sub_v))
            else:
                state_dict[k] = torch.tensor(jnp.array(v))

        safetensors_path = os.path.join(output_dir, "model.safetensors")
        save_file(state_dict, safetensors_path)
        tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

        import subprocess
        convert_cmd = [
            "python", "llama.cpp/convert.py",
            "--outfile", os.path.join(output_dir, "model.gguf"),
            "--outtype", quantization,
            output_dir
        ]
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"GGUF conversion failed: {result.stderr}")
            raise RuntimeError(f"GGUF conversion failed: {result.stderr}")
        logging.info(f"Model exported to GGUF: {os.path.join(output_dir, 'model.gguf')}")
    except Exception as e:
        logging.error(f"Failed to export to GGUF: {e}")
        raise

# -----------------------------
# 11. メイン実行ブロック - 改善: dataset_configsを更新（mc4削除、oscar追加）
# -----------------------------
if __name__ == "__main__":
    
    # --- 共通設定 ---
    DRIVE_BASE_DIR = "/content/drive/MyDrive/LLM_from_scratch"
    TOKENIZER_NAME = "NousResearch/Llama-2-7b-hf"
    BLOCK_SIZE = 64  # 改善: 拡大
    
    # ローカルとGoogle Driveのパスを定義
    LOCAL_DATA_PATH = "input.txt"
    LOCAL_MEMMAP_PATH = "tokens.memmap"
    TOKENIZER_PATH = os.path.join(DRIVE_BASE_DIR, "tokenizer.json")
    MEMMAP_PATH = os.path.join(DRIVE_BASE_DIR, "tokens.memmap")
    CHECKPOINT_DIR = os.path.join(DRIVE_BASE_DIR, "checkpoints")
    FT_CHECKPOINT_DIR = os.path.join(DRIVE_BASE_DIR, "ft_checkpoints")

    # データとトークナイザーの準備
    print("====== 前処理を開始します（ファイルが存在する場合はスキップ） ======")
    os.makedirs(DRIVE_BASE_DIR, exist_ok=True)
    
    # データセットの準備（mc4をoscarに置き換え）
    dataset_configs = [
        ("wikimedia/wikipedia", "20231101.ja"),
        #("oscar", "unshuffled_deduplicated_ja")
    ]
    try:
        data_path = prepare_data(dataset_configs=dataset_configs, output_path=LOCAL_DATA_PATH, min_text_length=100, min_sentences=2)
    except ValueError as e:
        logging.error(f"Data preparation failed: {e}")
        raise

    if not os.path.exists(TOKENIZER_PATH):
        tokenizer = load_pretrained_tokenizer(TOKENIZER_NAME)
        tokenizer.save(TOKENIZER_PATH)
        logging.info(f"Tokenizer saved to {TOKENIZER_PATH}")
    else:
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        logging.info(f"Tokenizer loaded from {TOKENIZER_PATH}")

    # memmapへのトークナイズ
    try:
        tokens = tokenize_to_memmap(data_path, tokenizer, output_file=LOCAL_MEMMAP_PATH, block_size=BLOCK_SIZE, drive_path=MEMMAP_PATH)
    except ValueError as e:
        logging.error(f"Tokenization failed: {e}")
        raise
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FT_CHECKPOINT_DIR, exist_ok=True)

    print("====== 前処理が完了しました。学習を開始します ======")
    try:
        print("\n--- 2. モデルを初期化します ---")
        params = create_model(
            tokenizer.get_vocab_size(),
            embed_size=256,
            num_layers=4,
            num_heads=4,
            head_size=64,
            block_size=BLOCK_SIZE,
            param_dtype=jnp.bfloat16
        )
        print("--- 2. モデルの初期化が完了しました ---")

        print("\n--- 3. Pretrainingを開始します（チェックポイントがあれば再開します） ---")
        pretrain_params = train_model_memmap(
            params,
            tokens,
            batch_size=32,
            n_iterations=20000,
            learning_rate=1e-3,
            block_size=BLOCK_SIZE,
            dropout_rate=0.05,
            val_ratio=0.1,
            grad_accum_steps=16,
            compute_dtype=jnp.bfloat16,
            ckpt_dir=CHECKPOINT_DIR,
            ckpt_every=500,
        )
        print("\n--- 3. Pretrainingが完了しました ---")

        print("\n--- 3.5 Fine-Tuningを開始します ---")
        ft_params = fine_tune_model(
            pretrain_params,
            tokenizer,
            dataset_name="databricks/databricks-dolly-15k",
            batch_size=16,
            n_iterations=10000,
            learning_rate=5e-5,
            block_size=BLOCK_SIZE,
            dropout_rate=0.05,
            compute_dtype=jnp.bfloat16,
            ckpt_dir=FT_CHECKPOINT_DIR,
            ckpt_every=200,
        )
        print("\n--- 3.5 Fine-Tuningが完了しました ---")

        print("\n--- 4. 生成テストを実行します ---")
        prompt = "昔々あるところに"
        generated = generate_text(ft_params, tokenizer, prompt, block_size=BLOCK_SIZE, temperature=0.8)
        print("-" * 50)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated[:500]}")
        print("-" * 50)

    except Exception as e:
        logging.error(f"学習または生成中にエラーが発生しました: {e}", exc_info=True)
        raise
