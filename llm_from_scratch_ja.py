# Google Colab 用: JAX で LLM をゼロから構築（日本語対応、GGUFエクスポート対応、memmap学習、GPU効率化、チェックポイント再開）

import os
import re
import pickle
import logging
from typing import Dict, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax
from datasets import load_dataset
from tokenizers import Tokenizer

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from safetensors.torch import save_file

# -----------------------------
# ロギング設定
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 乱数シード
rng_key = random.PRNGKey(0)

# デバッグ（必要に応じて）
jax.config.update("jax_debug_nans", False)

# -----------------------------
# 1. データセットの準備（ストリーミング & 低メモリ）
# -----------------------------
def prepare_data(dataset_name: str = "wikimedia/wikipedia", config: str = "20231101.ja", split: str = "train", chunk_size: int = 1000, output_path: str = "input.txt"):
    """
    日本語Wikipediaをストリーミングで読み出し、チャンク単位でシャッフル/加工して input.txt に逐次保存。
    """
    try:
        dataset = load_dataset(dataset_name, config, split=split, streaming=True)
        with open(output_path, "w", encoding="utf-8") as f:
            buffer = []
            for i, item in enumerate(dataset):
                if item.get("text", "").strip():
                    sentences = item["text"].split("。 ")
                    np.random.shuffle(sentences)
                    sentences = [s for s in sentences if np.random.random() > 0.05]
                    buffer.append("。 ".join(sentences))

                if (i + 1) % chunk_size == 0:
                    f.write(" ".join(buffer) + " ")
                    buffer = []
            if buffer:
                f.write(" ".join(buffer))
        logging.info(f"Dataset prepared in chunks and saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Failed to prepare dataset: {e}")
        raise

# -----------------------------
# 2. トークナイザーのロード
# -----------------------------
def load_pretrained_tokenizer(model_name: str = "NousResearch/Llama-2-7b-hf") -> Tokenizer:
    """
    🤗 tokenizers の Tokenizer を読み込み。Colab環境ではHugging Face Hubの認証は任意（公開モデルに限る）。
    """
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
# 3. memmap へトークナイズ（巨大コーパス対応）
# -----------------------------
def tokenize_to_memmap(
    text_path: str,
    tokenizer: Tokenizer,
    output_file: str = "tokens.memmap",
    block_size: int = 128,
    chunk_size: int = 5000
) -> np.memmap:
    """
    大規模テキストをチャンク単位でトークナイズし、memmapとして保存する。
    2パス: (1) トークン数を概算 -> (2) memmapに書き込み
    """
    logging.info("First pass: counting total tokens...")
    total_tokens = 0
    buffer = []
    with open(text_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            buffer.append(line)
            if (i + 1) % chunk_size == 0:
                ids = tokenizer.encode(" ".join(buffer)).ids
                total_tokens += len(ids)
                buffer = []
        if buffer:
            ids = tokenizer.encode(" ".join(buffer)).ids
            total_tokens += len(ids)

    total_tokens = total_tokens - (total_tokens % block_size)
    if total_tokens <= 0:
        raise ValueError("No tokens produced. Check your input or tokenizer.")
    logging.info(f"Total tokens (aligned to block_size={block_size}): {total_tokens:,}")

    tokens = np.memmap(output_file, dtype=np.int32, mode="w+", shape=(total_tokens,))

    logging.info("Second pass: writing tokens to memmap...")
    buffer = []
    offset = 0
    with open(text_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            buffer.append(line)
            if (i + 1) % chunk_size == 0:
                ids = tokenizer.encode(" ".join(buffer)).ids
                ids = ids[: len(ids) - (len(ids) % block_size)]
                if offset + len(ids) > total_tokens:
                    ids = ids[: total_tokens - offset]
                tokens[offset : offset + len(ids)] = ids
                offset += len(ids)
                buffer = []
        if buffer and offset < total_tokens:
            ids = tokenizer.encode(" ".join(buffer)).ids
            ids = ids[: len(ids) - (len(ids) % block_size)]
            if offset + len(ids) > total_tokens:
                ids = ids[: total_tokens - offset]
            tokens[offset : offset + len(ids)] = ids
            offset += len(ids)

    tokens.flush()
    logging.info(f"Tokenized dataset saved to {output_file}")
    return tokens

# -----------------------------
# 4. モデル作成（シンプルTransformer）
# -----------------------------
def create_model(vocab_size: int, embed_size: int = 512, num_layers: int = 6, num_heads: int = 8, head_size: int = 64, block_size: int = 128, dropout_rate: float = 0.05, param_dtype=jnp.bfloat16):
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
        params["final"] = u((embed_size, vocab_size), scale_e)
        return params
    return init_params(random.split(rng_key)[0])

# -----------------------------
# 5. 前処理ユーティリティ
# -----------------------------
def layer_norm(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias

def forward(params, x: jnp.ndarray, rng_key: jnp.ndarray, embed_size: int = 512, num_heads: int = 8, head_size: int = 64, dropout_rate: float = 0.05, training: bool = True, compute_dtype=jnp.bfloat16) -> jnp.ndarray:
    """
    1-GPU前提で jit する前提。内部演算は compute_dtype (bfloat16) で実施し、出力は float32 に変換。
    """
    x = x.astype(jnp.int32)
    rng, dropout_rng = random.split(rng_key)
    embeddings = params["embed"][x].astype(compute_dtype)
    embeddings = layer_norm(embeddings, params["embed_norm"]["scale"], params["embed_norm"]["bias"])

    seq_len = x.shape[-1]
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=compute_dtype)).reshape(1, 1, seq_len, seq_len)

    for i in range(len([k for k in params.keys() if k.startswith("layer_")])):
        layer = params[f"layer_{i}"]
        q = jnp.dot(embeddings, layer["w_q"]).reshape(-1, seq_len, num_heads, head_size)
        k = jnp.dot(embeddings, layer["w_k"]).reshape(-1, seq_len, num_heads, head_size)
        v = jnp.dot(embeddings, layer["w_v"]).reshape(-1, seq_len, num_heads, head_size)

        attn = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(jnp.array(head_size, dtype=compute_dtype))
        attn = attn * causal_mask - 1e9 * (1 - causal_mask)
        attn = jax.nn.softmax(attn, axis=-1)

        if training and dropout_rate > 0.0:
            dropout_mask = jax.random.bernoulli(dropout_rng, 1.0 - dropout_rate, attn.shape)
            attn = jnp.where(dropout_mask, attn / (1.0 - dropout_rate), 0.0)

        attn_out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(-1, seq_len, num_heads * head_size)
        embeddings = embeddings + jnp.dot(attn_out, layer["w_o"])
        embeddings = layer_norm(embeddings, layer["attn_norm"]["scale"], layer["attn_norm"]["bias"])

        ffn = jax.nn.relu(jnp.dot(embeddings, layer["ffn1"]))

        if training and dropout_rate > 0.0:
            dropout_mask = jax.random.bernoulli(dropout_rng, 1.0 - dropout_rate, ffn.shape)
            ffn = jnp.where(dropout_mask, ffn / (1.0 - dropout_rate), 0.0)

        ffn = jnp.dot(ffn, layer["ffn2"])
        embeddings = embeddings + ffn
        embeddings = layer_norm(embeddings, layer["ffn_norm"]["scale"], layer["ffn_norm"]["bias"])

    logits = jnp.dot(embeddings, params["final"]).astype(jnp.float32)
    return logits

def loss_fn(params, x: jnp.ndarray, y: jnp.ndarray, rng_key: jnp.ndarray, **fw_kwargs) -> jnp.ndarray:
    logits = forward(params, x, rng_key, training=True, **fw_kwargs)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    targets = jax.nn.one_hot(y, logits.shape[-1])
    return -jnp.mean(jnp.sum(log_probs * targets, axis=-1))

# -----------------------------
# 6. データローダ（memmap） + GPUプリフェッチ
# -----------------------------
def batch_sampler_memmap(tokens: np.memmap, batch_size: int, block_size: int, n_batches: int, rng: np.random.RandomState, start: int, end: int):
    """
    memmap の指定範囲 [start, end) からランダムサンプリングしたバッチを n_batches 生成（CPU側）
    """
    max_index = (end - start) - block_size - 1
    for _ in range(n_batches):
        idx = rng.randint(0, max_index, size=batch_size)
        # 直列コピーだが、バッチサイズが大きくても memmap のスライスは軽い
        X = np.stack([tokens[start + j : start + j + block_size] for j in idx]).astype(np.int32)
        Y = np.stack([tokens[start + j + 1 : start + j + block_size + 1] for j in idx]).astype(np.int32)
        yield X, Y

def prefetch_to_device(generator, prefetch_size: int = 2):
    """
    CPU -> GPU への非同期プリフェッチ。単純なイテレータベース。
    """
    queue = []
    for item in generator:
        item_dev = jax.tree_util.tree_map(jax.device_put, item)
        queue.append(item_dev)
        if len(queue) > prefetch_size:
            yield queue.pop(0)
    while queue:
        yield queue.pop(0)

# -----------------------------
# 7. チェックポイント関連
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

def load_checkpoint(path: str):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload

# -----------------------------
# 8. 学習ループ（memmap + GPU効率化 + 再開対応）
# -----------------------------
def train_model_memmap(
    params,
    tokens: np.memmap,
    batch_size: int = 128,
    n_iterations: int = 15000,
    learning_rate: float = 1e-3,
    block_size: int = 128,
    dropout_rate: float = 0.05,
    val_ratio: float = 0.2,
    grad_accum_steps: int = 1,
    compute_dtype=jnp.bfloat16,
    ckpt_dir: str = "checkpoints",
    ckpt_every: int = 200,
    resume: bool = True,
):
    """
    - memmapからランダムサンプリング
    - GPUへプリフェッチ
    - bfloat16で計算（可）
    - チェックポイント保存/再開
    - 勾配蓄積（grad_accum_steps）
    """
    warmup_steps = max(2000, n_iterations // 10)
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

    # JIT: update_step（勾配蓄積に対応）
    def loss_for_batch(p, bx, by, key):
        return loss_fn(p, bx, by, key, compute_dtype=compute_dtype)

    @jax.jit
    def update_step(p, ostate, bxs, bys, key):
        # bxs, bys: (grad_accum_steps, batch, seq)
        def step_carry(carry, inp):
            pi, ki, (bx, by) = carry
            loss, grads = jax.value_and_grad(loss_for_batch)(pi, bx, by, ki)
            updates, new_ostate = optimizer.update(grads, ostate)
            new_p = optax.apply_updates(pi, updates)
            return (new_p, ki, (bx, by)), (loss, new_ostate)

        # 逐次蓄積（JIT 内で scan しても可だが簡潔に単発処理）
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
        return loss_fn(p, bx, by, key, compute_dtype=compute_dtype)

    train_losses, val_losses = [], []
    best_val = float("inf")
    patience = 800
    patience_counter = 0
    start_iter = 0

    # 再開
    if resume:
        last = latest_checkpoint(ckpt_dir)
        if last is not None:
            payload = load_checkpoint(last)
            params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
            opt_state = jax.tree_util.tree_map(jnp.asarray, payload["opt_state"])
            best_val = float(payload.get("best_val", best_val))
            start_iter = int(payload.get("step", 0)) + 1
            logging.info(f"Resumed from {last} (step={start_iter-1}, best_val={best_val:.4f})")

    rng_np = np.random.RandomState(0)
    pbar = tqdm(range(start_iter, n_iterations), desc="Training Progress", ncols=120)

    # 1回の progress ステップで grad_accum_steps 分だけ CPUで作成→GPUプリフェッチ→更新
    for i in pbar:
        # --- サンプル生成（CPU）
        gen = batch_sampler_memmap(tokens, batch_size, block_size, n_batches=grad_accum_steps, rng=rng_np, start=train_start, end=train_end)
        # --- プリフェッチ -> デバイスに載せる
        batches = list(prefetch_to_device(gen, prefetch_size=2))
        # バッチテンソルをスタック (T, B, L)
        bxs = jnp.stack([jnp.array(x) for x, _ in batches], axis=0)
        bys = jnp.stack([jnp.array(y) for _, y in batches], axis=0)

        # --- 更新
        global rng_key
        rng_key, subkey = random.split(rng_key)
        params, opt_state, tr_loss = update_step(params, opt_state, bxs, bys, subkey)
        train_losses.append(float(tr_loss))

        if i % 50 == 0:
            # validation
            vgen = batch_sampler_memmap(tokens, batch_size, block_size, n_batches=1, rng=rng_np, start=val_start, end=val_end)
            (vx, vy) = next(vgen)
            vx = jax.device_put(vx)
            vy = jax.device_put(vy)
            rng_key, subkey = random.split(rng_key)
            v_loss = float(eval_loss(params, vx, vy, subkey))
            val_losses.append(v_loss)

            pbar.set_postfix({
                "Iter": i,
                "TrainLoss": f"{float(tr_loss):.4f}",
                "ValLoss": f"{v_loss:.4f}",
                "BestVal": f"{best_val:.4f}",
                "Pat": f"{patience_counter}/{patience}",
            })

            # checkpoint & early stopping
            improved = v_loss < best_val
            if improved:
                best_val = v_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (i % ckpt_every == 0) or improved:
                save_checkpoint(ckpt_dir, i, params, opt_state, best_val, extra={"block_size": block_size})

            if patience_counter >= patience:
                logging.info(f"Early stopping at iteration {i}")
                break

    # 可視化
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
# 9. 文章生成
# -----------------------------
def generate_text(params, tokenizer: Tokenizer, prompt: str, max_new_tokens: int = 200, block_size: int = 128, top_k: int = 40, top_p: float = 0.9, temperature: float = 0.9, compute_dtype=jnp.bfloat16) -> str:
    try:
        rng_key = random.PRNGKey(0)
        tokens = tokenizer.encode(prompt).ids
        tokens = tokens[-block_size:]
        tokens = jnp.array(tokens, dtype=jnp.int32).reshape(1, -1)
        pad_id = tokenizer.token_to_id("<|endoftext|>")
        eot_id = tokenizer.token_to_id("<|endoftext|>")

        for _ in range(max_new_tokens):
            rng_key, subkey = random.split(rng_key)
            logits = forward(params, tokens, subkey, training=False, compute_dtype=compute_dtype)
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

            if (pad_id is not None and int(next_token) == int(pad_id)) or (eot_id is not None and int(next_token) == int(eot_id)):
                continue
            tokens = jnp.concatenate([tokens, jnp.array([[next_token]])], axis=1)
            tokens = tokens[:, -block_size:]

        generated_text = tokenizer.decode([int(t) for t in tokens[0].tolist() if (pad_id is None or t != pad_id) and (eot_id is None or t != eot_id)])
        logging.info(f"Generated text: {generated_text[:120]}...")
        return generated_text
    except Exception as e:
        logging.error(f"Failed to generate text: {e}")
        raise

# -----------------------------
# 10. GGUF エクスポート
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

        # llama.cpp の convert.py を利用（事前にセットアップが必要）
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
# 11. メイン実行例
# -----------------------------
if __name__ == "__main__":
    block_size = 128
    try:
        # 1) データ準備
        data_path = prepare_data(dataset_name="wikimedia/wikipedia", config="20231101.ja")

        # 2) トークナイザー
        tokenizer = load_pretrained_tokenizer("NousResearch/Llama-2-7b-hf")

        # 3) memmap 方式でトークナイズ（Google Drive 上に保存推奨）
        tokens = tokenize_to_memmap(data_path, tokenizer, output_file="tokens.memmap", block_size=block_size)

        # 4) モデル
        params = create_model(tokenizer.get_vocab_size(), embed_size=512, num_layers=6, head_size=64, block_size=block_size, param_dtype=jnp.bfloat16)

        # 5) 学習（GPU効率化 + チェックポイント保存・再開）
        params = train_model_memmap(
            params,
            tokens,
            batch_size=128,
            n_iterations=3000,
            learning_rate=3e-3,
            block_size=block_size,
            dropout_rate=0.05,
            val_ratio=0.2,
            grad_accum_steps=1,       # GPUメモリに応じて 2,4,... に増やす
            compute_dtype=jnp.bfloat16,
            ckpt_dir="checkpoints",
            ckpt_every=200,
            resume=True,
        )

        # 6) GGUF 出力（任意）
        # export_to_gguf(params, tokenizer)

        # 7) 生成テスト
        prompt = "昔々あるところに"
        generated = generate_text(params, tokenizer, prompt, block_size=block_size, top_k=40, top_p=0.9, temperature=0.9, compute_dtype=jnp.bfloat16)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated[:500]}")

    except Exception as e:
        logging.error(f"Execution failed: {e}")
        raise
