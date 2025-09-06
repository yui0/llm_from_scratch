# Google Colab 用: JAX で LLM をゼロから構築（事前学習済みトークナイザー使用）
# 参考: https://speed1313.github.io/posts/llm-from-scratch/

import jax
import jax.numpy as jnp
from jax import random
import optax
from datasets import load_dataset
from tokenizers import Tokenizer
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt

# 乱数シードの設定
rng_key = random.PRNGKey(0)

# JAXのデバッグ設定
jax.config.update('jax_debug_nans', True)

# 1. データセットの準備
def prepare_data(dataset_name: str = "wikitext", split: str = "train+validation") -> str:
    dataset = load_dataset(dataset_name, "wikitext-103-raw-v1", split=split)
    text = " ".join([t for t in dataset["text"] if t.strip()])  # 空の文字列を除外
    # データ拡張: 文をシャッフルし、ランダムに一部を切り捨て
    sentences = text.split(". ")
    np.random.shuffle(sentences)
    sentences = [s for s in sentences if np.random.random() > 0.05]  # 5%をランダムにドロップ
    text = ". ".join(sentences)
    with open("input.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return "input.txt"

# 2. 事前学習済みトークナイザーのロード
def load_pretrained_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer.from_pretrained("gpt2")
    
    # Use <|endoftext|> as the padding token, or add a new one
    pad_token = "<|endoftext|>"
    pad_id = tokenizer.token_to_id(pad_token)
    
    if pad_id is None:
        # If <|endoftext|> is not in the vocabulary, add it
        tokenizer.add_special_tokens([pad_token])
        pad_id = tokenizer.token_to_id(pad_token)
    
    # Enable padding with the correct pad_id and pad_token
    tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
    
    # Add <|endoftext|> if not already present
    if "<|endoftext|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(["<|endoftext|>"])
    
    print(f"Vocab size: {tokenizer.get_vocab_size()}")  # Debug
    return tokenizer

# 3. トークン列への変換
def tokenize_text(text_path: str, tokenizer: Tokenizer, block_size: int = 128) -> np.ndarray:
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text).ids
    tokens = tokens[:len(tokens) - (len(tokens) % block_size)]
    print(f"Token sample: {tokens[:20]}")  # デバッグ用
    return np.array(tokens)

# 4. シンプルなトランスフォーマーモデルの定義
def create_model(vocab_size: int, embed_size: int = 512, num_layers: int = 6, num_heads: int = 8, head_size: int = 64, block_size: int = 128, dropout_rate: float = 0.05):
    def init_params(rng):
        params = {}
        params["embed"] = random.uniform(rng, (vocab_size, embed_size), minval=-np.sqrt(6/embed_size), maxval=np.sqrt(6/embed_size))
        params["embed_norm"] = {"scale": jnp.ones(embed_size), "bias": jnp.zeros(embed_size)}
        for i in range(num_layers):
            params[f"layer_{i}"] = {
                "w_q": random.uniform(rng, (embed_size, head_size * num_heads), minval=-np.sqrt(6/embed_size), maxval=np.sqrt(6/embed_size)),
                "w_k": random.uniform(rng, (embed_size, head_size * num_heads), minval=-np.sqrt(6/embed_size), maxval=np.sqrt(6/embed_size)),
                "w_v": random.uniform(rng, (embed_size, head_size * num_heads), minval=-np.sqrt(6/embed_size), maxval=np.sqrt(6/embed_size)),
                "w_o": random.uniform(rng, (head_size * num_heads, embed_size), minval=-np.sqrt(6/(head_size * num_heads)), maxval=np.sqrt(6/(head_size * num_heads))),
                "attn_norm": {"scale": jnp.ones(embed_size), "bias": jnp.zeros(embed_size)},
                "ffn1": random.uniform(rng, (embed_size, embed_size * 4), minval=-np.sqrt(6/embed_size), maxval=np.sqrt(6/embed_size)),
                "ffn2": random.uniform(rng, (embed_size * 4, embed_size), minval=-np.sqrt(6/(embed_size * 4)), maxval=np.sqrt(6/(embed_size * 4))),
                "ffn_norm": {"scale": jnp.ones(embed_size), "bias": jnp.zeros(embed_size)}
            }
        params["final"] = random.uniform(rng, (embed_size, vocab_size), minval=-np.sqrt(6/embed_size), maxval=np.sqrt(6/embed_size))
        return params
    return init_params(random.split(rng_key)[0])

# 5. フォワードパス（因果マスクを追加、トレーニングフラグをJAX互換に修正）
def layer_norm(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias

@jax.jit
def forward(params, x: jnp.ndarray, rng_key: jnp.ndarray, embed_size: int = 512, num_heads: int = 8, head_size: int = 64, dropout_rate: float = 0.05, training: bool = True) -> jnp.ndarray:
    rng, dropout_rng = random.split(rng_key)
    embeddings = params["embed"][x]
    embeddings = layer_norm(embeddings, params["embed_norm"]["scale"], params["embed_norm"]["bias"])
    
    seq_len = x.shape[-1]
    # 因果マスク
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len))).reshape(1, 1, seq_len, seq_len)
    
    for i in range(len([k for k in params.keys() if k.startswith("layer_")])):
        layer = params[f"layer_{i}"]
        
        # マルチヘッドアテンション
        q = jnp.dot(embeddings, layer["w_q"]).reshape(-1, seq_len, num_heads, head_size)
        k = jnp.dot(embeddings, layer["w_k"]).reshape(-1, seq_len, num_heads, head_size)
        v = jnp.dot(embeddings, layer["w_v"]).reshape(-1, seq_len, num_heads, head_size)
        attn = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(head_size)
        attn = attn * causal_mask - 1e9 * (1 - causal_mask)  # マスク適用
        attn = jax.nn.softmax(attn, axis=-1)
        
        # Dropout for attention (apply only during training)
        dropout_mask = jax.random.bernoulli(dropout_rng, 1.0 - dropout_rate, attn.shape)
        attn = jnp.where(training, dropout_mask * attn / (1.0 - dropout_rate), attn)
        
        attn_out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(-1, seq_len, num_heads * head_size)
        embeddings = embeddings + jnp.dot(attn_out, layer["w_o"])
        embeddings = layer_norm(embeddings, layer["attn_norm"]["scale"], layer["attn_norm"]["bias"])
        
        # フィードフォワード
        ffn = jax.nn.relu(jnp.dot(embeddings, layer["ffn1"]))
        
        # Dropout for feedforward (apply only during training)
        dropout_mask = jax.random.bernoulli(dropout_rng, 1.0 - dropout_rate, ffn.shape)
        ffn = jnp.where(training, dropout_mask * ffn / (1.0 - dropout_rate), ffn)
        
        ffn = jnp.dot(ffn, layer["ffn2"])
        embeddings = embeddings + ffn
        embeddings = layer_norm(embeddings, layer["ffn_norm"]["scale"], layer["ffn_norm"]["bias"])
    
    logits = jnp.dot(embeddings, params["final"])
    return logits

# 6. 損失関数（検証損失を追加）
def loss_fn(params, x: jnp.ndarray, y: jnp.ndarray, rng_key: jnp.ndarray) -> float:
    logits = forward(params, x, rng_key, training=True)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    targets = jax.nn.one_hot(y, logits.shape[-1])
    return -jnp.mean(jnp.sum(log_probs * targets, axis=-1))

# 7. トレーニングループ
def train_model(params, tokens: np.ndarray, batch_size: int = 128, n_iterations: int = 15000, learning_rate: float = 1e-3):
    warmup_steps = 2000  # Extended warmup
    schedule = optax.join_schedules([
        optax.linear_schedule(0.0, learning_rate, warmup_steps),
        optax.cosine_decay_schedule(learning_rate, n_iterations - warmup_steps, 0.5)  # Less aggressive decay
    ], [warmup_steps])
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),  # Tighter gradient clipping
        optax.adam(schedule)
    )
    opt_state = optimizer.init(params)
    
    val_size = len(tokens) // 5
    train_tokens, val_tokens = tokens[:-val_size], tokens[-val_size:]
    
    @jax.jit
    def update_step(params, opt_state, batch_x, batch_y, rng_key):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_x, batch_y, rng_key)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    @jax.jit
    def val_loss(params, x, y, rng_key):
        return loss_fn(params, x, y, rng_key)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 500  # Reduced patience
    rng_key = random.PRNGKey(0)
    
    for i in range(n_iterations):
        rng_key, subkey = random.split(rng_key)
        idx = jax.random.randint(subkey, (batch_size,), 0, len(train_tokens) - block_size - 1)
        batch_x = jnp.array([train_tokens[j:j+block_size] for j in idx])
        batch_y = jnp.array([train_tokens[j+1:j+block_size+1] for j in idx])
        
        params, opt_state, train_loss = update_step(params, opt_state, batch_x, batch_y, subkey)
        train_losses.append(train_loss)
        
        if i % 50 == 0:  # More frequent validation
            rng_key, subkey = random.split(rng_key)
            val_idx = np.random.randint(0, len(val_tokens) - block_size - 1, batch_size)  # Larger validation batch
            val_x = jnp.array([val_tokens[j:j+block_size] for j in val_idx])
            val_y = jnp.array([val_tokens[j+1:j+block_size+1] for j in val_idx])
            v_loss = val_loss(params, val_x, val_y, subkey)
            val_losses.append(v_loss)
            print(f"Iteration {i}, Train Loss: {train_loss:.4f}, Val Loss: {v_loss:.4f}")
            
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at iteration {i}")
                break
    
    plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
    plt.plot(range(0, len(val_losses)*50, 50), val_losses, label="Val Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    return params

# 8. 文章生成
def generate_text(params, tokenizer: Tokenizer, prompt: str, max_new_tokens: int = 200, block_size: int = 128, top_k: int = 40, top_p: float = 0.9, temperature: float = 0.9) -> str:
    rng_key = random.PRNGKey(0)  # 生成用の乱数シード
    tokens = tokenizer.encode(prompt).ids
    tokens = tokens[-block_size:]  # block_size に収まるようにトリミング
    tokens = jnp.array(tokens, dtype=jnp.int32).reshape(1, -1)
    pad_id = tokenizer.token_to_id(" ")
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    
    for _ in range(max_new_tokens):
        rng_key, subkey = random.split(rng_key)
        logits = forward(params, tokens, subkey, training=False)
        logits = logits[:, -1, :] / temperature
        
        # トップkとトップpサンプリング
        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        probs = jax.nn.softmax(top_k_logits, axis=-1)
        sorted_probs = jnp.sort(probs, axis=-1, descending=True)
        sorted_indices = jnp.argsort(probs, axis=-1, descending=True)
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
        top_p_mask = cumulative_probs[0] <= top_p
        top_p_indices = top_k_indices[0, top_p_mask]
        top_p_probs = sorted_probs[0, top_p_mask]
        
        # Handle empty top_p_indices case
        if top_p_indices.shape[0] == 0:
            # Fallback to top-1 token
            top_p_indices = top_k_indices[0, :1]
            top_p_probs = sorted_probs[0, :1]
        
        # Normalize probabilities
        top_p_probs_sum = jnp.sum(top_p_probs)
        top_p_probs = top_p_probs / (top_p_probs_sum + 1e-10)  # Add small epsilon to avoid division by zero
        
        next_token = jax.random.choice(subkey, top_p_indices, p=top_p_probs)
        
        # パディングやEOTトークンを避ける
        if next_token in [pad_id, eot_id]:
            continue
        tokens = jnp.concatenate([tokens, jnp.array([[next_token]])], axis=1)
        tokens = tokens[:, -block_size:]
    
    return tokenizer.decode([t for t in tokens[0].tolist() if t not in [pad_id, eot_id]])

# 実行
block_size = 128
data_path = prepare_data()
tokenizer = load_pretrained_tokenizer()
tokens = tokenize_text(data_path, tokenizer, block_size)
params = create_model(tokenizer.get_vocab_size(), embed_size=512, num_layers=6, head_size=64, block_size=block_size)
params = train_model(params, tokens, batch_size=128, n_iterations=3000, learning_rate=3e-3)

# サンプル生成
prompt = "Once upon a time"
generated = generate_text(params, tokenizer, prompt, block_size=block_size, top_k=40, top_p=0.9, temperature=0.9)
print(f"Prompt: {prompt}")
print(f"Generated: {generated}")
