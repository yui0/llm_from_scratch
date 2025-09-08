# JAXを使用したゼロからの言語モデル（LLM）構築

このプロジェクトは、JAXを使用してゼロから言語モデル（LLM）を構築する方法を示します。効率的なテキスト処理のために、事前学習済みのGPT-2トークナイザーを活用しています。データ準備、モデル構築、トレーニング、テキスト生成が含まれており、Google Colab向けに最適化されています。

## 機能
- **データセット準備**: `wikitext-103-raw-v1`データセットを使用し、データ拡張（文のランダムシャッフルと一部のドロップ）を行います。
- **トークナイザー**: 事前学習済みのGPT-2トークナイザーを使用し、パディングや特殊トークンを追加。
- **モデルアーキテクチャ**: マルチヘッドアテンション、フィードフォワード層、層正規化を備えたトランスフォーマーベースのモデル。
- **トレーニング**: Adamオプティマイザ、学習率スケジューリング、検証損失に基づく早期停止を備えたトレーニングループ。
- **テキスト生成**: 多様で制御されたテキスト生成のためのトップkおよびトップpサンプリングをサポート。

## 必要条件
- Python 3.8以上
- JAX
- Optax
- Datasets
- Tokenizers
- NumPy
- Matplotlib

## 使用方法
1. **環境の準備**:
   ```bash
   pip install jax jaxlib optax datasets tokenizers numpy matplotlib
   ```
2. **スクリプトの実行**:
   - `llm_from_scratch.py`をGoogle Colabまたはローカル環境で開きます。
   - スクリプトを実行してデータセットを準備し、モデルをトレーニングし、テキストを生成します。
3. **テキスト生成**:
   - トレーニング後、プロンプト（例: "Once upon a time"）に基づいてテキストを生成します。
   - `top_k`、`top_p`、`temperature`などの生成パラメータをカスタマイズして出力を調整できます。

## 例
```python
prompt = "Once upon a time"
generated = generate_text(params, tokenizer, prompt, block_size=128, top_k=40, top_p=0.9, temperature=0.9)
print(f"Prompt: {prompt}")
print(f"Generated: {generated}")
```

## 備考
- デフォルトでは3000イテレーションのトレーニングを行い、過学習を防ぐために早期停止を採用。
- パフォーマンス調整のために`block_size`、`batch_size`、`learning_rate`を調整可能。
- トークナイザーはパディングおよびテキスト終了マーカーとして`<|endoftext|>`を使用。

## ライセンス
MITライセンス
