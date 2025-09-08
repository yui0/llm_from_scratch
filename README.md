# LLM from Scratch with JAX

This project demonstrates how to build a Language Model (LLM) from scratch using JAX, leveraging a pretrained GPT-2 tokenizer for efficient text processing. The implementation includes data preparation, model creation, training, and text generation, all optimized for Google Colab.

## Features
- **Dataset Preparation**: Uses the `wikitext-103-raw-v1` dataset with data augmentation (random sentence shuffling and dropping).
- **Tokenizer**: Employs a pretrained GPT-2 tokenizer with added padding and special tokens.
- **Model Architecture**: A transformer-based model with multi-head attention, feedforward layers, and layer normalization.
- **Training**: Implements a training loop with Adam optimizer, learning rate scheduling, and early stopping based on validation loss.
- **Text Generation**: Supports top-k and top-p sampling for diverse and controlled text generation.

## Requirements
- Python 3.8+
- JAX
- Optax
- Datasets
- Tokenizers
- NumPy
- Matplotlib

## Usage
1. **Prepare the Environment**:
   ```bash
   pip install jax jaxlib optax datasets tokenizers numpy matplotlib
   ```
2. **Run the Script**:
   - Open `llm_from_scratch.py` in Google Colab or a local environment.
   - Execute the script to prepare the dataset, train the model, and generate text.
3. **Generate Text**:
   - After training, the script generates text based on a prompt (e.g., "Once upon a time").
   - Customize generation parameters like `top_k`, `top_p`, and `temperature` for varied outputs.

## Example
```python
prompt = "Once upon a time"
generated = generate_text(params, tokenizer, prompt, block_size=128, top_k=40, top_p=0.9, temperature=0.9)
print(f"Prompt: {prompt}")
print(f"Generated: {generated}")
```

## Notes
- The model is trained for 3000 iterations by default, with early stopping to prevent overfitting.
- Adjust `block_size`, `batch_size`, and `learning_rate` for performance tuning.
- The tokenizer uses `<|endoftext|>` for padding and end-of-text markers.

## License
MIT License

