"""Load and run Qwen2.5-7B-Instruct in 4-bit quantization for local inference."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

_model = None
_tokenizer = None


def print_vram(label: str):
    if not torch.cuda.is_available():
        print(f"[{label}] CUDA not available")
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[{label}] VRAM allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB | total: {total:.2f} GB")


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    print_vram("Before model load")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # extra compression, helps on tight VRAM
    )

    print(f"Loading {MODEL_NAME} in 4-bit...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="cuda:0",
        torch_dtype=torch.float16,
    )

    print_vram("After model load")
    return _model, _tokenizer


def generate(prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
    model, tokenizer = load_model()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print_vram("Before generation")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    print_vram("After generation")

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()