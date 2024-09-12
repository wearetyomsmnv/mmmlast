"""
This code developed by @wearetyomsmnv. Commercial use - prohibited.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
