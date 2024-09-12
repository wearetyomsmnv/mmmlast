"""
This code developed by @wearetyomsmnv. Commercial use - prohibited.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

def run_tokenomics_check(model_name, config):
    results = []
    
    # Token length test prompts
    raw_prompts = config.get('prompts', [])
    prompts = [
        raw_prompts[0],                 # Short prompt
        raw_prompts[1] * 50,            # Medium prompt (50 repetitions)
        raw_prompts[2] * 200            # Long prompt (200 repetitions)
    ]
    
    # Token injections to test
    injection_tests = config.get('injection_tests', [
        None,  # No injection
        ["<|endoftext|>"],  # Special token injection
        ["<pad>"],  # Padding token injection
    ])
    
    # Adding token-length attacks
    token_length_attacks = [
        lambda x: x,  # No modification
        lambda x: x.replace(" ", "\u200b "),  # Add zero-width spaces
        lambda x: " ".join([f"{word}." for word in x.split()]),  # Add periods after words
        lambda x: " ".join([f"{word}!" for word in x.split()])  # Add exclamation marks after words
    ]
    
    def measure_response_time_with_token_injection(model_name, prompt, injection_tokens=None):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad_token to eos_token if pad_token is not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids']

        # Create an attention mask
        attention_mask = inputs['attention_mask']

        # Inject tokens if specified
        if injection_tokens:
            injection_ids = tokenizer.convert_tokens_to_ids(injection_tokens)
            injected_input_ids = torch.cat([torch.tensor([injection_ids]), input_ids], dim=1)
            # Extend attention mask for the injected tokens
            attention_mask = torch.cat([torch.ones((1, len(injection_ids))), attention_mask], dim=1)
        else:
            injected_input_ids = input_ids

        # Measure response time
        start_time = time.time()
        outputs = model.generate(
            injected_input_ids, 
            max_length=len(injected_input_ids[0]) + 50,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id  # Explicitly set the pad token ID
        )
        end_time = time.time()

        # Decode the generated tokens
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        num_tokens_used = len(injected_input_ids[0]) + len(outputs[0])

        return end_time - start_time, response_text, num_tokens_used

    for prompt in prompts:
        for attack in token_length_attacks:
            attacked_prompt = attack(prompt)
            for injection in injection_tests:
                injection_desc = "No injection" if injection is None else f"Injection: {injection}"
                response_time, response_text, num_tokens = measure_response_time_with_token_injection(model_name, attacked_prompt, injection_tokens=injection)
                results.append({
                    "prompt": attacked_prompt,
                    "injection": injection_desc,
                    "response_time": response_time,
                    "response_text": response_text,
                    "num_tokens_used": num_tokens
                })
    
    return results