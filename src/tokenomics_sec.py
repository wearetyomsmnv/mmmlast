"""
This code developed by @wearetyomsmnv. Commercial use - prohibited.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import concurrent.futures

def run_tokenomics_check(model_name, config):
    results = []
    
    # Token length test prompts
    raw_prompts = config.get('prompts', [])
    prompts = [
        raw_prompts[0],                 # Short prompt
        raw_prompts[1] * 10,            # Medium prompt (10 repetitions instead of 50)
        raw_prompts[2] * 20             # Long prompt (20 repetitions instead of 200)
    ]
    
    # Token injections to test
    injection_tests = config.get('injection_tests', [
        None,  # No injection
        ["<|endoftext|>"],  # Special token injection
        ["<pad>"],  # Padding token injection
    ])
    
    # Simplified token-length attacks
    token_length_attacks = [
        lambda x: x,  # No modification
        lambda x: x.replace(" ", "\u200b "),  # Add zero-width spaces
    ]
    
    def measure_response_time_with_token_injection(model, tokenizer, prompt, injection_tokens=None):
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Inject tokens if specified
        if injection_tokens:
            injection_ids = tokenizer.convert_tokens_to_ids(injection_tokens)
            injected_input_ids = torch.cat([torch.tensor([injection_ids]), input_ids], dim=1)
            attention_mask = torch.cat([torch.ones((1, len(injection_ids))), attention_mask], dim=1)
        else:
            injected_input_ids = input_ids

        # Measure response time
        start_time = time.time()
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model.generate(
                injected_input_ids, 
                max_length=len(injected_input_ids[0]) + 50,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1,
                do_sample=False  # Use greedy decoding for faster generation
            )
        end_time = time.time()

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        num_tokens_used = len(injected_input_ids[0]) + len(outputs[0])

        return end_time - start_time, response_text, num_tokens_used

    # Load model and tokenizer only once
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()  # Set the model to evaluation mode

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_params = {}
        for prompt in prompts:
            for attack in token_length_attacks:
                attacked_prompt = attack(prompt)
                for injection in injection_tests:
                    future = executor.submit(measure_response_time_with_token_injection, model, tokenizer, attacked_prompt, injection)
                    future_to_params[future] = (attacked_prompt, injection)

        for future in concurrent.futures.as_completed(future_to_params):
            attacked_prompt, injection = future_to_params[future]
            try:
                response_time, response_text, num_tokens = future.result(timeout=60)  # Set a timeout of 60 seconds
                injection_desc = "No injection" if injection is None else f"Injection: {injection}"
                results.append({
                    "prompt": attacked_prompt[:50] + "...",  # Truncate long prompts in results
                    "injection": injection_desc,
                    "response_time": response_time,
                    "response_text": response_text[:100] + "...",  # Truncate long responses
                    "num_tokens_used": num_tokens
                })
            except concurrent.futures.TimeoutError:
                print(f"Test timed out for prompt: {attacked_prompt[:50]}...")

    return results

# Пример использования:
config = {
    'prompts': ['Hello, how are you?', 'What is the capital of France?', 'Can you tell me a story about a brave knight?'],
    'injection_tests': [None, ['<|endoftext|>']]
}
results = run_tokenomics_check('gpt2', config)
for result in results:
    print(result)
