"""
This code developed by @wearetyomsmnv. Commercial use - prohibited.
"""

import torch
from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance, Saliency


def check_interpretability(model, tokenizer, config):
    if model is None or not isinstance(model, torch.nn.Module):
        raise ValueError("Переданная модель недействительна.")
    
    if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
        raise ValueError("Токенизатор недействителен или не поддерживает метод encode.")
    
    # Sample input for interpretability checks
    input_text = "This is a sample input for interpretability analysis."
    if not input_text.strip():
        raise ValueError("Входной текст не может быть пустым.")
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    if input_ids is None or input_ids.size(0) == 0:
        raise ValueError("Ошибка токенизации: получен пустой список input_ids.")
    
    input_ids = input_ids.to(torch.long)  # Ensure long type
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Gather interpretability results
    interpretability_results = {
        "attention_interpretability": check_attention_interpretability(model, input_ids),
        "integrated_gradients": check_integrated_gradients(model, input_ids),
        "layer_conductance": check_layer_conductance(model, input_ids),
        "neuron_conductance": check_neuron_conductance(model, input_ids),
        "saliency_map": generate_saliency_map(model, input_ids),
    }

    # Perform cybersecurity-specific checks
    cybersecurity_assessment = {
        "adversarial_attack_risk": assess_adversarial_attack_risk(interpretability_results),
        "data_leakage_risk": assess_data_leakage_risk(interpretability_results),
        "criticality_of_vulnerabilities": assess_criticality_of_vulnerabilities(interpretability_results)
    }
    
    interpretability_results["cybersecurity_assessment"] = cybersecurity_assessment
    
    # Format and print the results
    formatted_output = format_interpretability_results(interpretability_results)
    print(formatted_output)
    
    return interpretability_results

def check_attention_interpretability(model, input_ids):
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
        
        if outputs.attentions is not None and len(outputs.attentions) > 0:
            attention_scores = outputs.attentions
            avg_attention_score = torch.mean(torch.stack(attention_scores)).item()
            if avg_attention_score < 0 or avg_attention_score > 1:
                raise ValueError("Attention scores выходят за пределы допустимых значений.")
            return {"average_attention_score": avg_attention_score}
        else:
            return {"average_attention_score": None, "error": "Attention scores not available"}
    except Exception as e:
        return {"error": str(e)}

def check_integrated_gradients(model, input_ids):
    try:
        # GPT2 Model outputs logits, so use a wrapper for the embeddings
        def model_forward(input_ids):
            outputs = model(input_ids)
            logits = outputs.logits
            return logits

        ig = IntegratedGradients(model_forward)
        attributions, delta = ig.attribute(inputs=input_ids, target=0, return_convergence_delta=True)
        
        total_attribution = attributions.sum().item()
        if not torch.isfinite(attributions).all():
            raise ValueError("Некорректные атрибуции: присутствуют бесконечные значения.")
        
        return {"integrated_gradients_score": total_attribution, "delta": delta.item()}
    except Exception as e:
        return {"error": str(e)}

def check_layer_conductance(model, input_ids):
    try:
        last_layer = model.transformer.h[-1]  # Adjusted for GPT-2 architecture
        lc = LayerConductance(model, last_layer)
        attributions = lc.attribute(inputs=input_ids, target=0)
        
        total_conductance = attributions.sum().item()
        if not torch.isfinite(attributions).all():
            raise ValueError("Некорректные атрибуции: присутствуют бесконечные значения.")
        
        return {"layer_conductance_score": total_conductance}
    except Exception as e:
        return {"error": str(e)}

def check_neuron_conductance(model, input_ids):
    try:
        last_layer = model.transformer.h[-1]  # Adjusted for GPT-2 architecture
        nc = NeuronConductance(model, last_layer)
        attributions = nc.attribute(inputs=input_ids, neuron_selector=0, target=0)
        
        total_conductance = attributions.sum().item()
        if not torch.isfinite(attributions).all():
            raise ValueError("Некорректные атрибуции: присутствуют бесконечные значения.")
        
        return {"neuron_conductance_score": total_conductance}
    except Exception as e:
        return {"error": str(e)}

def generate_saliency_map(model, input_ids):
    try:
        # Define a model forward function that outputs logits
        def model_forward(input_ids):
            outputs = model(input_ids)
            logits = outputs.logits
            return logits

        saliency = Saliency(model_forward)
        attributions = saliency.attribute(inputs=input_ids, target=0)
        
        total_saliency = attributions.sum().item()
        if not torch.isfinite(attributions).all():
            raise ValueError("Некорректные атрибуции: присутствуют бесконечные значения.")
        
        return {"saliency_map": total_saliency}
    except Exception as e:
        return {"error": str(e)}

def assess_adversarial_attack_risk(interpretability_results):
    try:
        ig_score = interpretability_results["integrated_gradients"].get("integrated_gradients_score")
        if ig_score is None:
            return {"status": "Unknown", "description": "Integrated gradients score is not available."}
        elif ig_score < 0:
            raise ValueError("Некорректное значение интегрированных градиентов: отрицательные значения.")
        elif ig_score > 10:  # Пример порогового значения, его нужно настроить
            return {"status": "High risk", "description": "Model highly sensitive to input changes, making it vulnerable to adversarial attacks."}
        else:
            return {"status": "Low risk", "description": "Model shows lower sensitivity, reducing the risk of adversarial attacks."}
    except Exception as e:
        return {"status": "Error", "description": f"An error occurred during assessment: {str(e)}"}

def assess_data_leakage_risk(interpretability_results):
    try:
        lc_score = interpretability_results["layer_conductance"].get("layer_conductance_score")
        if lc_score is None:
            return {"status": "Unknown", "description": "Layer conductance score is not available."}
        elif lc_score < 0:
            raise ValueError("Некорректное значение layer conductance: отрицательные значения.")
        elif lc_score > 15:  # Пример порогового значения
            return {"status": "High risk", "description": "Potential data leakage through specific layers or neurons detected."}
        else:
            return {"status": "Low risk", "description": "No significant data leakage risks detected."}
    except Exception as e:
        return {"status": "Error", "description": f"An error occurred during assessment: {str(e)}"}

def assess_criticality_of_vulnerabilities(interpretability_results):
    try:
        attack_risk = interpretability_results["cybersecurity_assessment"]["adversarial_attack_risk"]["status"]
        leakage_risk = interpretability_results["cybersecurity_assessment"]["data_leakage_risk"]["status"]
        
        if attack_risk == "High risk" or leakage_risk == "High risk":
            return {"status": "Critical", "description": "Detected vulnerabilities are critical and require immediate attention."}
        else:
            return {"status": "Non-critical", "description": "No critical vulnerabilities detected."}
    except Exception as e:
        return {"status": "Error", "description": f"An error occurred during assessment: {str(e)}"}

def format_interpretability_results(results):
    formatted_result = "Интерпретируемость модели:\n"
    
    # Process each result and assess its criticality
    for key, value in results.items():
        if key == "cybersecurity_assessment":
            formatted_result += "  Оценка кибербезопасности:\n"
            for assessment_key, assessment_value in value.items():
                formatted_result += f"    - {assessment_key.replace('_', ' ').capitalize()}: {format_criticality(assessment_value)}\n"
        else:
            formatted_result += f"  {key.replace('_', ' ').capitalize()}:\n"
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    formatted_result += f"    {sub_key}: {sub_value}\n"
            else:
                formatted_result += f"    {value}\n"
    
    return formatted_result

def format_criticality(assessment):
    status = assessment.get("status", "Unknown")
    description = assessment.get("description", "No details available")
    
    if status == "Low risk":
        icon = "🟢"
    elif status == "Medium risk":
        icon = "🟡"
    elif status == "High risk":
        icon = "🔴"
    else:
        icon = "⚪"
    
    return f"{icon} {status} ({description})"