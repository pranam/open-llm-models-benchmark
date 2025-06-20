import ollama
import pandas as pd
import time
import sys
import re
from datasets import load_dataset

# --- Configuration ---
MODELS_TO_BENCHMARK = [
    'tinyllama',
    'qwen2:1.5b',
    'gemma:2b',
    'phi3:mini',
    'mistral:7b',
    'qwen2:7b',
    'llama3:8b',
    'gemma2:9b',
]

# MMLU Configuration
MMLU_SUBSET = 'all'
NUM_QUESTIONS = 100  # Number of questions for MMLU evaluation

# Performance Benchmark Configuration
PERFORMANCE_PROMPT = "The sun is a star, a glowing ball of hot gas. It is the center of our solar system. Write a short, detailed paragraph about the sun's role in supporting life on Earth."

# --- Helper Functions ---
def get_installed_models():
    """Returns a set of model names that are already installed locally."""
    return {model['model'] for model in ollama.list()['models']}

def pull_model(model_name):
    """Pulls a model from the Ollama registry, showing progress."""
    print(f"\nPulling model: {model_name}")
    current_digest = ''
    try:
        for progress in ollama.pull(model_name, stream=True):
            digest = progress.get('digest', '')
            if digest != current_digest and digest != '':
                current_digest = digest
                print(f"  {digest} {progress.get('status')}")
            if 'total' in progress and 'completed' in progress:
                percent = round((progress['completed'] / progress['total']) * 100)
                bar = '█' * (percent // 2)
                sys.stdout.write(f'\r  Downloading: [{bar:<50}] {percent}% ')
                sys.stdout.flush()
        sys.stdout.write('\n')
        print(f"Successfully pulled {model_name}")
    except Exception as e:
        print(f"Error pulling {model_name}: {e}")
        return False
    return True

# --- Benchmark Functions ---
def benchmark_performance(model_name, prompt):
    """Runs a single model performance benchmark and returns the tokens/sec."""
    print(f"\n--- Benchmarking Performance: {model_name} ---")
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
        )
        total_duration = response.get('total_duration', 0)
        eval_count = response.get('eval_count', 0)

        if total_duration > 0:
            tokens_per_second = eval_count / (total_duration / 1e9)
        else:
            tokens_per_second = 0

        print(f"Tokens/sec: {tokens_per_second:.2f}")
        return round(tokens_per_second, 2)
    except Exception as e:
        print(f"Error during performance benchmark for {model_name}: {e}")
        return 0

def load_mmlu_dataset(subset, num_questions):
    """Loads and samples the MMLU dataset."""
    print(f"\nLoading MMLU dataset ({subset} subset)...")
    try:
        dataset = load_dataset("cais/mmlu", subset, split='test', streaming=True)
        sample = list(dataset.shuffle(seed=42).take(num_questions))
        print(f"Loaded {len(sample)} questions from MMLU.")
        return sample
    except Exception as e:
        print(f"Failed to load MMLU dataset: {e}")
        return None

def evaluate_model_on_mmlu(model_name, dataset):
    """Evaluates a model's accuracy on the MMLU dataset sample."""
    print(f"\n--- Evaluating {model_name} on MMLU ---")
    correct = 0
    total = len(dataset)
    choices = ['A', 'B', 'C', 'D']
    
    for i, item in enumerate(dataset):
        sys.stdout.write(f'\r  Question {i+1}/{total}...')
        sys.stdout.flush()
        
        # Format prompt
        prompt = f"Question: {item['question']}\n"
        for j, choice in enumerate(item['choices']):
            prompt += f"{choices[j]}. {choice}\n"
        prompt += "\nAnswer:"

        correct_answer_letter = choices[item['answer']]
        
        try:
            response = ollama.generate(model=model_name, prompt=prompt)
            response_text = response['response']
            # Parse answer
            match = re.search(r'\b([A-D])\b', response_text.strip().upper())
            parsed_answer = match.group(1) if match else None
            
            if parsed_answer == correct_answer_letter:
                correct += 1
        except Exception:
            continue

    sys.stdout.write('\n')
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"MMLU Accuracy: {accuracy:.2f}%")
    return accuracy

# --- Main Execution ---
def main():
    """Main function to run the combined benchmarks."""
    print("Starting Combined LLM Benchmark (Performance + MMLU)...")
    
    mmlu_sample = load_mmlu_dataset(MMLU_SUBSET, NUM_QUESTIONS)
    if not mmlu_sample:
        print("Exiting due to dataset loading failure.")
        return

    installed_models = get_installed_models()
    results = []

    for model_name in MODELS_TO_BENCHMARK:
        if model_name not in installed_models:
            if not pull_model(model_name):
                results.append({'model': model_name, 'error': 'Failed to pull model'})
                continue
        
        accuracy = evaluate_model_on_mmlu(model_name, mmlu_sample)
        tokens_sec = benchmark_performance(model_name, PERFORMANCE_PROMPT)
        
        results.append({
            'model': model_name,
            'mmlu_accuracy_%': round(accuracy, 2),
            'tokens_per_sec': tokens_sec
        })

    df = pd.DataFrame(results)
    print("\n--- Combined Benchmark Results ---")
    
    # Reorder and display results
    if 'error' in df.columns:
        df = df[['model', 'mmlu_accuracy_%', 'tokens_per_sec', 'error']]
    else:
        df = df[['model', 'mmlu_accuracy_%', 'tokens_per_sec']]
    
    print(df.to_string())

if __name__ == "__main__":
    main()

