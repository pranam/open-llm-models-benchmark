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
MMLU_SUBSET = 'all'  # Use the 'all' configuration of MMLU
NUM_QUESTIONS = 100  # Number of questions to evaluate for a quick test

# --- Helper Functions (from previous script) ---
def get_installed_models():
    """Returns a set of model names that are already installed locally."""
    return {model['model'] for model in ollama.list()['models']}

def pull_model(model_name):
    """Pulls a model from the Ollama registry, showing progress."""
    print(f"\nPulling model: {model_name}")
    # ... (rest of the function is unchanged, omitted for brevity)
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

# --- MMLU Evaluation Functions ---
def load_mmlu_dataset(subset, num_questions):
    """Loads and samples the MMLU dataset."""
    print(f"\nLoading MMLU dataset ({subset} subset)...")
    try:
        dataset = load_dataset("cais/mmlu", subset, split='test', streaming=True)
        shuffled_dataset = dataset.shuffle(seed=42)
        sample = list(shuffled_dataset.take(num_questions))
        print(f"Loaded {len(sample)} questions from MMLU.")
        return sample
    except Exception as e:
        print(f"Failed to load MMLU dataset: {e}")
        return None

def format_mmlu_prompt(item):
    """Formats a single MMLU question into a standardized prompt."""
    choices = ['A', 'B', 'C', 'D']
    prompt = f"The following is a multiple-choice question. Please provide the letter of the correct answer.\n\n"
    prompt += f"Question: {item['question']}\n"
    for i, choice in enumerate(item['choices']):
        prompt += f"{choices[i]}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt

def parse_model_answer(response_text):
    """Extracts the chosen letter (A, B, C, or D) from the model's response."""
    match = re.search(r'\b([A-D])\b', response_text.strip().upper())
    if match:
        return match.group(1)
    return None # Could not parse an answer

def evaluate_model_on_mmlu(model_name, dataset):
    """Evaluates a model's accuracy on the MMLU dataset sample."""
    print(f"\n--- Evaluating {model_name} on MMLU ---")
    correct = 0
    total = len(dataset)
    
    for i, item in enumerate(dataset):
        sys.stdout.write(f'\r  Question {i+1}/{total}...')
        sys.stdout.flush()
        
        prompt = format_mmlu_prompt(item)
        correct_answer_index = item['answer']
        correct_answer_letter = ['A', 'B', 'C', 'D'][correct_answer_index]
        
        try:
            response = ollama.generate(model=model_name, prompt=prompt)
            response_text = response['response']
            parsed_answer = parse_model_answer(response_text)
            
            if parsed_answer == correct_answer_letter:
                correct += 1
        except Exception as e:
            print(f"\nAn error occurred while evaluating {model_name}: {e}")
            # We don't count errors as incorrect, just skip
            continue

    sys.stdout.write('\n')
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Accuracy for {model_name}: {accuracy:.2f}%")
    return accuracy

# --- Main Execution ---
def main():
    """Main function to run the benchmark and evaluation."""
    print("Starting LLM Benchmark and MMLU Evaluation...")
    
    # Load MMLU data first
    mmlu_sample = load_mmlu_dataset(MMLU_SUBSET, NUM_QUESTIONS)
    if not mmlu_sample:
        print("Exiting due to dataset loading failure.")
        return

    installed_models = get_installed_models()
    results = []

    for model_name in MODELS_TO_BENCHMARK:
        # Pull model if not installed
        if model_name not in installed_models:
            if not pull_model(model_name):
                results.append({'model': model_name, 'error': 'Failed to pull model'})
                continue
        
        # Run MMLU Evaluation
        accuracy = evaluate_model_on_mmlu(model_name, mmlu_sample)
        
        # For now, we'll just store accuracy. You can add back the performance benchmark later.
        results.append({
            'model': model_name,
            'mmlu_accuracy_%': round(accuracy, 2)
        })

    # Display results in a table
    df = pd.DataFrame(results)
    print("\n--- MMLU Evaluation Results ---")
    print(df.to_string())

if __name__ == "__main__":
    main()

