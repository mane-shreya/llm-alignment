import sys
import pandas as pd
import ollama
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.toxicity_evaluator import ToxicityEvaluator
from src.metrics.jailbreak_evaluator import JailbreakEvaluator


def main():
    """Run full evaluation with toxicity and jailbreak detection"""
    
    print("Initializing evaluators...")
    toxicity_eval = ToxicityEvaluator()
    jailbreak_eval = JailbreakEvaluator()
    
    print("Loading prompts...")
    data = pd.read_csv("prompts/prompts.csv")
    results = []
    
    print(f"Processing {len(data)} prompts...\n")
    
    for idx, row in data.iterrows():
        print(f"[{idx+1}/{len(data)}] Category: {row['category']}")
        
        # Generate response
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": row['prompt']}]
        )
        response_text = response['message']['content']
        
        # Evaluate toxicity
        toxicity_result = toxicity_eval.evaluate(row['prompt'], response_text)
        
        # Evaluate jailbreak
        jailbreak_result = jailbreak_eval.evaluate(response_text)
        
        # Combine results
        results.append({
            "prompt": row['prompt'],
            "category": row['category'],
            "response": response_text,
            "toxicity_score": toxicity_result['toxicity_score'],
            "is_toxic": toxicity_result['is_toxic'],
            "jailbreak_label": jailbreak_result['jailbreak_label'],
            "is_jailbroken": jailbreak_result['is_jailbroken']
        })
        
        print(f"  Toxicity: {toxicity_result['toxicity_score']:.3f} | Jailbreak: {jailbreak_result['is_jailbroken']}\n")
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv("outputs/outputs.csv", index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total responses: {len(df_results)}")
    print(f"Toxic responses: {df_results['is_toxic'].sum()}")
    print(f"Jailbroken responses: {df_results['is_jailbroken'].sum()}")
    print(f"Average toxicity: {df_results['toxicity_score'].mean():.3f}")
    print("\nResults saved to outputs/outputs.csv")


if __name__ == "__main__":
    main()
