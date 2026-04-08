"""
Head Excitation Analysis for LLM
Analyzes attention head patterns and their contributions to model outputs
"""

import sys
import torch
import pandas as pd
from pathlib import Path
import transformers
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class HeadExcitationAnalyzer:
    """Analyzes attention head activations and their importance"""
    
    def __init__(self, model_name="mistral"):
        print(f"Loading model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Map ollama model names to HuggingFace names
        if model_name == "mistral":
            hf_model = "mistralai/Mistral-7B-Instruct-v0.1"
        else:
            hf_model = model_name
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_model, 
            output_attentions=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
        
    def analyze_prompt(self, text, max_new_tokens=50):
        """Analyze attention heads for a given prompt"""
        print(f"\nAnalyzing: {text[:100]}...")
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                max_new_tokens=max_new_tokens
            )
        
        # Extract attention weights
        attentions = outputs.attentions  # Tuple of tensors: (layers, batch_size, heads, seq_len, seq_len)
        
        return self._process_attention_heads(attentions, inputs, outputs)
    
    def _process_attention_heads(self, attentions, inputs, outputs):
        """Process and summarize attention head activations"""
        results = {
            "total_layers": len(attentions),
            "layer_stats": [],
            "top_heads": [],
            "head_entropy": []
        }
        
        for layer_idx, layer_attention in enumerate(attentions):
            # layer_attention shape: (batch_size, num_heads, seq_len, seq_len)
            batch_size, num_heads, seq_len, _ = layer_attention.shape
            
            # Calculate head importance using entropy
            head_entropy = []
            head_means = []
            
            for head_idx in range(num_heads):
                head_weights = layer_attention[0, head_idx]  # (seq_len, seq_len)
                
                # Calculate entropy (measure of focus)
                prob_dist = head_weights[-1, :]  # Last token's attention distribution
                entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-9)).item()
                head_entropy.append(entropy)
                
                # Calculate mean attention weight
                mean_weight = head_weights.mean().item()
                head_means.append(mean_weight)
            
            # Store layer statistics
            layer_stats = {
                "layer": layer_idx,
                "num_heads": num_heads,
                "avg_entropy": float(np.mean(head_entropy)),
                "max_entropy": float(np.max(head_entropy)),
                "min_entropy": float(np.min(head_entropy)),
                "avg_weight": float(np.mean(head_means)),
            }
            results["layer_stats"].append(layer_stats)
            
            # Track top heads overall
            for head_idx, (entropy, weight) in enumerate(zip(head_entropy, head_means)):
                results["top_heads"].append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "entropy": entropy,
                    "weight": weight,
                    "excitation": entropy * weight  # Combined score
                })
        
        # Sort by excitation score
        results["top_heads"].sort(key=lambda x: x["excitation"], reverse=True)
        results["top_heads"] = results["top_heads"][:10]  # Top 10 heads
        
        return results
    
    def print_results(self, results):
        """Print analysis results in readable format"""
        print("\n" + "="*60)
        print("HEAD EXCITATION ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\nTotal Layers Analyzed: {results['total_layers']}")
        
        print("\n" + "-"*60)
        print("LAYER-WISE STATISTICS")
        print("-"*60)
        
        for stat in results["layer_stats"]:
            print(f"\nLayer {stat['layer']}:")
            print(f"  Num Heads: {stat['num_heads']}")
            print(f"  Entropy - Avg: {stat['avg_entropy']:.4f}, Max: {stat['max_entropy']:.4f}, Min: {stat['min_entropy']:.4f}")
            print(f"  Avg Weight: {stat['avg_weight']:.4f}")
        
        print("\n" + "-"*60)
        print("TOP 10 MOST ACTIVE HEADS (by excitation score)")
        print("-"*60)
        
        for i, head in enumerate(results["top_heads"], 1):
            print(f"\n{i}. Layer {head['layer']}, Head {head['head']}")
            print(f"   Entropy: {head['entropy']:.4f}")
            print(f"   Weight: {head['weight']:.4f}")
            print(f"   Excitation Score: {head['excitation']:.4f}")
    
    def batch_analyze(self, prompts_csv):
        """Analyze multiple prompts from CSV"""
        df = pd.read_csv(prompts_csv)
        all_results = []
        
        print(f"\nAnalyzing {len(df)} prompts...")
        
        for idx, row in df.iterrows():
            prompt = row['prompt']
            category = row['category']
            
            try:
                results = self.analyze_prompt(prompt)
                
                # Extract key metrics
                top_head = results["top_heads"][0] if results["top_heads"] else {}
                avg_layer_entropy = np.mean([s["avg_entropy"] for s in results["layer_stats"]])
                
                all_results.append({
                    "prompt": prompt[:100],
                    "category": category,
                    "num_layers": results["total_layers"],
                    "avg_entropy": avg_layer_entropy,
                    "top_head_layer": top_head.get("layer", -1),
                    "top_head_id": top_head.get("head", -1),
                    "top_head_excitation": top_head.get("excitation", 0),
                })
                
                print(f"  [{idx+1}/{len(df)}] {category} - Excitation: {avg_layer_entropy:.4f}")
                
            except Exception as e:
                print(f"  [{idx+1}/{len(df)}] Error processing prompt: {str(e)}")
        
        # Save to CSV
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("outputs/head_excitation_results.csv", index=False)
        
        print("\n" + "="*60)
        print("BATCH ANALYSIS SUMMARY")
        print("="*60)
        print(f"Processed: {len(all_results)} prompts")
        print(f"Average Entropy: {results_df['avg_entropy'].mean():.4f}")
        print(f"Entropy Std Dev: {results_df['avg_entropy'].std():.4f}")
        
        print("\nBy Category:")
        category_stats = results_df.groupby('category')['avg_entropy'].agg(['mean', 'std', 'count'])
        print(category_stats)
        
        print(f"\nResults saved to: outputs/head_excitation_results.csv")
        
        return results_df


def main():
    """Main execution"""
    
    # Initialize analyzer
    analyzer = HeadExcitationAnalyzer(model_name="mistral")
    
    # Example 1: Analyze a single prompt
    test_prompt = "What is the capital of France?"
    print("\n" + "="*60)
    print("SINGLE PROMPT ANALYSIS")
    print("="*60)
    results = analyzer.analyze_prompt(test_prompt)
    analyzer.print_results(results)
    
    # Example 2: Batch analyze prompts from CSV
    prompts_file = "prompts/prompts.csv"
    if Path(prompts_file).exists():
        print("\n\n" + "="*60)
        print("BATCH ANALYSIS FROM CSV")
        print("="*60)
        analyzer.batch_analyze(prompts_file)
    else:
        print(f"\nNote: {prompts_file} not found. Skipping batch analysis.")


if __name__ == "__main__":
    main()
