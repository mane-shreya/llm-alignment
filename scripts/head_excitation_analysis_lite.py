"""
Head Excitation Analysis for LLM - LITE VERSION
Simulated attention head analysis without downloading full model
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class HeadExcitationAnalyzerLite:
    """Simulated head excitation analyzer for quick demo"""
    
    def __init__(self):
        print("Initializing Head Excitation Analyzer (LITE)...")
        self.num_layers = 32
        self.num_heads = 8
        self.seed = 42
    
    def analyze_prompt(self, text, simulation=True):
        """Simulate attention head analysis for a prompt"""
        print(f"\nAnalyzing: {text[:80]}...")
        
        np.random.seed(self.seed + hash(text) % 1000)
        
        results = {
            "total_layers": self.num_layers,
            "layer_stats": [],
            "top_heads": [],
        }
        
        # Simulate layer-wise statistics
        for layer_idx in range(self.num_layers):
            head_entropy = np.random.beta(2, 5, self.num_heads)  # Right-skewed distribution
            head_means = np.random.beta(3, 2, self.num_heads)    # Peaked distribution
            
            layer_stats = {
                "layer": layer_idx,
                "num_heads": self.num_heads,
                "avg_entropy": float(np.mean(head_entropy)),
                "max_entropy": float(np.max(head_entropy)),
                "min_entropy": float(np.min(head_entropy)),
                "avg_weight": float(np.mean(head_means)),
            }
            results["layer_stats"].append(layer_stats)
            
            # Track all heads
            for head_idx, (entropy, weight) in enumerate(zip(head_entropy, head_means)):
                results["top_heads"].append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "entropy": entropy,
                    "weight": weight,
                    "excitation": entropy * weight,
                })
        
        # Sort by excitation score
        results["top_heads"].sort(key=lambda x: x["excitation"], reverse=True)
        results["top_heads"] = results["top_heads"][:10]  # Top 10 heads
        
        return results
    
    def print_results(self, results):
        """Print analysis results"""
        print("\n" + "="*70)
        print("HEAD EXCITATION ANALYSIS RESULTS")
        print("="*70)
        
        print(f"\nTotal Layers Analyzed: {results['total_layers']}")
        
        print("\n" + "-"*70)
        print("LAYER-WISE STATISTICS")
        print("-"*70)
        
        for stat in results["layer_stats"][:5]:  # Show first 5 layers
            print(f"\nLayer {stat['layer']}:")
            print(f"  Num Heads: {stat['num_heads']}")
            print(f"  Entropy - Avg: {stat['avg_entropy']:.4f}, Max: {stat['max_entropy']:.4f}, Min: {stat['min_entropy']:.4f}")
            print(f"  Avg Weight: {stat['avg_weight']:.4f}")
        
        if len(results["layer_stats"]) > 5:
            print(f"\n... ({len(results['layer_stats']) - 5} more layers)")
        
        print("\n" + "-"*70)
        print("TOP 10 MOST ACTIVE HEADS (by excitation score)")
        print("-"*70)
        
        for i, head in enumerate(results["top_heads"], 1):
            print(f"\n{i}. Layer {head['layer']}, Head {head['head']}")
            print(f"   Entropy: {head['entropy']:.4f}")
            print(f"   Weight: {head['weight']:.4f}")
            print(f"   Excitation Score: {head['excitation']:.4f}")
    
    def batch_analyze(self, prompts_csv):
        """Analyze multiple prompts from CSV"""
        df = pd.read_csv(prompts_csv)
        all_results = []
        
        print(f"\nAnalyzing {len(df)} prompts from {prompts_csv}...")
        
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
                
                print(f"  [{idx+1:2d}/{len(df)}] {category:20s} | Excitation: {avg_layer_entropy:.4f}")
                
            except Exception as e:
                print(f"  [{idx+1:2d}/{len(df)}] Error: {str(e)}")
        
        # Save to CSV
        results_df = pd.DataFrame(all_results)
        output_path = "outputs/head_excitation_results.csv"
        results_df.to_csv(output_path, index=False)
        
        print("\n" + "="*70)
        print("BATCH ANALYSIS SUMMARY")
        print("="*70)
        print(f"Processed: {len(all_results)} prompts")
        print(f"Average Entropy: {results_df['avg_entropy'].mean():.4f}")
        print(f"Entropy Std Dev: {results_df['avg_entropy'].std():.4f}")
        
        print("\nBy Category:")
        category_stats = results_df.groupby('category')['avg_entropy'].agg(['mean', 'std', 'count'])
        print(category_stats.to_string())
        
        print(f"\nResults saved to: {output_path}")
        
        return results_df


def main():
    """Main execution"""
    
    # Initialize analyzer
    analyzer = HeadExcitationAnalyzerLite()
    
    # Example 1: Analyze a single prompt
    test_prompt = "What is the capital of France?"
    print("\n" + "="*70)
    print("SINGLE PROMPT ANALYSIS")
    print("="*70)
    results = analyzer.analyze_prompt(test_prompt)
    analyzer.print_results(results)
    
    # Example 2: Batch analyze prompts from CSV
    prompts_file = "prompts/prompts.csv"
    if Path(prompts_file).exists():
        print("\n\n" + "="*70)
        print("BATCH ANALYSIS FROM CSV")
        print("="*70)
        analyzer.batch_analyze(prompts_file)
    else:
        print(f"\nNote: {prompts_file} not found. Skipping batch analysis.")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
