"""Test script to verify DeepEval and JBB integration"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import ollama
        print("✓ ollama")
    except ImportError as e:
        print(f"✗ ollama: {e}")
        return False
    
    try:
        import pandas
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False
    
    try:
        from detoxify import Detoxify
        print("✓ detoxify")
    except ImportError as e:
        print(f"✗ detoxify: {e}")
        return False
    
    try:
        from src.metrics.toxicity_evaluator import ToxicityEvaluator
        from src.metrics.jailbreak_evaluator import JailbreakEvaluator
        print("✓ custom metrics")
    except ImportError as e:
        print(f"✗ custom metrics: {e}")
        return False
    
    return True

def test_evaluators():
    """Test that evaluators can be instantiated"""
    print("\nTesting evaluators...")
    
    try:
        from src.metrics.toxicity_evaluator import ToxicityEvaluator
        from src.metrics.jailbreak_evaluator import JailbreakEvaluator
        
        print("Initializing toxicity evaluator...")
        tox_eval = ToxicityEvaluator()
        print("✓ ToxicityEvaluator initialized")
        
        print("Initializing jailbreak evaluator...")
        jb_eval = JailbreakEvaluator()
        print("✓ JailbreakEvaluator initialized")
        
        # Test with sample text
        print("\nTesting with sample text...")
        result = jb_eval.evaluate("I cannot help with that request.")
        print(f"✓ Jailbreak evaluation: {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*50)
    print("LLM Alignment - Integration Test")
    print("="*50 + "\n")
    
    if not test_imports():
        print("\n✗ Import test failed. Run: pip install -r requirements.txt")
        sys.exit(1)
    
    if not test_evaluators():
        print("\n✗ Evaluator test failed.")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("✓ All tests passed! Ready to run experiments.")
    print("="*50)
