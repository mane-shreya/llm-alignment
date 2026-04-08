"""Simple test to verify Ollama + Mistral is working before running full evaluation"""

import ollama

def test_ollama_connection():
    """Test if Ollama is running and Mistral is available"""
    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": "Say hello in 5 words or less."}]
        )
        print("✓ Ollama is running")
        print("✓ Mistral model is available")
        print(f"\nTest response: {response['message']['content']}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running: ollama serve")
        print("  2. Mistral is installed: ollama pull mistral")
        return False

if __name__ == "__main__":
    print("="*50)
    print("Testing Ollama + Mistral Connection")
    print("="*50 + "\n")
    
    if test_ollama_connection():
        print("\n✓ Ready to run experiments!")
    else:
        print("\n✗ Fix the issues above before proceeding.")
        exit(1)
