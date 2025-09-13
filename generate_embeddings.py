#!/usr/bin/env python3
"""
Generate embeddings for different models and dimensions
"""
import os
import subprocess
import sys
from pathlib import Path

def generate_embeddings():
    """Generate embeddings for different models"""
    
    models = [
        ("sentence-transformers/all-MiniLM-L6-v2", "embeddings_384d.parquet"),
        ("sentence-transformers/all-mpnet-base-v2", "embeddings_768d.parquet"),
        ("sentence-transformers/all-distilroberta-v1", "embeddings_768d_distil.parquet"),
        ("openai:text-embedding-ada-002", "embeddings_1536d_ada.parquet"),
        ("openai:text-embedding-3-small", "embeddings_1536d_3small.parquet"),
        ("openai:text-embedding-3-large", "embeddings_3072d_3large.parquet"),
    ]
    
    for model, output_file in models:
        print(f"\n🔄 Generating embeddings for {model} -> {output_file}")
        
        if model.startswith("openai:"):
            # OpenAI models
            openai_model = model.replace("openai:", "")
            cmd = [
                "python", "embeddings/embed.py",
                "--csv", "data/sample_data.csv",
                "--out", f"data/{output_file}",
                "--use_openai",
                "--model", openai_model
            ]
        else:
            # Sentence Transformers models
            cmd = [
                "python", "embeddings/embed.py",
                "--csv", "data/sample_data.csv",
                "--out", f"data/{output_file}",
                "--model", model
            ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Success: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error generating {output_file}: {e.stderr}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    generate_embeddings()
