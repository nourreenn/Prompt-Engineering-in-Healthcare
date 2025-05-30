import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_and_save_model():
    """Download and save the model and tokenizer."""
    try:
        # Create models directory if it doesn't exist
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        # Initialize model and tokenizer
        logger.info("Downloading model and tokenizer...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-cnn",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        # Save model and tokenizer
        save_path = model_dir / "medical_qa_model"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model and tokenizer saved to {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    download_and_save_model() 