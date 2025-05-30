import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
from typing import Optional

class MedicalQAModel:
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the Medical QA Model."""
        self.device = device
        self.max_length = max_length
        
        # Initialize model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        logging.info(f"Model initialized on {device}")

    def generate_response(
        self,
        input_text: str,
        prompt_template: Optional[str] = None
    ) -> str:
        """Generate a response for the given input text."""
        try:
            # Prepare input
            if prompt_template:
                input_text = prompt_template.format(question=input_text)
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=4,
                    temperature=1.0,
                    top_p=0.9,
                    do_sample=True,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def load_model(self, path: str):
        """Load the model and tokenizer."""
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"Model loaded from {path}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = MedicalQAModel()
    
    # Example prompt template
    prompt_template = """
    Medical Question: {question}
    Please provide a detailed and accurate medical response:
    """
    
    # Example usage
    question = "What are the symptoms of diabetes?"
    response = model.generate_response(question, prompt_template)
    print(f"Question: {question}")
    print(f"Response: {response}") 