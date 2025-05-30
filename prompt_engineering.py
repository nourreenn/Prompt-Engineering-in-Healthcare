import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import logging
from data_loader import MedicalQADataLoader
from pathlib import Path
import gc
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_gpu_memory():
    """
    Configure GPU memory settings for PyTorch, optimized for P100 GPU.
    """
    try:
        if torch.cuda.is_available():
            # Get the number of GPUs
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} GPU(s)")
            
            # Set memory allocation for P100 (16GB HBM2)
            torch.cuda.empty_cache()
            # P100 is optimized for FP32, so we can use more memory
            torch.cuda.set_per_process_memory_fraction(0.9)  # 90% of available memory
            
            # Get GPU memory info
            for i in range(gpu_count):
                gpu = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {gpu.name}")
                logger.info(f"Memory: {gpu.total_memory / 1024**3:.2f} GB")
                logger.info(f"CUDA Capability: {gpu.major}.{gpu.minor}")
            
            # Set CUDA device properties for P100
            if "P100" in torch.cuda.get_device_name(0):
                logger.info("P100 GPU detected, optimizing settings...")
                # P100 is optimized for FP32, so we'll use that
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                # Set cudnn benchmark for better performance
                torch.backends.cudnn.benchmark = True
                # Set cudnn deterministic for reproducibility
                torch.backends.cudnn.deterministic = False
                # Enable memory pinning for better transfer speeds
                torch.cuda.set_device(0)
            
            return True
        else:
            logger.info("No GPU found, using CPU")
            return False
    except Exception as e:
        logger.error(f"Error setting up GPU memory: {str(e)}")
        return False

class MedicalPromptEngineering:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the Medical Prompt Engineering system.
        
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        # Set up GPU memory management
        self.has_gpu = setup_gpu_memory()
        self.device = "cuda" if self.has_gpu else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize data loader first
        self.data_loader = MedicalQADataLoader()
        
        # Load and preprocess data immediately
        if not self.load_and_preprocess_data():
            logger.error("Failed to load initial data")
        
        # Initialize tokenizer and model with memory optimization
        try:
            # Clear any existing models from memory
            gc.collect()
            if self.has_gpu:
                torch.cuda.empty_cache()
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Initialize model with memory optimization for P100
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32  # P100 is optimized for FP32
            ).to(self.device)
            
            # Initialize summarization pipeline with memory optimization
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.has_gpu else -1,
                torch_dtype=torch.float32,  # Use FP32 for inference
                max_length=200,
                min_length=50,
                do_sample=False,
                num_beams=5,
                length_penalty=1.5,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2
            )
            
            logger.info("Successfully initialized model and tokenizer")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize semantic similarity model
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize BLEU score smoothing
        self.smooth = SmoothingFunction().method1
        
        # Store results
        self.results = {
            'baseline': [],
            'prompt_engineered': []
        }
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess the medical datasets.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load all datasets
            if not self.data_loader.load_all_datasets():
                logger.error("Failed to load datasets")
                return False
            
            # Combine datasets
            if not self.data_loader.combine_datasets():
                logger.error("Failed to combine datasets")
                return False
            
            # Get dataset summary
            summary = self.data_loader.get_dataset_summary()
            logger.info("Dataset Summary:")
            logger.info(summary)
            
            return True
        except Exception as e:
            logger.error(f"Error in load_and_preprocess_data: {str(e)}")
            return False
    
    def create_prompt(self, text: str, prompt_type: str = "medical_qa") -> str:
        """
        Create a more specific medical Q&A prompt for the input text.
        
        Args:
            text (str): Input text to process
            prompt_type (str): Type of prompt (kept for compatibility)
            
        Returns:
            str: Prompted text
        """
        return f"""As a medical expert, provide a detailed and accurate answer to this medical question. 
        Include evidence-based information and medical guidelines.

        Question: {text}

        Provide a structured answer with these components:
        1. Common symptoms and signs
        2. When to seek immediate medical attention
        3. Risk factors and causes
        4. Brief medical explanation
        5. General management tips

        Answer:"""
    
    def generate_summary(self, text: str, prompt_type: str = "medical_qa", max_length: int = 200) -> str:
        """
        Generate summary using the model with memory management and improved parameters.
        
        Args:
            text (str): Input text to summarize
            prompt_type (str): Type of prompt to use
            max_length (int): Maximum length of summary
            
        Returns:
            str: Generated summary
        """
        try:
            # First, check if we have similar questions in our dataset
            if hasattr(self, 'data_loader') and hasattr(self.data_loader, 'combined_data'):
                # Extract key medical terms from the question
                medical_terms = text.lower().split()
                # Find similar questions in the dataset using more precise matching
                similar_questions = self.data_loader.combined_data[
                    self.data_loader.combined_data['Question'].str.contains(
                        '|'.join(medical_terms), 
                        case=False, 
                        na=False
                    )
                ]
                
                # Filter for most relevant matches
                if not similar_questions.empty:
                    # Sort by relevance (exact matches first)
                    similar_questions['relevance'] = similar_questions['Question'].apply(
                        lambda x: sum(term in x.lower() for term in medical_terms)
                    )
                    similar_questions = similar_questions.sort_values('relevance', ascending=False)
                    
                    # Get the most relevant answer
                    reference_answer = similar_questions.iloc[0]['Answer']
                    logger.info(f"Found matching question in dataset: {similar_questions.iloc[0]['Question']}")
                    return reference_answer
            
            # If no dataset match, use the model with appropriate prompt based on question type
            logger.info("No exact match found in dataset, using model generation")
            
            # Determine question type and create appropriate prompt
            if "what do i have" in text.lower() or "what could it be" in text.lower():
                # Symptom-based diagnosis question
                prompted_text = f"""As a medical expert, analyze these symptoms and provide possible conditions.
                Focus on providing accurate, evidence-based medical information.

                Symptoms: {text.lower()}

                Provide a medical assessment that includes:
                - Possible conditions based on the symptoms
                - When to seek immediate medical attention
                - Recommended next steps
                - General management tips
                - Important warning signs to watch for

                Answer:"""
            elif "how to" in text.lower() or "treatment" in text.lower():
                # Treatment/management question
                prompted_text = f"""As a medical expert, provide treatment and management advice.
                Focus on providing accurate, evidence-based medical information.

                Question: {text}

                Provide a medical answer that includes:
                - Recommended treatments
                - Management strategies
                - Lifestyle modifications
                - When to seek medical help
                - Important precautions

                Answer:"""
            elif "symptoms" in text.lower() or "signs" in text.lower():
                # Symptom information question
                prompted_text = f"""As a medical expert, provide detailed information about these symptoms.
                Focus on providing accurate, evidence-based medical information.

                Question: {text}

                Provide a medical answer that includes:
                - Common symptoms and signs
                - When to seek medical attention
                - Risk factors
                - Brief medical explanation
                - Management tips

                Answer:"""
            else:
                # General medical question
                prompted_text = f"""As a medical expert, provide specific information about this medical topic.
                Focus on providing accurate, evidence-based medical information.

                Question: {text}

                Provide a clear, medical answer that includes:
                - Key information
                - Medical explanation
                - Important considerations
                - When to seek help
                - Management tips

                Answer:"""
            
            # Calculate appropriate length based on input
            input_length = len(self.tokenizer.encode(prompted_text))
            max_length = min(max_length, max(100, input_length // 2))
            min_length = min(50, max_length // 2)
            
            # Clear memory before generation
            if self.has_gpu:
                torch.cuda.empty_cache()
                gc.collect()
            
            # Generate summary with improved parameters
            summary = self.summarizer(
                prompted_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=5,
                length_penalty=1.5,
                early_stopping=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2
            )[0]['summary_text']
            
            # Post-process the summary
            if summary:
                # Remove any prompt artifacts
                summary = summary.replace("Answer:", "").strip()
                # Ensure the summary starts with a proper sentence
                if not summary[0].isupper():
                    summary = summary[0].upper() + summary[1:]
                
                # Remove any remaining prompt-like text
                summary = summary.replace("As a medical expert", "").strip()
                summary = summary.replace("Provide a clear, medical answer", "").strip()
                summary = summary.replace("Question:", "").strip()
                
                # Clean up any remaining artifacts
                summary = ' '.join(summary.split())  # Remove extra whitespace
            
            # Clear memory after generation
            if self.has_gpu:
                torch.cuda.empty_cache()
                gc.collect()
            
            if not summary or len(summary.strip()) < 10:
                return "I apologize, but I couldn't generate a proper medical response. Please try rephrasing your question."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "I apologize, but I encountered an error while generating the response. Please try again."
    
    def calculate_metrics(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate various metrics for the generated summary.
        
        Args:
            reference (str): Reference summary
            hypothesis (str): Generated summary
            
        Returns:
            Dict[str, float]: Dictionary of metric scores
        """
        try:
            # Calculate ROUGE scores
            rouge_scores = self.rouge_scorer.score(reference, hypothesis)
            
            # Calculate BLEU score
            reference_tokens = [reference.split()]
            hypothesis_tokens = hypothesis.split()
            bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=self.smooth)
            
            # Calculate semantic similarity
            embeddings1 = self.similarity_model.encode([reference])
            embeddings2 = self.similarity_model.encode([hypothesis])
            similarity = np.dot(embeddings1, embeddings2.T)[0][0]
            
            # Calculate word overlap
            reference_words = set(reference.lower().split())
            hypothesis_words = set(hypothesis.lower().split())
            word_overlap = len(reference_words.intersection(hypothesis_words)) / len(reference_words) if reference_words else 0
            
            # Clear embeddings from memory
            del embeddings1
            del embeddings2
            if self.has_gpu:
                torch.cuda.empty_cache()
            
            return {
                'rouge1': rouge_scores['rouge1'].fmeasure,
                'rouge2': rouge_scores['rouge2'].fmeasure,
                'rougeL': rouge_scores['rougeL'].fmeasure,
                'bleu': bleu_score,
                'semantic_similarity': similarity,
                'word_overlap': word_overlap
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'bleu': 0.0,
                'semantic_similarity': 0.0,
                'word_overlap': 0.0
            }
    
    def generate_summary_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """
        Generate summaries in batches for better GPU utilization.
        
        Args:
            texts (List[str]): List of input texts
            batch_size (int): Batch size for processing
            
        Returns:
            List[str]: List of generated summaries
        """
        summaries = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Create progress bar
        pbar = tqdm(total=len(texts), desc="Generating summaries")
        
        for i in range(0, len(texts), batch_size):
            try:
                batch_texts = texts[i:i + batch_size]
                batch_prompts = [self.create_prompt(text, "medical_qa") for text in batch_texts]
                
                # Calculate appropriate lengths for each prompt
                batch_max_lengths = []
                batch_min_lengths = []
                for prompt in batch_prompts:
                    input_length = len(self.tokenizer.encode(prompt))
                    max_len = min(100, max(30, input_length // 2))
                    min_len = min(20, max_len // 2)
                    batch_max_lengths.append(max_len)
                    batch_min_lengths.append(min_len)
                
                # Clear memory before batch processing
                if self.has_gpu:
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Process batch
                batch_summaries = self.summarizer(
                    batch_prompts,
                    max_length=max(batch_max_lengths),
                    min_length=min(batch_min_lengths),
                    do_sample=False,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    batch_size=batch_size
                )
                
                summaries.extend([s['summary_text'] for s in batch_summaries])
                
                # Update progress bar
                pbar.update(len(batch_texts))
                
                # Clear memory after batch
                if self.has_gpu:
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}/{total_batches}: {str(e)}")
                # Add empty strings for failed generations
                summaries.extend([''] * len(batch_texts))
                pbar.update(len(batch_texts))
        
        pbar.close()
        return summaries

    def run_experiment(self, test_data: pd.DataFrame, prompt_types: List[str] = ["medical_qa"]):
        """
        Run the experiment using batched processing.
        
        Args:
            test_data (pd.DataFrame): DataFrame containing test examples
            prompt_types (List[str]): List of prompt types (kept for compatibility)
        """
        try:
            # Sample 2,500 examples
            test_data = test_data.sample(n=2500, random_state=42)
            results = []
            total_examples = len(test_data)
            
            logger.info(f"Starting experiment with {total_examples} examples using Medical Q&A prompt")
            
            # Prepare batches
            texts = test_data['Question'].tolist()
            references = test_data['Answer'].tolist()
            sources = test_data.get('source', ['unknown'] * len(test_data)).tolist()
            topics = test_data.get('topic', ['unknown'] * len(test_data)).tolist()
            
            # Generate summaries in batches
            summaries = self.generate_summary_batch(texts, batch_size=8)
            
            # Calculate metrics in parallel with progress bar
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                metrics_futures = [
                    executor.submit(self.calculate_metrics, ref, hyp)
                    for ref, hyp in zip(references, summaries)
                ]
                
                # Create progress bar for metrics
                pbar = tqdm(total=len(metrics_futures), desc="Calculating metrics")
                
                for i, future in enumerate(metrics_futures):
                    try:
                        metrics = future.result()
                        results.append({
                            'rouge1': metrics['rouge1'],
                            'rouge2': metrics['rouge2'],
                            'rougeL': metrics['rougeL'],
                            'bleu': metrics['bleu'],
                            'semantic_similarity': metrics['semantic_similarity'],
                            'word_overlap': metrics['word_overlap'],
                            'source': sources[i],
                            'topic': topics[i]
                        })
                    except Exception as e:
                        logger.error(f"Error processing example {i}: {str(e)}")
                        results.append({
                            'rouge1': 0.0,
                            'rouge2': 0.0,
                            'rougeL': 0.0,
                            'bleu': 0.0,
                            'semantic_similarity': 0.0,
                            'word_overlap': 0.0,
                            'source': sources[i],
                            'topic': topics[i]
                        })
                    
                    pbar.update(1)
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"\nProcessed {i + 1}/{total_examples} examples")
                        logger.info(f"Question: {texts[i][:100]}...")
                        logger.info(f"Generated Answer: {summaries[i][:100]}...")
                        logger.info(f"Reference Answer: {references[i][:100]}...")
                        logger.info(f"Metrics: {metrics}\n")
                
                pbar.close()
            
            # Convert results to DataFrame
            self.results_df = pd.DataFrame(results)
            
            # Calculate average scores (only for numeric columns)
            numeric_cols = self.results_df.select_dtypes(include='number').columns
            self.average_scores = self.results_df[numeric_cols].mean()
            
            logger.info("\nExperiment completed!")
            logger.info("\nAverage Scores:")
            logger.info(self.average_scores)
            
            return self.average_scores
            
        except Exception as e:
            logger.error(f"Error in run_experiment: {str(e)}")
            raise
    
    def plot_results(self, save_path: str = "results.png"):
        """
        Plot the results of the experiment.
        
        Args:
            save_path (str): Path to save the plot
        """
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: ROUGE scores
        rouge_data = self.results_df.melt(value_vars=['rouge1', 'rouge2', 'rougeL'])
        sns.barplot(data=rouge_data,
                   x='variable',
                   y='value',
                   ax=axes[0])
        axes[0].set_title('ROUGE Scores')
        axes[0].set_xlabel('Metric')
        axes[0].set_ylabel('Score')
        
        # Plot 2: Other metrics
        other_metrics = self.results_df.melt(value_vars=['bleu', 'semantic_similarity', 'word_overlap'])
        sns.barplot(data=other_metrics,
                   x='variable',
                   y='value',
                   ax=axes[1])
        axes[1].set_title('BLEU, Semantic Similarity, and Word Overlap Scores')
        axes[1].set_xlabel('Metric')
        axes[1].set_ylabel('Score')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        # Create separate plots for each topic
        for topic in self.results_df['topic'].unique():
            topic_plot_path = f"results_{topic.replace(' ', '_')}.png"
            fig, axes = plt.subplots(2, 1, figsize=(15, 12))
            
            topic_data = self.results_df[self.results_df['topic'] == topic]
            
            # Plot ROUGE scores
            rouge_data = topic_data.melt(value_vars=['rouge1', 'rouge2', 'rougeL'])
            sns.barplot(data=rouge_data,
                       x='variable',
                       y='value',
                       ax=axes[0])
            axes[0].set_title(f'ROUGE Scores for {topic}')
            axes[0].set_xlabel('Metric')
            axes[0].set_ylabel('Score')
            
            # Plot other metrics
            other_metrics = topic_data.melt(value_vars=['bleu', 'semantic_similarity', 'word_overlap'])
            sns.barplot(data=other_metrics,
                       x='variable',
                       y='value',
                       ax=axes[1])
            axes[1].set_title(f'Other Metrics for {topic}')
            axes[1].set_xlabel('Metric')
            axes[1].set_ylabel('Score')
            
            plt.tight_layout()
            plt.savefig(topic_plot_path)
            plt.close()
    
    def save_results(self, output_path: str = "experiment_results.csv"):
        """
        Save the experiment results to a CSV file.
        
        Args:
            output_path (str): Path to save the results
        """
        self.results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    def __del__(self):
        """
        Cleanup when the object is destroyed.
        """
        try:
            # Clear CUDA cache
            if self.has_gpu:
                torch.cuda.empty_cache()
            
            # Clear model and tokenizer
            del self.model
            del self.tokenizer
            del self.summarizer
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Cleaned up model resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def main():
    # Initialize the system
    system = MedicalPromptEngineering()
    
    # Load and preprocess data
    if not system.load_and_preprocess_data():
        logger.error("Failed to load and preprocess data. Exiting...")
        return
    
    # Get the combined dataset
    test_data = system.data_loader.combined_data
    
    # Define topics of interest
    target_topics = [
        'Cancer',
        'Heart Disease',
        'Diabetes',
        'Neurological Disorders',
        'Genetic Disorders'
    ]
    
    # Filter data for specific topics
    filtered_data = test_data[test_data['topic'].str.contains('|'.join(target_topics), case=False, na=False)]
    
    # Sample 2,500 examples
    filtered_data = filtered_data.sample(n=2500, random_state=42)
    
    logger.info(f"\nFiltered dataset statistics:")
    logger.info(f"Total examples: {len(filtered_data)}")
    logger.info(f"Topics distribution:")
    logger.info(filtered_data['topic'].value_counts())
    
    # Run experiment with filtered data
    results = system.run_experiment(filtered_data)
    
    # Plot and save results
    system.plot_results()
    system.save_results()
    
    # Additional analysis by topic
    numeric_cols = system.results_df.select_dtypes(include='number').columns
    topic_results = system.results_df.groupby(['topic'])[numeric_cols].mean()
    logger.info("\nResults by Topic:")
    logger.info(topic_results)
    
    # Save topic-specific results
    topic_results.to_csv("topic_results.csv")
    logger.info("Topic-specific results saved to topic_results.csv")

if __name__ == "__main__":
    main() 