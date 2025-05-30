import pandas as pd
import os
from pathlib import Path
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalQADataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        self.combined_data = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        """Preprocess text using NLP techniques."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_medical_terms(self, text):
        """Special preprocessing for medical terms."""
        if not isinstance(text, str):
            return ""
            
        # Keep medical abbreviations and terms
        text = re.sub(r'[^\w\s-]', '', text)  # Keep hyphens for compound terms
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def load_all_datasets(self):
        """Load all CSV files from the data directory."""
        try:
            # Get all CSV files
            csv_files = list(self.data_dir.glob('*.csv'))
            logger.info(f"Found {len(csv_files)} CSV files")
            
            # Load each dataset
            for file_path in csv_files:
                dataset_name = file_path.stem
                logger.info(f"Loading {dataset_name}...")
                
                try:
                    # Read CSV file
                    df = pd.read_csv(file_path)
                    
                    # Preprocess text columns
                    text_columns = df.select_dtypes(include=['object']).columns
                    for col in text_columns:
                        if 'question' in col.lower() or 'answer' in col.lower():
                            df[f'{col}_processed'] = df[col].apply(self.preprocess_text)
                            df[f'{col}_medical'] = df[col].apply(self.preprocess_medical_terms)
                    
                    self.datasets[dataset_name] = df
                    logger.info(f"Successfully loaded and preprocessed {dataset_name} with {len(df)} rows")
                    
                    # Display basic information
                    logger.info(f"\nDataset: {dataset_name}")
                    logger.info(f"Shape: {df.shape}")
                    logger.info(f"Columns: {df.columns.tolist()}")
                    logger.info(f"Sample processed data:\n{df.head(2)}\n")
                    
                except Exception as e:
                    logger.error(f"Error loading {dataset_name}: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in load_all_datasets: {str(e)}")
            return False
    
    def combine_datasets(self):
        """Combine all datasets into a single DataFrame."""
        try:
            if not self.datasets:
                logger.error("No datasets loaded. Please load datasets first.")
                return False
            
            # Combine all datasets
            dfs = []
            for name, df in self.datasets.items():
                # Add source column
                df['source'] = name
                dfs.append(df)
            
            self.combined_data = pd.concat(dfs, ignore_index=True)
            logger.info(f"Successfully combined {len(self.datasets)} datasets")
            logger.info(f"Total rows: {len(self.combined_data)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error combining datasets: {str(e)}")
            return False
    
    def get_dataset_summary(self):
        """Get summary statistics for all datasets."""
        if not self.datasets:
            return "No datasets loaded"
        
        summary = {}
        for name, df in self.datasets.items():
            summary[name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'missing_values': df.isnull().sum().to_dict(),
                'processed_columns': [col for col in df.columns if 'processed' in col or 'medical' in col]
            }
        
        return summary
    
    def get_common_terms(self, column_name, top_n=20):
        """Get most common terms in a specific column across all datasets."""
        if not self.combined_data is not None:
            logger.error("No combined data available. Please load and combine datasets first.")
            return None
        
        try:
            # Combine all text from the specified column
            all_text = ' '.join(self.combined_data[column_name].dropna().astype(str))
            
            # Tokenize and count terms
            tokens = word_tokenize(all_text.lower())
            term_freq = pd.Series(tokens).value_counts()
            
            return term_freq.head(top_n)
            
        except Exception as e:
            logger.error(f"Error getting common terms: {str(e)}")
            return None
    
    def save_preprocessed_data(self, output_dir='preprocessed_data'):
        """Save preprocessed data to CSV files."""
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save individual datasets
            for name, df in self.datasets.items():
                output_file = output_path / f"{name}_preprocessed.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Saved preprocessed {name} to {output_file}")
            
            # Save combined dataset
            if self.combined_data is not None:
                combined_file = output_path / "combined_preprocessed.csv"
                self.combined_data.to_csv(combined_file, index=False)
                logger.info(f"Saved combined preprocessed data to {combined_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {str(e)}")
            return False
    
    def create_unified_dataset(self):
        """Create a unified dataset from all preprocessed data."""
        try:
            if not self.datasets:
                logger.error("No datasets loaded. Please load datasets first.")
                return None

            unified_data = []
            for name, df in self.datasets.items():
                # Create a new DataFrame with standardized columns
                unified_df = pd.DataFrame()
                
                # Map existing columns to standard columns
                if 'Question' in df.columns:
                    unified_df['question'] = df['Question']
                if 'Answer' in df.columns:
                    unified_df['answer'] = df['Answer']
                if 'Question_processed' in df.columns:
                    unified_df['question_processed'] = df['Question_processed']
                if 'Answer_processed' in df.columns:
                    unified_df['answer_processed'] = df['Answer_processed']
                if 'Question_medical' in df.columns:
                    unified_df['question_medical'] = df['Question_medical']
                if 'Answer_medical' in df.columns:
                    unified_df['answer_medical'] = df['Answer_medical']
                
                # Add metadata
                unified_df['source_dataset'] = name
                if 'topic' in df.columns:
                    unified_df['topic'] = df['topic']
                if 'split' in df.columns:
                    unified_df['split'] = df['split']
                
                unified_data.append(unified_df)
            
            # Combine all DataFrames
            final_unified = pd.concat(unified_data, ignore_index=True)
            
            # Save unified dataset
            output_path = Path('preprocessed_data')
            output_path.mkdir(exist_ok=True)
            unified_file = output_path / "unified_dataset.csv"
            final_unified.to_csv(unified_file, index=False)
            logger.info(f"Saved unified dataset to {unified_file}")
            
            # Print summary
            logger.info(f"\nUnified Dataset Summary:")
            logger.info(f"Total rows: {len(final_unified)}")
            logger.info(f"Columns: {final_unified.columns.tolist()}")
            logger.info(f"Sample data:\n{final_unified.head(2)}")
            
            return final_unified
            
        except Exception as e:
            logger.error(f"Error creating unified dataset: {str(e)}")
            return None

def main():
    # Initialize data loader
    loader = MedicalQADataLoader()
    
    # Load all datasets
    if loader.load_all_datasets():
        # Combine datasets
        if loader.combine_datasets():
            # Get and display summary
            summary = loader.get_dataset_summary()
            print("\nDataset Summary:")
            for name, stats in summary.items():
                print(f"\n{name}:")
                print(f"Rows: {stats['rows']}")
                print(f"Columns: {stats['columns']}")
                print(f"Processed columns: {stats['processed_columns']}")
                print(f"Missing values: {stats['missing_values']}")
            
            # Get common terms in processed columns
            print("\nCommon Terms in Processed Data:")
            for dataset_name, df in loader.datasets.items():
                processed_cols = [col for col in df.columns if 'processed' in col]
                for col in processed_cols:
                    print(f"\n{dataset_name} - {col}:")
                    common_terms = loader.get_common_terms(col)
                    if common_terms is not None:
                        print(common_terms)
            
            # Create and save unified dataset
            print("\nCreating unified dataset...")
            unified_data = loader.create_unified_dataset()
            if unified_data is not None:
                print("Successfully created unified dataset")
            else:
                print("Failed to create unified dataset")
            
            # Save preprocessed data
            print("\nSaving preprocessed data...")
            if loader.save_preprocessed_data():
                print("Successfully saved all preprocessed data to 'preprocessed_data' directory")
            else:
                print("Failed to save preprocessed data")
    else:
        print("Failed to load datasets. Please check the error messages above.")

if __name__ == "__main__":
    main() 