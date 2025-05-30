import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResultsVisualizer:
    def __init__(self, results_path: str = "experiment_results.csv"):
        """
        Initialize the ResultsVisualizer.
        
        Args:
            results_path (str): Path to the experiment results CSV file
        """
        try:
            self.results_df = pd.read_csv(results_path)
            logger.info(f"Successfully loaded results from {results_path}")
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            raise

    def create_metric_correlation_heatmap(self, save_path: str = "metric_correlation_heatmap.png"):
        """
        Create a heatmap showing correlations between different metrics.
        
        Args:
            save_path (str): Path to save the heatmap
        """
        try:
            # Select numeric columns for correlation
            numeric_cols = self.results_df.select_dtypes(include='number').columns
            correlation_matrix = self.results_df[numeric_cols].corr()
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Create heatmap with simple styling
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'}
            )
            
            plt.title('Correlation Between Evaluation Metrics')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Metric correlation heatmap saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating metric correlation heatmap: {str(e)}")

    def create_topic_metric_heatmap(self, save_path: str = "topic_metric_heatmap.png"):
        """
        Create a heatmap showing average metric scores by topic.
        
        Args:
            save_path (str): Path to save the heatmap
        """
        try:
            # Calculate average scores by topic
            numeric_cols = self.results_df.select_dtypes(include='number').columns
            topic_metrics = self.results_df.groupby('topic')[numeric_cols].mean()
            
            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Create heatmap
            sns.heatmap(
                topic_metrics,
                annot=True,
                cmap='YlOrRd',
                fmt='.2f',
                cbar_kws={'label': 'Average Score'}
            )
            
            plt.title('Average Metric Scores by Medical Topic')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Topic-metric heatmap saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating topic-metric heatmap: {str(e)}")

    def create_source_metric_heatmap(self, save_path: str = "source_metric_heatmap.png"):
        """
        Create a heatmap showing average metric scores by data source.
        
        Args:
            save_path (str): Path to save the heatmap
        """
        try:
            # Calculate average scores by source
            numeric_cols = self.results_df.select_dtypes(include='number').columns
            source_metrics = self.results_df.groupby('source')[numeric_cols].mean()
            
            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Create heatmap
            sns.heatmap(
                source_metrics,
                annot=True,
                cmap='YlOrRd',
                fmt='.2f',
                cbar_kws={'label': 'Average Score'}
            )
            
            plt.title('Average Metric Scores by Data Source')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Source-metric heatmap saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating source-metric heatmap: {str(e)}")

    def create_topic_source_heatmap(self, metric: str, save_path: Optional[str] = None):
        """
        Create a heatmap showing average scores for a specific metric across topics and sources.
        
        Args:
            metric (str): Metric to visualize
            save_path (str, optional): Path to save the heatmap. If None, uses default name.
        """
        try:
            if save_path is None:
                save_path = f"topic_source_{metric}_heatmap.png"
            
            # Create pivot table
            pivot_data = self.results_df.pivot_table(
                values=metric,
                index='topic',
                columns='source',
                aggfunc='mean'
            )
            
            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Create heatmap
            sns.heatmap(
                pivot_data,
                annot=True,
                cmap='YlOrRd',
                fmt='.2f',
                cbar_kws={'label': f'Average {metric} Score'}
            )
            
            plt.title(f'Average {metric} Scores by Topic and Source')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Topic-source heatmap for {metric} saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating topic-source heatmap: {str(e)}")

    def create_summary_diagram(self, save_path: str = "experiment_summary.png"):
        """
        Create a comprehensive summary diagram showing all key findings from the experiment.
        
        Args:
            save_path (str): Path to save the summary diagram
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 25))
            gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])
            
            # 1. Overall Performance Summary (Top Left)
            ax1 = fig.add_subplot(gs[0, 0])
            numeric_cols = self.results_df.select_dtypes(include='number').columns
            overall_means = self.results_df[numeric_cols].mean()
            sns.barplot(x=overall_means.index, y=overall_means.values, ax=ax1)
            ax1.set_title('Overall Performance Across All Metrics')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
            ax1.set_ylim(0, 1)
            
            # 2. Topic Performance (Top Right)
            ax2 = fig.add_subplot(gs[0, 1])
            topic_means = self.results_df.groupby('topic')[numeric_cols].mean().mean(axis=1)
            sns.barplot(x=topic_means.index, y=topic_means.values, ax=ax2)
            ax2.set_title('Average Performance by Medical Topic')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            ax2.set_ylim(0, 1)
            
            # 3. Metric Correlations (Middle Left)
            ax3 = fig.add_subplot(gs[1, 0])
            correlation_matrix = self.results_df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
            ax3.set_title('Metric Correlations')
            
            # 4. Source Performance (Middle Right)
            ax4 = fig.add_subplot(gs[1, 1])
            source_means = self.results_df.groupby('source')[numeric_cols].mean().mean(axis=1)
            sns.barplot(x=source_means.index, y=source_means.values, ax=ax4)
            ax4.set_title('Average Performance by Data Source')
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
            ax4.set_ylim(0, 1)
            
            # 5. Topic-Source Interaction (Bottom Left)
            ax5 = fig.add_subplot(gs[2, 0])
            pivot_data = self.results_df.pivot_table(
                values=numeric_cols[0],  # Using first metric as example
                index='topic',
                columns='source',
                aggfunc='mean'
            )
            sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', ax=ax5)
            ax5.set_title(f'Topic-Source Interaction ({numeric_cols[0]})')
            
            # 6. Metric Distribution (Bottom Right)
            ax6 = fig.add_subplot(gs[2, 1])
            self.results_df[numeric_cols].boxplot(ax=ax6)
            ax6.set_title('Distribution of Metrics')
            ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
            
            # 7. Performance Trends (Bottom)
            ax7 = fig.add_subplot(gs[3, :])
            # Calculate rolling average of performance
            rolling_avg = self.results_df[numeric_cols].mean(axis=1).rolling(window=100).mean()
            sns.lineplot(data=rolling_avg, ax=ax7)
            ax7.set_title('Performance Trend (Rolling Average)')
            ax7.set_xlabel('Example Index')
            ax7.set_ylabel('Average Score')
            
            # Add summary statistics as text
            summary_text = f"""
            Experiment Summary:
            - Total Examples: {len(self.results_df)}
            - Number of Topics: {len(self.results_df['topic'].unique())}
            - Number of Sources: {len(self.results_df['source'].unique())}
            - Best Performing Topic: {topic_means.idxmax()} ({topic_means.max():.3f})
            - Best Performing Source: {source_means.idxmax()} ({source_means.max():.3f})
            - Most Correlated Metrics: {self._get_most_correlated_metrics(correlation_matrix)}
            """
            
            fig.text(0.5, 0.02, summary_text, ha='center', va='center', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8))
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Summary diagram saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating summary diagram: {str(e)}")
    
    def _get_most_correlated_metrics(self, correlation_matrix: pd.DataFrame) -> str:
        """
        Helper method to find the most correlated metrics.
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            
        Returns:
            str: Description of most correlated metrics
        """
        # Get upper triangle of correlation matrix
        upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        
        # Find the pair with highest correlation
        max_corr = upper.max().max()
        max_corr_pair = upper[upper == max_corr].stack().index[0]
        
        return f"{max_corr_pair[0]} & {max_corr_pair[1]} ({max_corr:.3f})"

    def create_all_heatmaps(self, output_dir: str = "heatmaps"):
        """
        Create all heatmaps and save them to the specified directory.
        
        Args:
            output_dir (str): Directory to save the heatmaps
        """
        try:
            # Create output directory if it doesn't exist
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Create all heatmaps
            self.create_metric_correlation_heatmap(
                os.path.join(output_dir, "metric_correlation_heatmap.png")
            )
            
            self.create_topic_metric_heatmap(
                os.path.join(output_dir, "topic_metric_heatmap.png")
            )
            
            self.create_source_metric_heatmap(
                os.path.join(output_dir, "source_metric_heatmap.png")
            )
            
            # Create topic-source heatmaps for each metric
            numeric_cols = self.results_df.select_dtypes(include='number').columns
            for metric in numeric_cols:
                self.create_topic_source_heatmap(
                    metric,
                    os.path.join(output_dir, f"topic_source_{metric}_heatmap.png")
                )
            
            # Create comprehensive summary diagram
            self.create_summary_diagram(
                os.path.join(output_dir, "experiment_summary.png")
            )
            
            logger.info(f"All visualizations saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating all visualizations: {str(e)}")

    def create_focused_metric_correlation(self, save_path: str = "metric_correlation_focused.png"):
        """
        Create a focused and well-styled metric correlation heatmap.
        
        Args:
            save_path (str): Path to save the heatmap
        """
        try:
            # Select numeric columns for correlation
            numeric_cols = self.results_df.select_dtypes(include='number').columns
            correlation_matrix = self.results_df[numeric_cols].corr()
            
            # Create figure with specific size
            plt.figure(figsize=(12, 10))
            
            # Create custom colormap
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
            
            # Create heatmap with improved styling
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(
                correlation_matrix,
                mask=mask,
                annot=True,
                cmap=cmap,
                center=0,
                fmt='.2f',
                square=True,
                linewidths=.5,
                cbar_kws={
                    'label': 'Correlation Coefficient',
                    'shrink': .8
                },
                annot_kws={
                    'size': 10,
                    'weight': 'bold'
                }
            )
            
            # Customize the plot
            plt.title('Metric Correlations in Medical Q&A Performance', pad=20, size=14, weight='bold')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right', size=10)
            plt.yticks(rotation=0, size=10)
            
            # Add a description
            plt.figtext(
                0.5, 0.01,
                'Correlation values range from -1 (perfect negative correlation) to 1 (perfect positive correlation)',
                ha='center',
                fontsize=10,
                style='italic'
            )
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Focused metric correlation heatmap saved to {save_path}")
            
            # Return the correlation matrix for reference
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error creating focused metric correlation heatmap: {str(e)}")
            return None

def main():
    # Initialize visualizer
    visualizer = ResultsVisualizer()
    
    # Create metric correlation heatmap
    visualizer.create_metric_correlation_heatmap()
    
    logger.info("Visualization complete!")

if __name__ == "__main__":
    main() 