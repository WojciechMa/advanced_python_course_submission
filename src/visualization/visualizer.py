"""Comprehensive visualization system for patient analysis results."""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Matplotlib backend setup (use Agg for non-GUI environments)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """
    Create visualizations for patient analysis results.
    
    Features:
    - Patient timelines (imaging dates + lab test dates)
    - Correlation heatmaps (predictions vs biomarkers)
    - Biomarker trend analysis over time
    - Distribution comparisons (normal vs abnormal ranges)
    - Interactive HTML reports
    - Multiple export formats (PNG, HTML, PDF)
    """
    
    def __init__(
        self,
        figure_size: Tuple[int, int] = (12, 8),
        style: str = 'seaborn-v0_8-darkgrid',
        color_palette: str = 'Set2'
    ):
        """
        Initialize visualizer.
        
        Args:
            figure_size: Default figure size (width, height) in inches
            style: Matplotlib style
            color_palette: Seaborn color palette
        """
        self.figure_size = figure_size
        self.style = style
        self.color_palette = color_palette
        
        # Set style
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not available, using default")
        
        sns.set_palette(color_palette)
        
        logger.info(f"ResultsVisualizer initialized: size={figure_size}, style={style}")
    
    def plot_patient_timeline(
        self,
        patient_report: Dict[str, Any],
        save_path: Optional[Path] = None,
        show_plot: bool = False
    ) -> Optional[plt.Figure]:
        """
        Create timeline visualization showing imaging and lab test dates.
        
        Args:
            patient_report: Patient report dictionary with predictions and correlations
            save_path: Path to save figure (None = don't save)
            show_plot: Whether to display plot
            
        Returns:
            Matplotlib figure or None
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Extract dates from predictions
        predictions = patient_report.get('predictions', [])
        pred_dates = []
        pred_labels = []
        
        for i, pred in enumerate(predictions):
            if 'study_date' in pred and pred['study_date']:
                try:
                    if isinstance(pred['study_date'], str):
                        date = pd.to_datetime(pred['study_date'])
                    else:
                        date = pred['study_date']
                    pred_dates.append(date)
                    pred_labels.append(f"{pred.get('prediction', 'Unknown')} ({pred.get('confidence', 0):.2f})")
                except:
                    pass
        
        # Extract dates from correlations (lab tests)
        correlations = patient_report.get('correlations', [])
        lab_dates = []
        lab_labels = []
        
        for corr in correlations:
            matched_labs = corr.get('matched_labs', [])
            for lab in matched_labs:
                if 'test_date' in lab and lab['test_date']:
                    try:
                        if isinstance(lab['test_date'], str):
                            date = pd.to_datetime(lab['test_date'])
                        else:
                            date = lab['test_date']
                        lab_dates.append(date)
                        lab_labels.append(f"{lab.get('test_name', 'Unknown')}: {lab.get('value', 0):.1f}")
                    except:
                        pass
        
        # Plot predictions
        if pred_dates:
            ax.scatter(pred_dates, [1]*len(pred_dates), c='red', s=100, 
                      label='Imaging Studies', marker='s', alpha=0.7, zorder=3)
            for date, label in zip(pred_dates, pred_labels):
                ax.annotate(label, (date, 1), xytext=(0, 10), 
                           textcoords='offset points', ha='center', fontsize=8, rotation=45)
        
        # Plot lab tests
        if lab_dates:
            ax.scatter(lab_dates, [0]*len(lab_dates), c='blue', s=100,
                      label='Lab Tests', marker='o', alpha=0.7, zorder=3)
            for date, label in zip(lab_dates, lab_labels):
                ax.annotate(label, (date, 0), xytext=(0, -10),
                           textcoords='offset points', ha='center', fontsize=8, rotation=45)
        
        # Formatting
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Lab Tests', 'Imaging'])
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title(f"Patient Timeline: {patient_report.get('patient_id', 'Unknown')}", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Timeline saved: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            return fig
        
        return fig
    
    def plot_correlation_heatmap(
        self,
        correlation_results: Dict[str, Any],
        save_path: Optional[Path] = None,
        show_plot: bool = False
    ) -> Optional[plt.Figure]:
        """
        Create heatmap of correlations between predictions and biomarkers.
        
        Args:
            correlation_results: Dictionary with biomarker correlation results
            save_path: Path to save figure
            show_plot: Whether to display plot
            
        Returns:
            Matplotlib figure or None
        """
        # Extract correlation coefficients
        biomarkers = []
        correlations = []
        p_values = []
        
        for biomarker, result in correlation_results.items():
            if 'correlation' in result and not np.isnan(result['correlation']):
                biomarkers.append(biomarker)
                correlations.append(result['correlation'])
                p_values.append(result.get('p_value', 1.0))
        
        if not biomarkers:
            logger.warning("No correlation data to plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, len(biomarkers) * 0.5)))
        
        # Create DataFrame for heatmap
        data = pd.DataFrame({
            'Correlation': correlations,
            'Biomarker': biomarkers
        })
        data = data.set_index('Biomarker')
        
        # Create heatmap
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   vmin=-1, vmax=1, cbar_kws={'label': 'Correlation Coefficient'},
                   ax=ax, linewidths=1, linecolor='gray')
        
        # Mark significant correlations
        for i, (biomarker, p_val) in enumerate(zip(biomarkers, p_values)):
            if p_val < 0.05:
                ax.text(1.5, i + 0.5, '*', ha='center', va='center', 
                       fontsize=16, fontweight='bold', color='black')
        
        ax.set_title('Prediction Confidence vs Biomarker Correlation\n(* = p < 0.05)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            return fig
        
        return fig
    
    def plot_biomarker_trends(
        self,
        lab_data: pd.DataFrame,
        biomarker_name: str,
        reference_range: Optional[Tuple[float, float]] = None,
        save_path: Optional[Path] = None,
        show_plot: bool = False
    ) -> Optional[plt.Figure]:
        """
        Plot biomarker values over time with reference ranges.
        
        Args:
            lab_data: DataFrame with lab test data (must have 'test_date', 'value')
            biomarker_name: Name of biomarker to plot
            reference_range: Tuple of (lower, upper) reference values
            save_path: Path to save figure
            show_plot: Whether to display plot
            
        Returns:
            Matplotlib figure or None
        """
        # Filter for specific biomarker
        biomarker_data = lab_data[lab_data['test_name'] == biomarker_name].copy()
        
        if biomarker_data.empty:
            logger.warning(f"No data for biomarker: {biomarker_name}")
            return None
        
        # Ensure test_date is datetime
        biomarker_data['test_date'] = pd.to_datetime(biomarker_data['test_date'])
        biomarker_data = biomarker_data.sort_values('test_date')
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot trend line
        ax.plot(biomarker_data['test_date'], biomarker_data['value'],
               marker='o', linewidth=2, markersize=8, label=biomarker_name, color='navy')
        
        # Add reference range if provided
        if reference_range:
            ax.axhline(y=reference_range[0], color='green', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label='Normal Range')
            ax.axhline(y=reference_range[1], color='green', linestyle='--', 
                      linewidth=1.5, alpha=0.7)
            ax.fill_between(biomarker_data['test_date'], reference_range[0], 
                           reference_range[1], alpha=0.1, color='green')
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f"{biomarker_name} Value", fontsize=12)
        ax.set_title(f'{biomarker_name} Trend Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trend plot saved: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            return fig
        
        return fig
    
    def plot_prediction_distribution(
        self,
        predictions: List[Dict[str, Any]],
        save_path: Optional[Path] = None,
        show_plot: bool = False
    ) -> Optional[plt.Figure]:
        """
        Plot distribution of predictions across classes.
        
        Args:
            predictions: List of prediction dictionaries
            save_path: Path to save figure
            show_plot: Whether to display plot
            
        Returns:
            Matplotlib figure or None
        """
        if not predictions:
            logger.warning("No predictions to plot")
            return None
        
        # Count predictions by class
        pred_classes = [p.get('prediction', 'Unknown') for p in predictions]
        class_counts = pd.Series(pred_classes).value_counts()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot
        colors = sns.color_palette(self.color_palette, len(class_counts))
        class_counts.plot(kind='bar', ax=ax1, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_title('Prediction Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Pie chart
        ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 11})
        ax2.set_title('Prediction Proportions', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plot saved: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            return fig
        
        return fig
    
    def plot_confidence_distribution(
        self,
        predictions: List[Dict[str, Any]],
        by_class: bool = True,
        save_path: Optional[Path] = None,
        show_plot: bool = False
    ) -> Optional[plt.Figure]:
        """
        Plot distribution of prediction confidence scores.
        
        Args:
            predictions: List of prediction dictionaries
            by_class: Whether to separate by prediction class
            save_path: Path to save figure
            show_plot: Whether to display plot
            
        Returns:
            Matplotlib figure or None
        """
        if not predictions:
            logger.warning("No predictions to plot")
            return None
        
        # Create DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        if by_class and 'prediction' in pred_df.columns:
            # Box plot by class
            pred_df.boxplot(column='confidence', by='prediction', ax=ax)
            ax.set_title('Confidence Distribution by Class', fontsize=14, fontweight='bold')
            ax.set_xlabel('Prediction Class', fontsize=12)
            ax.set_ylabel('Confidence Score', fontsize=12)
            plt.suptitle('')  # Remove default title
        else:
            # Histogram
            ax.hist(pred_df['confidence'], bins=20, edgecolor='black', 
                   color='skyblue', alpha=0.7)
            ax.set_title('Overall Confidence Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Confidence Score', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.axvline(pred_df['confidence'].mean(), color='red', 
                      linestyle='--', linewidth=2, label=f"Mean: {pred_df['confidence'].mean():.2f}")
            ax.legend()
        
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confidence plot saved: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            return fig
        
        return fig
    
    def generate_summary_report(
        self,
        patient_report: Dict[str, Any],
        correlation_results: Dict[str, Any],
        output_dir: Path,
        format: str = 'png'
    ) -> Dict[str, Path]:
        """
        Generate comprehensive summary report with all visualizations.
        
        Args:
            patient_report: Patient report dictionary
            correlation_results: Correlation analysis results
            output_dir: Directory to save visualizations
            format: Output format ('png', 'pdf', 'svg')
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        patient_id = patient_report.get('patient_id', 'unknown')
        saved_plots = {}
        
        # 1. Patient timeline
        timeline_path = output_dir / f'{patient_id}_timeline.{format}'
        self.plot_patient_timeline(patient_report, save_path=timeline_path)
        saved_plots['timeline'] = timeline_path
        
        # 2. Correlation heatmap
        if correlation_results:
            heatmap_path = output_dir / f'{patient_id}_correlations.{format}'
            self.plot_correlation_heatmap(correlation_results, save_path=heatmap_path)
            saved_plots['correlations'] = heatmap_path
        
        # 3. Prediction distribution
        predictions = patient_report.get('predictions', [])
        if predictions:
            dist_path = output_dir / f'{patient_id}_predictions.{format}'
            self.plot_prediction_distribution(predictions, save_path=dist_path)
            saved_plots['prediction_distribution'] = dist_path
            
            conf_path = output_dir / f'{patient_id}_confidence.{format}'
            self.plot_confidence_distribution(predictions, save_path=conf_path)
            saved_plots['confidence_distribution'] = conf_path
        
        logger.info(f"Generated {len(saved_plots)} visualizations for patient {patient_id}")
        return saved_plots
    
    def create_html_report(
        self,
        patient_report: Dict[str, Any],
        correlation_results: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """
        Create interactive HTML report.
        
        Args:
            patient_report: Patient report dictionary
            correlation_results: Correlation analysis results
            output_path: Path for HTML file
            
        Returns:
            Path to created HTML file
        """
        patient_id = patient_report.get('patient_id', 'Unknown')
        summary = patient_report.get('summary', {})
        predictions = patient_report.get('predictions', [])
        
        # Build HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Patient Report: {patient_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 8px; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .summary {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-label {{ font-weight: bold; color: #7f8c8d; }}
                .metric-value {{ font-size: 1.2em; color: #2c3e50; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .high-confidence {{ color: #27ae60; font-weight: bold; }}
                .low-confidence {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Patient Analysis Report: {patient_id}</h1>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <div class="metric">
                        <span class="metric-label">Images Analyzed:</span>
                        <span class="metric-value">{summary.get('num_images', 0)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Predictions:</span>
                        <span class="metric-value">{summary.get('num_predictions', 0)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Lab Tests:</span>
                        <span class="metric-value">{summary.get('num_lab_tests', 0)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Correlations:</span>
                        <span class="metric-value">{summary.get('num_correlations', 0)}</span>
                    </div>
                </div>
                
                <h2>Predictions</h2>
                <table>
                    <tr>
                        <th>#</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>Study Date</th>
                    </tr>
        """
        
        for i, pred in enumerate(predictions, 1):
            confidence = pred.get('confidence', 0)
            conf_class = 'high-confidence' if confidence > 0.8 else ('low-confidence' if confidence < 0.6 else '')
            html_content += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{pred.get('prediction', 'Unknown')}</td>
                        <td class="{conf_class}">{confidence:.2%}</td>
                        <td>{pred.get('study_date', 'N/A')}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <p style="margin-top: 40px; text-align: center; color: #7f8c8d;">
                    Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
                </p>
            </div>
        </body>
        </html>
        """
        
        output_path = Path(output_path)
        output_path.write_text(html_content, encoding='utf-8')
        logger.info(f"HTML report created: {output_path}")
        
        return output_path
