#!/usr/bin/env python3
"""
gradio_extensions.py

Advanced extensions and utilities for the Gradio UI
Includes data visualization, analytics, and enhanced export features.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class GradioAnalytics:
    """Analytics and visualization tools for the Gradio interface"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        
    def load_session_data(self) -> List[Dict]:
        """Load all session data from saved files"""
        session_files = list(self.results_dir.glob("gradio_sessions/*.json"))
        all_data = []
        
        for file_path in session_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
        
        return all_data
    
    def load_batch_data(self) -> List[Dict]:
        """Load all batch processing data"""
        batch_dirs = list(self.results_dir.glob("batch_runs/batch_*"))
        all_data = []
        
        for batch_dir in batch_dirs:
            results_file = batch_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        else:
                            all_data.append(data)
                except Exception as e:
                    logger.warning(f"Could not load {results_file}: {e}")
        
        return all_data
    
    def create_success_rate_chart(self) -> go.Figure:
        """Create a success rate visualization"""
        all_data = self.load_session_data() + self.load_batch_data()
        
        if not all_data:
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No data available yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Process data by date
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Calculate daily success rates
        daily_stats = df.groupby('date').agg({
            'success': ['count', 'sum']
        }).round(2)
        
        daily_stats.columns = ['total_tasks', 'successful_tasks']
        daily_stats['success_rate'] = (daily_stats['successful_tasks'] / daily_stats['total_tasks'] * 100).round(1)
        daily_stats = daily_stats.reset_index()
        
        # Create chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['success_rate'],
            mode='lines+markers',
            name='Success Rate (%)',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Daily Success Rate Trend',
            xaxis_title='Date',
            yaxis_title='Success Rate (%)',
            yaxis=dict(range=[0, 100]),
            hovermode='x unified'
        )
        
        return fig
    
    def create_task_type_distribution(self) -> go.Figure:
        """Create task type distribution chart"""
        all_data = self.load_session_data() + self.load_batch_data()
        
        if not all_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Extract task types from task descriptions or use default
        task_types = []
        for item in all_data:
            task = item.get('task', '').lower()
            if 'extract' in task:
                task_types.append('extract')
            elif 'interact' in task or 'fill' in task or 'click' in task:
                task_types.append('interact')
            elif 'search' in task or 'find' in task:
                task_types.append('search')
            else:
                task_types.append('other')
        
        # Count task types
        task_counts = pd.Series(task_types).value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=task_counts.index,
            values=task_counts.values,
            hole=0.3,
            textinfo='label+percent+value'
        )])
        
        fig.update_layout(
            title='Task Type Distribution',
            annotations=[dict(text='Tasks', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig
    
    def create_performance_metrics(self) -> Dict[str, Any]:
        """Generate performance metrics summary"""
        session_data = self.load_session_data()
        batch_data = self.load_batch_data()
        all_data = session_data + batch_data
        
        if not all_data:
            return {
                'total_tasks': 0,
                'success_rate': 0,
                'avg_elements_extracted': 0,
                'total_sessions': 0,
                'total_batch_runs': 0
            }
        
        total_tasks = len(all_data)
        successful_tasks = sum(1 for item in all_data if item.get('success', False))
        success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Calculate average elements extracted
        elements_counts = []
        for item in all_data:
            extracted_data = item.get('extracted_data', {})
            if isinstance(extracted_data, dict) and 'elements' in extracted_data:
                elements_counts.append(len(extracted_data['elements']))
        
        avg_elements = sum(elements_counts) / len(elements_counts) if elements_counts else 0
        
        # Count sessions and batch runs
        session_files = list(self.results_dir.glob("gradio_sessions/*.json"))
        batch_dirs = list(self.results_dir.glob("batch_runs/batch_*"))
        
        return {
            'total_tasks': total_tasks,
            'success_rate': round(success_rate, 1),
            'avg_elements_extracted': round(avg_elements, 1),
            'total_sessions': len(session_files),
            'total_batch_runs': len(batch_dirs)
        }
    
    def export_analytics_report(self, output_path: str = None) -> str:
        """Export comprehensive analytics report"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/analytics_report_{timestamp}.json"
        
        # Gather all analytics data
        metrics = self.create_performance_metrics()
        session_data = self.load_session_data()
        batch_data = self.load_batch_data()
        
        # Create comprehensive report
        report = {
            'report_generated': datetime.now().isoformat(),
            'summary_metrics': metrics,
            'session_history': {
                'total_sessions': len(session_data),
                'recent_sessions': session_data[-10:] if session_data else []
            },
            'batch_history': {
                'total_batches': len(batch_data),
                'recent_batches': batch_data[-10:] if batch_data else []
            },
            'data_quality': {
                'sessions_with_errors': len([s for s in session_data if s.get('error')]),
                'batches_with_errors': len([b for b in batch_data if b.get('error')]),
                'most_common_errors': self._get_common_errors(session_data + batch_data)
            }
        }
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_path
    
    def _get_common_errors(self, data: List[Dict]) -> List[Dict]:
        """Extract and count common errors"""
        errors = [item.get('error', '') for item in data if item.get('error')]
        error_counts = {}
        
        for error in errors:
            # Simplify error messages for grouping
            simplified = error.split(':')[0] if ':' in error else error
            simplified = simplified[:50]  # Truncate long errors
            error_counts[simplified] = error_counts.get(simplified, 0) + 1
        
        # Return top 5 errors
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'error': error, 'count': count} for error, count in sorted_errors[:5]]

class GradioDataExporter:
    """Advanced data export utilities"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
    
    def export_to_excel(self, data: List[Dict], output_path: str) -> str:
        """Export data to Excel with multiple sheets"""
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill
        except ImportError:
            raise ImportError("openpyxl required for Excel export. Install with: pip install openpyxl")
        
        if not data:
            raise ValueError("No data to export")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Tasks', 'Successful Tasks', 'Failed Tasks', 'Success Rate'],
                'Value': [
                    len(df),
                    len(df[df['success'] == True]),
                    len(df[df['success'] == False]),
                    f"{len(df[df['success'] == True]) / len(df) * 100:.1f}%"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # URL analysis sheet
            if 'url' in df.columns:
                url_stats = df.groupby('url').agg({
                    'success': ['count', 'sum'],
                    'task': 'first'
                }).round(2)
                url_stats.columns = ['total_attempts', 'successful_attempts', 'sample_task']
                url_stats['success_rate'] = (url_stats['successful_attempts'] / url_stats['total_attempts'] * 100).round(1)
                url_stats = url_stats.reset_index()
                url_stats.to_excel(writer, sheet_name='URL_Analysis', index=False)
        
        return output_path
    
    def create_csv_with_metadata(self, data: List[Dict], output_path: str) -> str:
        """Create CSV with embedded metadata"""
        if not data:
            raise ValueError("No data to export")
        
        # Create main DataFrame
        df = pd.DataFrame(data)
        
        # Create metadata
        metadata = [
            f"# Playwright LangGraph Agent Export",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Total Records: {len(df)}",
            f"# Success Rate: {len(df[df['success'] == True]) / len(df) * 100:.1f}%",
            f"# Columns: {', '.join(df.columns)}",
            "#"
        ]
        
        # Write file with metadata header
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            # Write metadata as comments
            for line in metadata:
                f.write(line + '\n')
            
            # Write CSV data
            df.to_csv(f, index=False)
        
        return output_path

def create_advanced_analytics_tab():
    """Create an advanced analytics tab for the Gradio interface"""
    import gradio as gr
    
    analytics = GradioAnalytics()
    exporter = GradioDataExporter()
    
    def generate_analytics():
        """Generate analytics visualizations"""
        try:
            success_chart = analytics.create_success_rate_chart()
            task_chart = analytics.create_task_type_distribution()
            metrics = analytics.create_performance_metrics()
            
            metrics_text = f"""
📊 **Performance Metrics**

🎯 **Total Tasks Executed**: {metrics['total_tasks']}
✅ **Success Rate**: {metrics['success_rate']}%
📄 **Avg Elements Extracted**: {metrics['avg_elements_extracted']}
💾 **Total Sessions**: {metrics['total_sessions']}
📦 **Total Batch Runs**: {metrics['total_batch_runs']}
            """.strip()
            
            return success_chart, task_chart, metrics_text
            
        except Exception as e:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text=f"Error generating analytics: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return empty_fig, empty_fig, f"❌ Error: {str(e)}"
    
    def export_analytics_report():
        """Export analytics report"""
        try:
            report_path = analytics.export_analytics_report()
            return f"✅ Analytics report exported to: {report_path}", report_path
        except Exception as e:
            return f"❌ Export failed: {str(e)}", ""
    
    with gr.Tab("📈 Analytics & Reports"):
        gr.Markdown("## 📊 Advanced Analytics Dashboard")
        
        with gr.Row():
            refresh_analytics_btn = gr.Button("🔄 Refresh Analytics", variant="primary")
            export_report_btn = gr.Button("📤 Export Report", variant="secondary")
        
        with gr.Row():
            with gr.Column():
                success_rate_plot = gr.Plot(label="📈 Success Rate Trend")
            with gr.Column():
                task_type_plot = gr.Plot(label="🍕 Task Type Distribution")
        
        metrics_display = gr.Markdown("Click 'Refresh Analytics' to load metrics")
        
        with gr.Row():
            export_status = gr.Textbox(label="📁 Export Status", interactive=False)
            export_path = gr.Textbox(label="📁 Export Path", interactive=False)
        
        # Event handlers
        refresh_analytics_btn.click(
            fn=generate_analytics,
            outputs=[success_rate_plot, task_type_plot, metrics_display]
        )
        
        export_report_btn.click(
            fn=export_analytics_report,
            outputs=[export_status, export_path]
        )
    
    return gr.Tab  # Return the tab for integration

if __name__ == "__main__":
    # Demo/test the analytics functionality
    analytics = GradioAnalytics()
    
    print("🔍 Testing Analytics Components...")
    
    # Test metrics
    metrics = analytics.create_performance_metrics()
    print(f"📊 Performance Metrics: {metrics}")
    
    # Test report export
    try:
        report_path = analytics.export_analytics_report()
        print(f"📤 Report exported to: {report_path}")
    except Exception as e:
        print(f"❌ Report export failed: {e}")
    
    print("✅ Analytics components tested successfully!")
