"""
Professional Results Manager for Deepfake Detection Framework
Organizes all outputs in a structured, professional manner
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd


class ResultsManager:
    """
    Manages and organizes all project outputs in a professional structure
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_directory_structure()
        
    def setup_directory_structure(self):
        """Create comprehensive directory structure for organized results"""
        
        # Main results directory
        self.results_dir = self.project_root / "results"
        
        # Session-specific results
        self.session_dir = self.results_dir / f"session_{self.session_id}"
        
        # Organized subdirectories
        self.directories = {
            'main_results': self.results_dir,
            'session': self.session_dir,
            'data_analysis': self.session_dir / "data_analysis",
            'visualizations': self.session_dir / "visualizations",
            'statistical_reports': self.session_dir / "statistical_reports", 
            'performance_metrics': self.session_dir / "performance_metrics",
            'bayesian_analysis': self.session_dir / "bayesian_analysis",
            'interactive_dashboards': self.session_dir / "interactive_dashboards",
            'detailed_logs': self.session_dir / "detailed_logs",
            'summary_reports': self.session_dir / "summary_reports",
            'raw_data': self.session_dir / "raw_data",
            'processed_data': self.session_dir / "processed_data"
        }
        
        # Create all directories
        for dir_path in self.directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Create README files for each directory
        self.create_directory_readmes()
        
    def create_directory_readmes(self):
        """Create README files explaining each directory's contents"""
        
        readme_contents = {
            'session': """# Analysis Session Results
            
This directory contains all results from a single analysis session.

## Directory Structure:
- `data_analysis/` - Core data analysis results and CSV files
- `visualizations/` - All generated plots and charts
- `statistical_reports/` - Statistical analysis reports
- `performance_metrics/` - System performance and processing metrics
- `bayesian_analysis/` - Bayesian inference results and reports
- `interactive_dashboards/` - Interactive HTML dashboards
- `detailed_logs/` - Comprehensive processing logs
- `summary_reports/` - Executive summary reports
- `raw_data/` - Original input data references
- `processed_data/` - Processed feature data
""",
            
            'data_analysis': """# Data Analysis Results
            
Contains core data analysis outputs:
- Physics features summaries
- Classification results
- Feature extraction outputs
""",
            
            'visualizations': """# Visualization Gallery
            
Contains all generated visualizations:
- Physics feature distribution plots
- Statistical comparison charts
- Performance analysis graphs
- Correlation matrices
""",
            
            'statistical_reports': """# Statistical Analysis Reports
            
Contains detailed statistical analysis:
- Feature discrimination analysis
- Statistical significance tests
- Distribution analysis reports
- Comparative statistics
""",
            
            'bayesian_analysis': """# Bayesian Analysis Results
            
Contains Bayesian inference outputs:
- Probabilistic classification results
- Uncertainty quantification
- Causal analysis reports
- Risk assessment summaries
""",
            
            'summary_reports': """# Executive Summary Reports
            
High-level summary reports for stakeholders:
- Executive summary
- Key findings report
- Recommendations summary
- Performance overview
"""
        }
        
        for dir_name, content in readme_contents.items():
            if dir_name in self.directories:
                readme_path = self.directories[dir_name] / "README.md"
                with open(readme_path, 'w') as f:
                    f.write(content.strip())
    
    def organize_existing_outputs(self):
        """Organize existing outputs from various locations into the structured format"""
        
        # Move/copy visualization files
        viz_source = self.project_root / "inference" / "visualization"
        if viz_source.exists():
            self._copy_visualizations(viz_source)
            
        # Move/copy results files
        results_source = self.project_root / "results"
        if results_source.exists():
            self._copy_results(results_source)
            
        # Move/copy logs
        logs_source = self.project_root / "scripts" / "logs"
        if logs_source.exists():
            self._copy_logs(logs_source)
            
        # Move/copy output files
        output_source = self.project_root / "scripts" / "output"
        if output_source.exists():
            self._copy_outputs(output_source)
    
    def _copy_visualizations(self, source_dir: Path):
        """Copy visualization files to organized structure"""
        
        # Copy static plots
        static_dir = source_dir / "static"
        if static_dir.exists():
            dest_dir = self.directories['visualizations'] / "static_plots"
            dest_dir.mkdir(exist_ok=True)
            for file in static_dir.glob("*.png"):
                shutil.copy2(file, dest_dir / file.name)
                
        # Copy physics analysis plots
        physics_dir = source_dir / "physics_analysis"
        if physics_dir.exists():
            dest_dir = self.directories['visualizations'] / "physics_analysis"
            dest_dir.mkdir(exist_ok=True)
            for file in physics_dir.glob("*.png"):
                shutil.copy2(file, dest_dir / file.name)
                
        # Copy interactive dashboards
        interactive_dir = source_dir / "interactive"
        if interactive_dir.exists():
            dest_dir = self.directories['interactive_dashboards']
            for file in interactive_dir.glob("*.html"):
                shutil.copy2(file, dest_dir / file.name)
                
        # Copy reports
        reports_dir = source_dir / "reports"
        if reports_dir.exists():
            for file in reports_dir.glob("*"):
                if file.suffix == '.json':
                    shutil.copy2(file, self.directories['statistical_reports'] / file.name)
                elif file.suffix == '.md':
                    shutil.copy2(file, self.directories['summary_reports'] / file.name)
                elif file.suffix == '.csv':
                    shutil.copy2(file, self.directories['data_analysis'] / file.name)
    
    def _copy_results(self, source_dir: Path):
        """Copy results files to organized structure"""
        
        for file in source_dir.glob("*.csv"):
            if "physics_features" in file.name:
                shutil.copy2(file, self.directories['data_analysis'] / file.name)
                # Also create a copy in the main results for quick access
                shutil.copy2(file, self.directories['main_results'] / f"latest_{file.name}")
    
    def _copy_logs(self, source_dir: Path):
        """Copy log files to organized structure"""
        
        dest_dir = self.directories['detailed_logs']
        for file in source_dir.rglob("*.log"):
            # Create subdirectory structure in logs
            rel_path = file.relative_to(source_dir)
            dest_file = dest_dir / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest_file)
    
    def _copy_outputs(self, source_dir: Path):
        """Copy output files to organized structure"""
        
        for file in source_dir.rglob("*"):
            if file.is_file():
                if file.suffix == '.json':
                    if "config" in file.name:
                        shutil.copy2(file, self.directories['raw_data'] / file.name)
                    else:
                        shutil.copy2(file, self.directories['processed_data'] / file.name)
    
    def create_executive_summary(self, results_data: Dict[str, Any]):
        """Create comprehensive executive summary"""
        
        summary = {
            "session_info": {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "project_version": "1.0.0",
                "analysis_type": "Physics-Based Deepfake Detection"
            },
            "data_overview": results_data.get('data_overview', {}),
            "performance_metrics": results_data.get('performance_metrics', {}),
            "key_findings": results_data.get('key_findings', []),
            "recommendations": results_data.get('recommendations', []),
            "file_locations": {
                "main_results": str(self.directories['data_analysis']),
                "visualizations": str(self.directories['visualizations']),
                "interactive_dashboard": str(self.directories['interactive_dashboards']),
                "detailed_reports": str(self.directories['statistical_reports'])
            }
        }
        
        # Save JSON summary
        json_path = self.directories['summary_reports'] / "executive_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Create Markdown summary
        self._create_markdown_summary(summary)
        
        # Create HTML summary
        self._create_html_summary(summary)
        
        return summary
    
    def _create_markdown_summary(self, summary: Dict[str, Any]):
        """Create executive summary in Markdown format"""
        
        md_content = f"""# Deepfake Detection Analysis Report
        
## Session Information
- **Session ID**: {summary['session_info']['session_id']}
- **Timestamp**: {summary['session_info']['timestamp']}
- **Analysis Type**: {summary['session_info']['analysis_type']}

## Data Overview
"""
        
        if 'data_overview' in summary and summary['data_overview']:
            for key, value in summary['data_overview'].items():
                md_content += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        md_content += "\n## Performance Metrics\n"
        if 'performance_metrics' in summary and summary['performance_metrics']:
            for key, value in summary['performance_metrics'].items():
                md_content += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        md_content += "\n## Key Findings\n"
        if 'key_findings' in summary:
            for i, finding in enumerate(summary['key_findings'], 1):
                md_content += f"{i}. {finding}\n"
        
        md_content += "\n## Recommendations\n"
        if 'recommendations' in summary:
            for i, rec in enumerate(summary['recommendations'], 1):
                md_content += f"{i}. {rec}\n"
        
        md_content += f"""
## File Locations

### Quick Access
- **Main Results**: `{summary['file_locations']['main_results']}`
- **Visualizations**: `{summary['file_locations']['visualizations']}`
- **Interactive Dashboard**: `{summary['file_locations']['interactive_dashboard']}`
- **Detailed Reports**: `{summary['file_locations']['detailed_reports']}`

### Complete Session Results
All session results are organized in: `{self.session_dir}`

---
*Generated by Deepfake Detection Framework v1.0.0*
"""
        
        md_path = self.directories['summary_reports'] / "executive_summary.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
    
    def _create_html_summary(self, summary: Dict[str, Any]):
        """Create executive summary in HTML format"""
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deepfake Detection Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 20px; border-left: 4px solid #3498db; background: #f8f9fa; }}
        .metric {{ background: white; padding: 10px; margin: 5px 0; border-radius: 3px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .file-path {{ font-family: monospace; background: #e9ecef; padding: 2px 4px; border-radius: 3px; }}
        ul {{ padding-left: 20px; }}
        li {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Deepfake Detection Analysis Report</h1>
        <p><strong>Session:</strong> {summary['session_info']['session_id']} | <strong>Generated:</strong> {summary['session_info']['timestamp']}</p>
    </div>
    
    <div class="section">
        <h2>📊 Data Overview</h2>
"""
        
        if 'data_overview' in summary:
            for key, value in summary['data_overview'].items():
                html_content += f'        <div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>\n'
        
        html_content += """    </div>
    
    <div class="section">
        <h2>⚡ Performance Metrics</h2>
"""
        
        if 'performance_metrics' in summary:
            for key, value in summary['performance_metrics'].items():
                html_content += f'        <div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>\n'
        
        html_content += """    </div>
    
    <div class="section">
        <h2>🔍 Key Findings</h2>
        <ul>
"""
        
        if 'key_findings' in summary:
            for finding in summary['key_findings']:
                html_content += f'            <li>{finding}</li>\n'
        
        html_content += """        </ul>
    </div>
    
    <div class="section">
        <h2>💡 Recommendations</h2>
        <ul>
"""
        
        if 'recommendations' in summary:
            for rec in summary['recommendations']:
                html_content += f'            <li>{rec}</li>\n'
        
        html_content += f"""        </ul>
    </div>
    
    <div class="section">
        <h2>📁 File Locations</h2>
        <div class="metric"><strong>Main Results:</strong> <span class="file-path">{summary['file_locations']['main_results']}</span></div>
        <div class="metric"><strong>Visualizations:</strong> <span class="file-path">{summary['file_locations']['visualizations']}</span></div>
        <div class="metric"><strong>Interactive Dashboard:</strong> <span class="file-path">{summary['file_locations']['interactive_dashboard']}</span></div>
        <div class="metric"><strong>Detailed Reports:</strong> <span class="file-path">{summary['file_locations']['detailed_reports']}</span></div>
        <div class="metric"><strong>Complete Session:</strong> <span class="file-path">{self.session_dir}</span></div>
    </div>
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666;">
        <p>Generated by Deepfake Detection Framework v1.0.0</p>
    </footer>
</body>
</html>
"""
        
        html_path = self.directories['summary_reports'] / "executive_summary.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
    
    def create_index_file(self):
        """Create an index file that links to all results"""
        
        index_content = f"""# Deepfake Detection Results Index
        
Session: {self.session_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Quick Navigation

### 📋 Summary Reports
- [Executive Summary (HTML)]({self.directories['summary_reports']}/executive_summary.html)
- [Executive Summary (Markdown)]({self.directories['summary_reports']}/executive_summary.md)
- [Executive Summary (JSON)]({self.directories['summary_reports']}/executive_summary.json)

### 📊 Data Analysis
- [Main Results CSV]({self.directories['data_analysis']})
- [Processed Data]({self.directories['processed_data']})

### 📈 Visualizations
- [Static Plots]({self.directories['visualizations']}/static_plots/)
- [Physics Analysis]({self.directories['visualizations']}/physics_analysis/)
- [Interactive Dashboards]({self.directories['interactive_dashboards']})

### 📊 Statistical Reports
- [Statistical Analysis]({self.directories['statistical_reports']})
- [Performance Metrics]({self.directories['performance_metrics']})

### 🧠 Bayesian Analysis
- [Bayesian Results]({self.directories['bayesian_analysis']})

### 📝 Detailed Logs
- [Processing Logs]({self.directories['detailed_logs']})

---

## Directory Structure
```
results/session_{self.session_id}/
├── summary_reports/          # Executive summaries and overviews
├── data_analysis/            # Core analysis results (CSV files)
├── visualizations/           # All plots and charts
├── statistical_reports/      # Statistical analysis reports
├── performance_metrics/      # System performance data
├── bayesian_analysis/        # Bayesian inference results
├── interactive_dashboards/   # Interactive HTML dashboards
├── detailed_logs/           # Comprehensive processing logs
├── raw_data/                # Original input data references
└── processed_data/          # Processed feature data
```
"""
        
        index_path = self.session_dir / "index.md"
        with open(index_path, 'w') as f:
            f.write(index_content)
            
        # Also create in main results for quick access
        main_index_path = self.directories['main_results'] / "latest_session_index.md"
        with open(main_index_path, 'w') as f:
            f.write(index_content)
    
    def get_session_directory(self) -> Path:
        """Get the current session directory"""
        return self.session_dir
    
    def get_directories(self) -> Dict[str, Path]:
        """Get all organized directories"""
        return self.directories 