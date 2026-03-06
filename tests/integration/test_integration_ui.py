"""
Unit tests for UI module components.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.visualizations import VisualizationBuilder, create_visualization_from_result
from ui.export_utils import ResultExporter
from ui.ui_config import UIConfig, UITheme, ChatConfig, VisualizationConfig


class TestVisualizationBuilder:
    """Tests for VisualizationBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = VisualizationBuilder()
        
        # Create sample time series data
        base_time = datetime.now()
        self.time_series_data = [
            {
                'timestamp': (base_time - timedelta(hours=i)).isoformat(),
                'value': 20 + i * 0.5
            }
            for i in range(24)
        ]
        
        # Create sample aggregated data
        self.aggregated_data = [
            {'period': f'Day {i}', 'value': 20 + i}
            for i in range(7)
        ]
        
        # Create sample comparison data
        self.comparison_data = {
            'Node 14': {'mean': 22.5, 'min': 20.0, 'max': 25.0, 'std': 1.2},
            'Node 15': {'mean': 23.8, 'min': 21.5, 'max': 26.0, 'std': 1.1},
            'Node 16': {'mean': 21.2, 'min': 19.0, 'max': 23.5, 'std': 1.3}
        }
    
    def test_create_time_series(self):
        """Test time series chart creation."""
        fig = self.builder.create_time_series(
            data=self.time_series_data,
            sensor_type='temperature',
            location='Node 15',
            unit='°C'
        )
        
        assert fig is not None
        assert 'Temperature Over Time' in fig.layout.title.text
        assert len(fig.data) == 1
    
    def test_create_aggregated_series(self):
        """Test aggregated series chart creation."""
        fig = self.builder.create_aggregated_series(
            data=self.aggregated_data,
            sensor_type='temperature',
            location='Node 15',
            unit='°C',
            aggregation_level='daily'
        )
        
        assert fig is not None
        assert 'Daily Average Temperature' in fig.layout.title.text
    
    def test_create_comparison_chart(self):
        """Test comparison chart creation."""
        fig = self.builder.create_comparison_chart(
            comparison_data=self.comparison_data,
            sensor_type='temperature',
            unit='°C',
            metric='mean'
        )
        
        assert fig is not None
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 3  # 3 locations
    
    def test_create_statistical_summary(self):
        """Test statistical summary visualization."""
        statistics = {
            'min': 18.0,
            'max': 28.0,
            'mean': 23.0,
            'median': 22.5,
            '25%': 20.0,
            '75%': 26.0,
            'std': 2.1
        }
        
        fig = self.builder.create_statistical_summary(
            statistics=statistics,
            sensor_type='temperature',
            location='Node 15',
            unit='°C'
        )
        
        assert fig is not None
        assert 'Statistical Summary' in fig.layout.title.text


class TestResultExporter:
    """Tests for ResultExporter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = ResultExporter(output_dir=self.temp_dir)
        
        # Create sample state
        self.sample_state = {
            'user_query': 'What was the average temperature?',
            'success': True,
            'task_spec': {
                'intent_type': 'query',
                'sensor_type': 'temperature',
                'location': 'Node 15',
                'operation': 'mean'
            },
            'analytics_result': {
                'value': 23.5,
                'unit': '°C',
                'metadata': {
                    'operation': 'mean',
                    'sample_size': 100
                }
            },
            'explanation': 'The average temperature was 23.5°C',
            'execution_trace': [
                {
                    'step': 'interpret_query',
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat(),
                    'duration_ms': 150.0,
                    'details': {}
                }
            ]
        }
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_export_to_json(self):
        """Test JSON export."""
        filepath = self.exporter.export_to_json(self.sample_state)
        
        assert filepath.exists()
        assert filepath.suffix == '.json'
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert data['user_query'] == self.sample_state['user_query']
        assert data['success'] == True
    
    def test_export_to_csv(self):
        """Test CSV export."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
            'value': range(10),
            'unit': ['°C'] * 10
        })
        
        filepath = self.exporter.export_to_csv(df)
        
        assert filepath.exists()
        assert filepath.suffix == '.csv'
        
        # Verify content
        loaded_df = pd.read_csv(filepath)
        assert len(loaded_df) == 10
    
    def test_export_execution_trace(self):
        """Test execution trace export."""
        trace = self.sample_state['execution_trace']
        filepath = self.exporter.export_execution_trace(trace)
        
        assert filepath.exists()
        assert filepath.suffix == '.json'
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 1
        assert data[0]['step'] == 'interpret_query'
    
    def test_export_conversation(self):
        """Test conversation export."""
        messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'},
            {'role': 'user', 'content': 'What is the temperature?'},
        ]
        
        filepath = self.exporter.export_conversation(messages)
        
        assert filepath.exists()
        assert filepath.suffix == '.json'
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 3
        assert data[0]['role'] == 'user'


class TestUIConfig:
    """Tests for UIConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = UIConfig.default()
        
        assert config.page_title == "Smart Building Analytics"
        assert config.page_icon == "🏢"
        assert config.layout == "wide"
        assert isinstance(config.theme, UITheme)
        assert isinstance(config.chat, ChatConfig)
        assert isinstance(config.visualization, VisualizationConfig)
    
    def test_config_serialization(self):
        """Test configuration save/load."""
        config = UIConfig.default()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            config.to_yaml(temp_path)
            
            # Load
            loaded_config = UIConfig.from_yaml(temp_path)
            
            assert loaded_config.page_title == config.page_title
            assert loaded_config.theme.primary_color == config.theme.primary_color
            assert loaded_config.chat.max_message_length == config.chat.max_message_length
        finally:
            Path(temp_path).unlink()
    
    def test_custom_config(self):
        """Test custom configuration values."""
        custom_theme = UITheme(
            primary_color="#ff0000",
            background_color="#000000"
        )
        
        config = UIConfig(
            theme=custom_theme,
            chat=ChatConfig(),
            visualization=VisualizationConfig(),
            page_title="Custom Title"
        )
        
        assert config.page_title == "Custom Title"
        assert config.theme.primary_color == "#ff0000"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])