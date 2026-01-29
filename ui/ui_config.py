"""
UI configuration settings.
"""

from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class UITheme:
    """UI theme configuration."""
    primary_color: str = "#1f77b4"
    background_color: str = "#ffffff"
    secondary_background_color: str = "#f0f2f6"
    text_color: str = "#262730"
    font: str = "sans serif"


@dataclass
class ChatConfig:
    """Chat interface configuration."""
    max_message_length: int = 2000
    show_timestamps: bool = True
    enable_markdown: bool = True
    avatar_user: str = "👤"
    avatar_assistant: str = "🤖"


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    default_height: int = 400
    default_width: Optional[int] = None
    template: str = "plotly_white"
    enable_animation: bool = False
    color_scheme: str = "blues"


@dataclass
class UIConfig:
    """Complete UI configuration."""
    theme: UITheme
    chat: ChatConfig
    visualization: VisualizationConfig
    
    # General settings
    page_title: str = "Smart Building Analytics"
    page_icon: str = "🏢"
    layout: str = "wide"
    sidebar_state: str = "expanded"
    
    # Feature flags
    enable_execution_trace: bool = True
    enable_export: bool = True
    enable_download: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> 'UIConfig':
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            return cls.default()
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(
            theme=UITheme(**config_data.get('theme', {})),
            chat=ChatConfig(**config_data.get('chat', {})),
            visualization=VisualizationConfig(**config_data.get('visualization', {})),
            **{k: v for k, v in config_data.items() 
               if k not in ['theme', 'chat', 'visualization']}
        )
    
    @classmethod
    def default(cls) -> 'UIConfig':
        """Create default configuration."""
        return cls(
            theme=UITheme(),
            chat=ChatConfig(),
            visualization=VisualizationConfig()
        )
    
    def to_yaml(self, output_path: str | Path):
        """Save configuration to YAML file."""
        config_dict = {
            'page_title': self.page_title,
            'page_icon': self.page_icon,
            'layout': self.layout,
            'sidebar_state': self.sidebar_state,
            'enable_execution_trace': self.enable_execution_trace,
            'enable_export': self.enable_export,
            'enable_download': self.enable_download,
            'theme': {
                'primary_color': self.theme.primary_color,
                'background_color': self.theme.background_color,
                'secondary_background_color': self.theme.secondary_background_color,
                'text_color': self.theme.text_color,
                'font': self.theme.font
            },
            'chat': {
                'max_message_length': self.chat.max_message_length,
                'show_timestamps': self.chat.show_timestamps,
                'enable_markdown': self.chat.enable_markdown,
                'avatar_user': self.chat.avatar_user,
                'avatar_assistant': self.chat.avatar_assistant
            },
            'visualization': {
                'default_height': self.visualization.default_height,
                'default_width': self.visualization.default_width,
                'template': self.visualization.template,
                'enable_animation': self.visualization.enable_animation,
                'color_scheme': self.visualization.color_scheme
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)