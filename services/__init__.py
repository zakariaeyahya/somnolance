"""
Package des services pour Guardian Eye
Système de détection de somnolence intelligent
"""

from .model_service import ModelService
from .detection_service import DetectionService
from .audio_service import AudioService
from .analytics_service import AnalyticsService
from .dashboard_service import DashboardService

__all__ = [
    'ModelService',
    'DetectionService', 
    'AudioService',
    'AnalyticsService',
    'DashboardService'
]

__version__ = "1.0.0"
__author__ = "Guardian Eye Team"