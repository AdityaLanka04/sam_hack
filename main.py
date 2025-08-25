# Complete Behavioral Fraud Detection Backend System
# Production-ready implementation for hackathon

import os
import json
import logging
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import uuid
from collections import defaultdict, deque
import pickle

# Core libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, g
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
import redis
from celery import Celery
import socketio
from threading import Lock

# ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import joblib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration Class
class Config:
    """Application configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'fraud-detection-hackathon-key')
    
    # Database configuration
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///fraud_detection.db')
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
    # Redis configuration
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # Celery configuration
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', REDIS_URL)
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', REDIS_URL)
    
    # ML Model parameters
    ANOMALY_THRESHOLD = 0.6
    MIN_TRAINING_SAMPLES = 50
    PROFILE_ESTABLISHMENT_THRESHOLD = 100
    
    # Real-time processing
    MAX_SESSION_DURATION = 3600  # 1 hour
    BATCH_PROCESSING_SIZE = 100
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = 'fraud_detection.log'

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Extensions initialization
db = SQLAlchemy()
migrate = Migrate()
sio = socketio.Server(cors_allowed_origins="*", async_mode='threading')

def init_extensions(app):
    """Initialize Flask extensions"""
    db.init_app(app)
    migrate.init_app(app, db)
    CORS(app, resources={
        r"/api/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

init_extensions(app)

# Redis client initialization
try:
    redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True)
    redis_client.ping()
except Exception as e:
    print(f"Redis connection failed: {e}. Using in-memory storage.")
    redis_client = None

# Celery setup
def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)
    
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    
    celery.Task = ContextTask
    return celery

celery = make_celery(app)

# Logging setup
def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    
    # Suppress noisy libraries
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger(__name__)

# Data Classes for Type Safety
@dataclass
class KeystrokeDynamics:
    """Keystroke timing and dynamics data"""
    user_id: str
    session_id: str
    key: str
    key_code: int
    dwell_time: float  # Time key is held down
    flight_time: float  # Time between key releases
    timestamp: float
    pressure: Optional[float] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class MouseBehavior:
    """Mouse movement and interaction data"""
    user_id: str
    session_id: str
    x: int
    y: int
    velocity: float
    acceleration: float
    jerk: float  # Rate of change of acceleration
    timestamp: float
    event_type: str  # 'move', 'click', 'scroll', 'drag'
    button: Optional[str] = None  # 'left', 'right', 'middle'
    
    def to_dict(self):
        return asdict(self)

@dataclass
class NavigationPattern:
    """User navigation behavior data"""
    user_id: str
    session_id: str
    page_url: str
    action_type: str  # 'page_load', 'form_fill', 'button_click'
    time_spent: float
    scroll_depth: float
    timestamp: float
    
    def to_dict(self):
        return asdict(self)

@dataclass
class BehavioralProfile:
    """Complete user behavioral profile"""
    user_id: str
    keystroke_stats: Dict[str, float]
    mouse_stats: Dict[str, float]
    navigation_stats: Dict[str, float]
    temporal_patterns: Dict[str, Any]
    baseline_established: bool
    confidence_level: float
    last_updated: datetime
    anomaly_threshold: float
    
    def to_dict(self):
        data = asdict(self)
        data['last_updated'] = self.last_updated.isoformat()
        return data

# Database Models
class User(db.Model):
    """User table"""
    __tablename__ = 'users'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    profile_established = db.Column(db.Boolean, default=False, nullable=False)
    risk_level = db.Column(db.String(20), default='low')  # low, medium, high
    
    # Relationships
    sessions = db.relationship('BehavioralSession', backref='user', lazy='dynamic')
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat() if self.last_active else None,
            'is_active': self.is_active,
            'profile_established': self.profile_established,
            'risk_level': self.risk_level
        }

class BehavioralSession(db.Model):
    """Behavioral session tracking"""
    __tablename__ = 'behavioral_sessions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False, index=True)
    session_start = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    session_end = db.Column(db.DateTime)
    ip_address = db.Column(db.String(45))  # IPv6 compatible
    user_agent = db.Column(db.Text)
    device_fingerprint = db.Column(db.Text)
    
    # Risk metrics
    risk_score = db.Column(db.Float, default=0.0)
    anomaly_count = db.Column(db.Integer, default=0)
    is_flagged = db.Column(db.Boolean, default=False)
    
    # Performance metrics
    total_keystrokes = db.Column(db.Integer, default=0)
    total_mouse_events = db.Column(db.Integer, default=0)
    
    # Relationships
    keystroke_events = db.relationship('KeystrokeEvent', backref='session', lazy='dynamic')
    mouse_events = db.relationship('MouseEvent', backref='session', lazy='dynamic')
    anomaly_alerts = db.relationship('AnomalyAlert', backref='session', lazy='dynamic')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'session_start': self.session_start.isoformat(),
            'session_end': self.session_end.isoformat() if self.session_end else None,
            'risk_score': self.risk_score,
            'anomaly_count': self.anomaly_count,
            'is_flagged': self.is_flagged,
            'total_keystrokes': self.total_keystrokes,
            'total_mouse_events': self.total_mouse_events
        }

class KeystrokeEvent(db.Model):
    """Individual keystroke events"""
    __tablename__ = 'keystroke_events'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = db.Column(db.String(36), db.ForeignKey('behavioral_sessions.id'), nullable=False, index=True)
    
    # Keystroke data
    key = db.Column(db.String(10))
    key_code = db.Column(db.Integer)
    dwell_time = db.Column(db.Float)  # milliseconds
    flight_time = db.Column(db.Float)  # milliseconds
    pressure = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Context
    input_field = db.Column(db.String(100))  # field being typed in
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'key': self.key,
            'key_code': self.key_code,
            'dwell_time': self.dwell_time,
            'flight_time': self.flight_time,
            'timestamp': self.timestamp.isoformat()
        }

class MouseEvent(db.Model):
    """Mouse movement and click events"""
    __tablename__ = 'mouse_events'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = db.Column(db.String(36), db.ForeignKey('behavioral_sessions.id'), nullable=False, index=True)
    
    # Mouse data
    x_position = db.Column(db.Integer)
    y_position = db.Column(db.Integer)
    velocity = db.Column(db.Float)
    acceleration = db.Column(db.Float)
    jerk = db.Column(db.Float)
    event_type = db.Column(db.String(20))  # move, click, scroll
    button = db.Column(db.String(10))  # left, right, middle
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'x_position': self.x_position,
            'y_position': self.y_position,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat()
        }

class AnomalyAlert(db.Model):
    """Anomaly detection alerts"""
    __tablename__ = 'anomaly_alerts'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = db.Column(db.String(36), db.ForeignKey('behavioral_sessions.id'), nullable=False, index=True)
    
    # Alert details
    anomaly_type = db.Column(db.String(50), nullable=False)  # keystroke, mouse, navigation
    confidence_score = db.Column(db.Float, nullable=False)
    severity = db.Column(db.String(20), default='medium')  # low, medium, high, critical
    details = db.Column(db.Text)  # JSON string with detailed info
    
    # Status
    resolved = db.Column(db.Boolean, default=False)
    false_positive = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    resolved_at = db.Column(db.DateTime)
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'anomaly_type': self.anomaly_type,
            'confidence_score': self.confidence_score,
            'severity': self.severity,
            'details': json.loads(self.details) if self.details else {},
            'resolved': self.resolved,
            'false_positive': self.false_positive,
            'created_at': self.created_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

# Feature Engineering Engine
class BehavioralFeatureExtractor:
    """Advanced feature extraction for behavioral analysis"""
    
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
    def extract_keystroke_features(self, keystroke_data: List[KeystrokeDynamics]) -> Dict[str, float]:
        """Extract comprehensive keystroke dynamics features"""
        if not keystroke_data or len(keystroke_data) < 2:
            return self._get_default_keystroke_features()
            
        dwell_times = [k.dwell_time for k in keystroke_data if k.dwell_time > 0]
        flight_times = [k.flight_time for k in keystroke_data if k.flight_time > 0]
        
        if not dwell_times:
            return self._get_default_keystroke_features()
        
        # Basic statistics
        features = {
            # Dwell time features
            'dwell_mean': np.mean(dwell_times),
            'dwell_std': np.std(dwell_times),
            'dwell_median': np.median(dwell_times),
            'dwell_q25': np.percentile(dwell_times, 25),
            'dwell_q75': np.percentile(dwell_times, 75),
            'dwell_iqr': np.percentile(dwell_times, 75) - np.percentile(dwell_times, 25),
            'dwell_skewness': stats.skew(dwell_times),
            'dwell_kurtosis': stats.kurtosis(dwell_times),
            
            # Flight time features
            'flight_mean': np.mean(flight_times) if flight_times else 0,
            'flight_std': np.std(flight_times) if flight_times else 0,
            'flight_median': np.median(flight_times) if flight_times else 0,
            'flight_skewness': stats.skew(flight_times) if len(flight_times) > 2 else 0,
            
            # Rhythm and timing
            'typing_rhythm_consistency': self._calculate_rhythm_consistency(flight_times),
            'typing_speed_wpm': self._calculate_typing_speed(keystroke_data),
            'pause_ratio': self._calculate_pause_ratio(flight_times),
            
            # Advanced features
            'keystroke_entropy': self._calculate_keystroke_entropy(keystroke_data),
            'bigram_consistency': self._calculate_bigram_consistency(keystroke_data),
            'pressure_variation': self._calculate_pressure_features(keystroke_data)
        }
        
        return features
    
    def extract_mouse_features(self, mouse_data: List[MouseBehavior]) -> Dict[str, float]:
        """Extract comprehensive mouse behavior features"""
        if not mouse_data or len(mouse_data) < 2:
            return self._get_default_mouse_features()
        
        velocities = [m.velocity for m in mouse_data if m.velocity >= 0]
        accelerations = [m.acceleration for m in mouse_data]
        x_positions = [m.x for m in mouse_data]
        y_positions = [m.y for m in mouse_data]
        
        if not velocities:
            return self._get_default_mouse_features()
        
        features = {
            # Velocity features
            'velocity_mean': np.mean(velocities),
            'velocity_std': np.std(velocities),
            'velocity_max': np.max(velocities),
            'velocity_95th': np.percentile(velocities, 95),
            'velocity_skewness': stats.skew(velocities),
            
            # Acceleration features
            'acceleration_mean': np.mean(accelerations),
            'acceleration_std': np.std(accelerations),
            'acceleration_rms': np.sqrt(np.mean(np.square(accelerations))),
            
            # Movement patterns
            'movement_efficiency': self._calculate_movement_efficiency(x_positions, y_positions),
            'movement_smoothness': self._calculate_movement_smoothness(velocities),
            'trajectory_complexity': self._calculate_trajectory_complexity(x_positions, y_positions),
            'direction_changes': self._count_direction_changes(x_positions, y_positions),
            
            # Click patterns
            'click_frequency': len([m for m in mouse_data if m.event_type == 'click']) / len(mouse_data),
            'double_click_ratio': self._calculate_double_click_ratio(mouse_data),
            'drag_ratio': len([m for m in mouse_data if m.event_type == 'drag']) / len(mouse_data),
            
            # Spatial features
            'movement_area': self._calculate_movement_area(x_positions, y_positions),
            'position_entropy': self._calculate_position_entropy(x_positions, y_positions)
        }
        
        return features
    
    def _get_default_keystroke_features(self) -> Dict[str, float]:
        """Default keystroke features when no data available"""
        return {
            'dwell_mean': 0.0, 'dwell_std': 0.0, 'dwell_median': 0.0,
            'dwell_q25': 0.0, 'dwell_q75': 0.0, 'dwell_iqr': 0.0,
            'dwell_skewness': 0.0, 'dwell_kurtosis': 0.0,
            'flight_mean': 0.0, 'flight_std': 0.0, 'flight_median': 0.0,
            'flight_skewness': 0.0, 'typing_rhythm_consistency': 0.0,
            'typing_speed_wpm': 0.0, 'pause_ratio': 0.0,
            'keystroke_entropy': 0.0, 'bigram_consistency': 0.0,
            'pressure_variation': 0.0
        }
    
    def _get_default_mouse_features(self) -> Dict[str, float]:
        """Default mouse features when no data available"""
        return {
            'velocity_mean': 0.0, 'velocity_std': 0.0, 'velocity_max': 0.0,
            'velocity_95th': 0.0, 'velocity_skewness': 0.0,
            'acceleration_mean': 0.0, 'acceleration_std': 0.0, 'acceleration_rms': 0.0,
            'movement_efficiency': 0.0, 'movement_smoothness': 0.0,
            'trajectory_complexity': 0.0, 'direction_changes': 0.0,
            'click_frequency': 0.0, 'double_click_ratio': 0.0, 'drag_ratio': 0.0,
            'movement_area': 0.0, 'position_entropy': 0.0
        }
    
    def _calculate_rhythm_consistency(self, flight_times: List[float]) -> float:
        """Calculate typing rhythm consistency"""
        if len(flight_times) < 3:
            return 0.0
        return 1.0 / (np.std(flight_times) + 1e-6)
    
    def _calculate_typing_speed(self, keystroke_data: List[KeystrokeDynamics]) -> float:
        """Calculate typing speed in WPM"""
        if len(keystroke_data) < 2:
            return 0.0
        
        duration_minutes = (keystroke_data[-1].timestamp - keystroke_data[0].timestamp) / 60.0
        if duration_minutes <= 0:
            return 0.0
            
        # Assume average word length of 5 characters
        words = len(keystroke_data) / 5.0
        return words / duration_minutes
    
    def _calculate_pause_ratio(self, flight_times: List[float]) -> float:
        """Calculate ratio of long pauses to total flight times"""
        if not flight_times:
            return 0.0
            
        threshold = np.mean(flight_times) + 2 * np.std(flight_times)
        long_pauses = sum(1 for ft in flight_times if ft > threshold)
        return long_pauses / len(flight_times)
    
    def _calculate_keystroke_entropy(self, keystroke_data: List[KeystrokeDynamics]) -> float:
        """Calculate entropy of keystroke patterns"""
        if not keystroke_data:
            return 0.0
            
        keys = [k.key for k in keystroke_data if k.key]
        if not keys:
            return 0.0
            
        # Calculate frequency distribution
        unique, counts = np.unique(keys, return_counts=True)
        probabilities = counts / len(keys)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _calculate_bigram_consistency(self, keystroke_data: List[KeystrokeDynamics]) -> float:
        """Calculate consistency of bigram (two-key) timing patterns"""
        if len(keystroke_data) < 3:
            return 0.0
            
        bigrams = {}
        for i in range(len(keystroke_data) - 1):
            if keystroke_data[i].key and keystroke_data[i + 1].key:
                bigram = keystroke_data[i].key + keystroke_data[i + 1].key
                flight_time = keystroke_data[i + 1].flight_time
                
                if bigram not in bigrams:
                    bigrams[bigram] = []
                bigrams[bigram].append(flight_time)
        
        if not bigrams:
            return 0.0
            
        # Calculate average consistency across all bigrams
        consistencies = []
        for times in bigrams.values():
            if len(times) > 1:
                consistency = 1.0 / (np.std(times) + 1e-6)
                consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 0.0
    
    def _calculate_pressure_features(self, keystroke_data: List[KeystrokeDynamics]) -> float:
        """Calculate pressure variation features"""
        pressures = [k.pressure for k in keystroke_data if k.pressure is not None]
        if len(pressures) < 2:
            return 0.0
        return np.std(pressures)
    
    def _calculate_movement_efficiency(self, x_pos: List[int], y_pos: List[int]) -> float:
        """Calculate mouse movement efficiency"""
        if len(x_pos) < 2:
            return 1.0
            
        direct_distance = np.sqrt((x_pos[-1] - x_pos[0])**2 + (y_pos[-1] - y_pos[0])**2)
        if direct_distance == 0:
            return 1.0
            
        actual_distance = sum(
            np.sqrt((x_pos[i] - x_pos[i-1])**2 + (y_pos[i] - y_pos[i-1])**2)
            for i in range(1, len(x_pos))
        )
        
        return direct_distance / (actual_distance + 1e-6) if actual_distance > 0 else 1.0
    
    def _calculate_movement_smoothness(self, velocities: List[float]) -> float:
        """Calculate smoothness of mouse movements"""
        if len(velocities) < 2:
            return 1.0
            
        velocity_changes = [abs(velocities[i] - velocities[i-1]) for i in range(1, len(velocities))]
        return 1.0 / (np.mean(velocity_changes) + 1e-6)
    
    def _calculate_trajectory_complexity(self, x_pos: List[int], y_pos: List[int]) -> float:
        """Calculate complexity of mouse trajectory"""
        if len(x_pos) < 3:
            return 0.0
            
        # Calculate curvature at each point
        curvatures = []
        for i in range(1, len(x_pos) - 1):
            # Vector from previous to current point
            v1 = np.array([x_pos[i] - x_pos[i-1], y_pos[i] - y_pos[i-1]])
            # Vector from current to next point
            v2 = np.array([x_pos[i+1] - x_pos[i], y_pos[i+1] - y_pos[i]])
            
            # Calculate angle between vectors
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)  # Ensure valid range
                angle = np.arccos(cos_angle)
                curvatures.append(angle)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    def _count_direction_changes(self, x_pos: List[int], y_pos: List[int]) -> float:
        """Count significant direction changes in mouse movement"""
        if len(x_pos) < 3:
            return 0.0
            
        direction_changes = 0
        threshold = np.pi / 4  # 45 degrees
        
        for i in range(1, len(x_pos) - 1):
            v1 = np.array([x_pos[i] - x_pos[i-1], y_pos[i] - y_pos[i-1]])
            v2 = np.array([x_pos[i+1] - x_pos[i], y_pos[i+1] - y_pos[i]])
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                if angle > threshold:
                    direction_changes += 1
        
        return direction_changes / (len(x_pos) - 2) if len(x_pos) > 2 else 0.0
    
    def _calculate_double_click_ratio(self, mouse_data: List[MouseBehavior]) -> float:
        """Calculate ratio of double clicks to total clicks"""
        clicks = [m for m in mouse_data if m.event_type == 'click']
        if len(clicks) < 2:
            return 0.0
            
        double_clicks = 0
        double_click_threshold = 500  # 500ms
        
        for i in range(1, len(clicks)):
            time_diff = (clicks[i].timestamp - clicks[i-1].timestamp) * 1000  # Convert to ms
            if time_diff <= double_click_threshold:
                double_clicks += 1
        
        return double_clicks / len(clicks) if clicks else 0.0
    
    def _calculate_movement_area(self, x_pos: List[int], y_pos: List[int]) -> float:
        """Calculate area covered by mouse movements"""
        if len(x_pos) < 2:
            return 0.0
            
        min_x, max_x = min(x_pos), max(x_pos)
        min_y, max_y = min(y_pos), max(y_pos)
        
        return (max_x - min_x) * (max_y - min_y)
    
    def _calculate_position_entropy(self, x_pos: List[int], y_pos: List[int]) -> float:
        """Calculate entropy of mouse position distribution"""
        if len(x_pos) < 2:
            return 0.0
            
        # Create 2D histogram
        try:
            hist, _, _ = np.histogram2d(x_pos, y_pos, bins=10)
            hist = hist.flatten()
            hist = hist[hist > 0]  # Remove empty bins
            
            if len(hist) == 0:
                return 0.0
                
            probabilities = hist / np.sum(hist)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return entropy
        except:
            return 0.0

# Advanced Anomaly Detection Engine
class AnomalyDetectionEngine:
    """Multi-model anomaly detection system"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        self.one_class_svm = None
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        
        self.scaler = RobustScaler()
        self.feature_extractor = BehavioralFeatureExtractor()
        self.models_trained = False
        self.feature_names = []
        
        # Real-time anomaly detection
        self.anomaly_history = defaultdict(deque)  # Per-user anomaly scores
        self.baseline_profiles = {}  # Per-user baseline profiles
        
        # Thread safety
        self._lock = Lock()
    
    def train_models(self, training_data: Dict[str, Dict]) -> bool:
        """Train all anomaly detection models"""
        logger.info("Training anomaly detection models...")
        
        with self._lock:
            try:
                # Prepare features from training data
                feature_matrix, user_ids = self._prepare_training_features(training_data)
                
                if len(feature_matrix) < Config.MIN_TRAINING_SAMPLES:
                    logger.warning(f"Insufficient training data: {len(feature_matrix)} samples")
                    return False
                
                # Scale features
                scaled_features = self.scaler.fit_transform(feature_matrix)
                
                # Train Isolation Forest
                self.isolation_forest.fit(scaled_features)
                
                # Train One-Class SVM for comparison
                from sklearn.svm import OneClassSVM
                self.one_class_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
                self.one_class_svm.fit(scaled_features)
                
                # Build baseline profiles per user
                self._build_baseline_profiles(training_data)
                
                self.models_trained = True
                logger.info(f"Models trained successfully with {len(feature_matrix)} samples")
                return True
                
            except Exception as e:
                logger.error(f"Model training failed: {str(e)}")
                return False
    
    def _prepare_training_features(self, training_data: Dict) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix from training data"""
        features = []
        user_ids = []
        
        for user_id, sessions in training_data.items():
            for session_id, session_data in sessions.items():
                # Extract features for this session
                keystroke_features = self.feature_extractor.extract_keystroke_features(
                    session_data.get('keystrokes', [])
                )
                mouse_features = self.feature_extractor.extract_mouse_features(
                    session_data.get('mouse', [])
                )
                
                # Combine all features
                combined_features = {**keystroke_features, **mouse_features}
                
                if combined_features and any(v != 0 for v in combined_features.values()):
                    features.append(list(combined_features.values()))
                    user_ids.append(user_id)
                    
                    # Store feature names for later reference
                    if not self.feature_names:
                        self.feature_names = list(combined_features.keys())
        
        return np.array(features) if features else np.array([]), user_ids
    
    def _build_baseline_profiles(self, training_data: Dict):
        """Build baseline behavioral profiles for each user"""
        for user_id, sessions in training_data.items():
            user_features = []
            
            for session_data in sessions.values():
                keystroke_features = self.feature_extractor.extract_keystroke_features(
                    session_data.get('keystrokes', [])
                )
                mouse_features = self.feature_extractor.extract_mouse_features(
                    session_data.get('mouse', [])
                )
                
                combined_features = {**keystroke_features, **mouse_features}
                if combined_features:
                    user_features.append(list(combined_features.values()))
            
            if user_features:
                feature_matrix = np.array(user_features)
                baseline_profile = {
                    'mean': np.mean(feature_matrix, axis=0),
                    'std': np.std(feature_matrix, axis=0),
                    'median': np.median(feature_matrix, axis=0),
                    'q25': np.percentile(feature_matrix, 25, axis=0),
                    'q75': np.percentile(feature_matrix, 75, axis=0)
                }
                self.baseline_profiles[user_id] = baseline_profile
    
    def detect_anomaly(self, user_id: str, current_data: Dict) -> Dict[str, Any]:
        """Detect anomalies in current behavioral data"""
        if not self.models_trained:
            return {
                'anomaly_score': 0.0,
                'is_anomaly': False,
                'confidence': 0.0,
                'risk_level': 'low',
                'details': {'error': 'Models not trained'}
            }
        
        try:
            # Extract features from current data
            keystroke_features = self.feature_extractor.extract_keystroke_features(
                current_data.get('keystrokes', [])
            )
            mouse_features = self.feature_extractor.extract_mouse_features(
                current_data.get('mouse', [])
            )
            
            combined_features = {**keystroke_features, **mouse_features}
            
            if not combined_features or all(v == 0 for v in combined_features.values()):
                return {
                    'anomaly_score': 0.0,
                    'is_anomaly': False,
                    'confidence': 0.0,
                    'risk_level': 'low',
                    'details': {'error': 'No valid features extracted'}
                }
            
            # Prepare feature vector
            feature_vector = np.array(list(combined_features.values())).reshape(1, -1)
            
            with self._lock:
                scaled_features = self.scaler.transform(feature_vector)
            
            # Get predictions from multiple models
            isolation_score = self.isolation_forest.decision_function(scaled_features)[0]
            isolation_anomaly = self.isolation_forest.predict(scaled_features)[0] == -1
            
            svm_score = 0.0
            svm_anomaly = False
            if self.one_class_svm:
                svm_score = self.one_class_svm.decision_function(scaled_features)[0]
                svm_anomaly = self.one_class_svm.predict(scaled_features)[0] == -1
            
            # User-specific baseline comparison
            baseline_score = self._compare_with_baseline(user_id, feature_vector[0])
            
            # Temporal anomaly detection
            temporal_score = self._detect_temporal_anomaly(user_id, combined_features)
            
            # Combine all scores
            final_score = self._combine_anomaly_scores(
                isolation_score, svm_score, baseline_score, temporal_score
            )
            
            # Determine if anomaly based on multiple factors
            is_anomaly = (isolation_anomaly or svm_anomaly or 
                         baseline_score > 0.7 or temporal_score > 0.8)
            
            # Calculate confidence and risk level
            confidence = min(abs(final_score), 1.0)
            risk_level = self._calculate_risk_level(final_score, is_anomaly)
            
            # Update anomaly history for temporal analysis
            self.anomaly_history[user_id].append(final_score)
            if len(self.anomaly_history[user_id]) > 100:  # Keep last 100 scores
                self.anomaly_history[user_id].popleft()
            
            return {
                'anomaly_score': float(final_score),
                'is_anomaly': bool(is_anomaly),
                'confidence': float(confidence),
                'risk_level': risk_level,
                'details': {
                    'isolation_score': float(isolation_score),
                    'svm_score': float(svm_score),
                    'baseline_score': float(baseline_score),
                    'temporal_score': float(temporal_score),
                    'feature_count': len(combined_features),
                    'anomalous_features': self._identify_anomalous_features(
                        user_id, combined_features
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return {
                'anomaly_score': 0.0,
                'is_anomaly': False,
                'confidence': 0.0,
                'risk_level': 'low',
                'details': {'error': str(e)}
            }
    
    def _compare_with_baseline(self, user_id: str, features: np.ndarray) -> float:
        """Compare current features with user's baseline profile"""
        if user_id not in self.baseline_profiles:
            return 0.0
        
        baseline = self.baseline_profiles[user_id]
        
        # Calculate z-scores for each feature
        z_scores = []
        for i, value in enumerate(features):
            if i < len(baseline['std']) and baseline['std'][i] > 0:
                z_score = abs(value - baseline['mean'][i]) / baseline['std'][i]
                z_scores.append(z_score)
        
        if not z_scores:
            return 0.0
        
        # Anomaly score based on how many features are beyond 2 standard deviations
        anomalous_features = sum(1 for z in z_scores if z > 2.0)
        return anomalous_features / len(z_scores)
    
    def _detect_temporal_anomaly(self, user_id: str, current_features: Dict) -> float:
        """Detect temporal anomalies based on recent behavior patterns"""
        if user_id not in self.anomaly_history or len(self.anomaly_history[user_id]) < 5:
            return 0.0
        
        recent_scores = list(self.anomaly_history[user_id])[-10:]  # Last 10 scores
        mean_recent = np.mean(recent_scores)
        std_recent = np.std(recent_scores)
        
        # Current score based on feature similarity to recent behavior
        # This is a simplified version - in production you'd want more sophisticated temporal analysis
        current_score = sum(current_features.values()) / len(current_features)
        
        if std_recent > 0:
            z_score = abs(current_score - mean_recent) / std_recent
            return min(z_score / 3.0, 1.0)  # Normalize to 0-1 range
        
        return 0.0
    
    def _combine_anomaly_scores(self, isolation: float, svm: float, baseline: float, temporal: float) -> float:
        """Combine multiple anomaly scores into final score"""
        # Weighted combination of different scoring methods
        weights = {
            'isolation': 0.3,
            'svm': 0.2,
            'baseline': 0.3,
            'temporal': 0.2
        }
        
        # Normalize scores to 0-1 range
        isolation_norm = max(0, min(1, abs(isolation)))
        svm_norm = max(0, min(1, abs(svm)))
        baseline_norm = max(0, min(1, baseline))
        temporal_norm = max(0, min(1, temporal))
        
        final_score = (
            weights['isolation'] * isolation_norm +
            weights['svm'] * svm_norm +
            weights['baseline'] * baseline_norm +
            weights['temporal'] * temporal_norm
        )
        
        return final_score
    
    def _calculate_risk_level(self, score: float, is_anomaly: bool) -> str:
        """Calculate risk level based on anomaly score"""
        if not is_anomaly and score < 0.3:
            return 'low'
        elif score < 0.5:
            return 'medium'
        elif score < 0.8:
            return 'high'
        else:
            return 'critical'
    
    def _identify_anomalous_features(self, user_id: str, features: Dict) -> List[str]:
        """Identify which specific features are anomalous"""
        if user_id not in self.baseline_profiles:
            return []
        
        baseline = self.baseline_profiles[user_id]
        anomalous = []
        
        for feature_name, value in features.items():
            if feature_name in self.feature_names:
                idx = self.feature_names.index(feature_name)
                if idx < len(baseline['std']) and baseline['std'][idx] > 0:
                    z_score = abs(value - baseline['mean'][idx]) / baseline['std'][idx]
                    if z_score > 2.5:  # More than 2.5 standard deviations
                        anomalous.append(feature_name)
        
        return anomalous

# Profile Management System
class ProfileManager:
    """Manages user behavioral profiles"""
    
    def __init__(self):
        self.feature_extractor = BehavioralFeatureExtractor()
        self._cache = {}  # In-memory cache if Redis unavailable
        self._lock = Lock()
    
    def build_user_profile(self, user_id: str, historical_data: Dict) -> BehavioralProfile:
        """Build comprehensive behavioral profile for user"""
        logger.info(f"Building behavioral profile for user {user_id}")
        
        try:
            # Aggregate all historical data
            all_keystrokes = []
            all_mouse = []
            session_durations = []
            
            for session_data in historical_data.values():
                keystrokes = session_data.get('keystrokes', [])
                mouse_events = session_data.get('mouse', [])
                
                all_keystrokes.extend(keystrokes)
                all_mouse.extend(mouse_events)
                
                # Calculate session duration
                if keystrokes and len(keystrokes) > 1:
                    duration = keystrokes[-1].timestamp - keystrokes[0].timestamp
                    session_durations.append(duration)
            
            # Extract comprehensive features
            keystroke_stats = self.feature_extractor.extract_keystroke_features(all_keystrokes)
            mouse_stats = self.feature_extractor.extract_mouse_features(all_mouse)
            
            # Calculate temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(historical_data)
            
            # Navigation patterns
            navigation_stats = self._calculate_navigation_stats(session_durations)
            
            # Determine baseline establishment and confidence
            total_events = len(all_keystrokes) + len(all_mouse)
            baseline_established = total_events >= Config.PROFILE_ESTABLISHMENT_THRESHOLD
            confidence_level = min(total_events / Config.PROFILE_ESTABLISHMENT_THRESHOLD, 1.0)
            
            # Calculate personalized anomaly threshold
            anomaly_threshold = self._calculate_anomaly_threshold(keystroke_stats, mouse_stats)
            
            profile = BehavioralProfile(
                user_id=user_id,
                keystroke_stats=keystroke_stats,
                mouse_stats=mouse_stats,
                navigation_stats=navigation_stats,
                temporal_patterns=temporal_patterns,
                baseline_established=baseline_established,
                confidence_level=confidence_level,
                last_updated=datetime.utcnow(),
                anomaly_threshold=anomaly_threshold
            )
            
            # Cache the profile
            self._cache_profile(user_id, profile)
            
            logger.info(f"Profile built for user {user_id}: {total_events} events, "
                       f"confidence: {confidence_level:.2f}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Profile building failed for user {user_id}: {str(e)}")
            raise
    
    def _analyze_temporal_patterns(self, historical_data: Dict) -> Dict[str, Any]:
        """Analyze temporal behavior patterns"""
        patterns = {
            'session_count': len(historical_data),
            'avg_session_duration': 0.0,
            'preferred_hours': [],
            'activity_consistency': 0.0,
            'break_patterns': []
        }
        
        session_durations = []
        session_hours = []
        
        for session_data in historical_data.values():
            keystrokes = session_data.get('keystrokes', [])
            if len(keystrokes) >= 2:
                duration = keystrokes[-1].timestamp - keystrokes[0].timestamp
                session_durations.append(duration)
                
                # Extract hour of day
                session_start = datetime.fromtimestamp(keystrokes[0].timestamp)
                session_hours.append(session_start.hour)
        
        if session_durations:
            patterns['avg_session_duration'] = np.mean(session_durations)
            patterns['activity_consistency'] = 1.0 / (np.std(session_durations) + 1e-6)
        
        if session_hours:
            # Find most common hours
            hour_counts = np.bincount(session_hours, minlength=24)
            top_hours = np.argsort(hour_counts)[-3:]  # Top 3 hours
            patterns['preferred_hours'] = top_hours.tolist()
        
        return patterns
    
    def _calculate_navigation_stats(self, session_durations: List[float]) -> Dict[str, float]:
        """Calculate navigation and session statistics"""
        if not session_durations:
            return {
                'avg_session_duration': 0.0,
                'session_consistency': 0.0,
                'total_sessions': 0
            }
        
        return {
            'avg_session_duration': np.mean(session_durations),
            'session_consistency': 1.0 / (np.std(session_durations) + 1e-6),
            'total_sessions': len(session_durations)
        }
    
    def _calculate_anomaly_threshold(self, keystroke_stats: Dict, mouse_stats: Dict) -> float:
        """Calculate personalized anomaly threshold based on behavioral consistency"""
        # Users with more consistent behavior get lower thresholds (more sensitive)
        keystroke_consistency = keystroke_stats.get('typing_rhythm_consistency', 0)
        mouse_consistency = mouse_stats.get('movement_smoothness', 0)
        
        avg_consistency = np.mean([keystroke_consistency, mouse_consistency])
        
        # Threshold inversely related to consistency
        # More consistent users = lower threshold = more sensitive detection
        base_threshold = 0.6
        adjustment = 0.3 * (1.0 / (avg_consistency + 1e-6))
        threshold = base_threshold - min(adjustment, 0.4)
        
        return max(0.2, min(threshold, 0.9))  # Clamp between 0.2 and 0.9
    
    def _cache_profile(self, user_id: str, profile: BehavioralProfile):
        """Cache user profile"""
        with self._lock:
            try:
                profile_data = profile.to_dict()
                
                if redis_client:
                    # Cache in Redis with 24-hour expiry
                    redis_client.setex(
                        f"profile:{user_id}",
                        timedelta(hours=24),
                        json.dumps(profile_data)
                    )
                else:
                    # Use in-memory cache as fallback
                    self._cache[user_id] = profile_data
                    
            except Exception as e:
                logger.warning(f"Profile caching failed: {str(e)}")
                # Fallback to in-memory cache
                self._cache[user_id] = profile.to_dict()
    
    def get_cached_profile(self, user_id: str) -> Optional[BehavioralProfile]:
        """Retrieve cached user profile"""
        try:
            profile_data = None
            
            if redis_client:
                cached_data = redis_client.get(f"profile:{user_id}")
                if cached_data:
                    profile_data = json.loads(cached_data)
            
            # Fallback to in-memory cache
            if not profile_data and user_id in self._cache:
                profile_data = self._cache[user_id]
            
            if profile_data:
                # Convert ISO string back to datetime
                profile_data['last_updated'] = datetime.fromisoformat(
                    profile_data['last_updated']
                )
                return BehavioralProfile(**profile_data)
                
            return None
            
        except Exception as e:
            logger.warning(f"Profile retrieval failed: {str(e)}")
            return None
    
    def update_profile(self, user_id: str, new_data: Dict):
        """Update existing profile with new behavioral data"""
        try:
            existing_profile = self.get_cached_profile(user_id)
            if not existing_profile:
                logger.warning(f"No existing profile found for user {user_id}")
                return None
            
            # Merge new data with existing profile
            # This is a simplified update - in production you'd want incremental learning
            updated_profile = self.build_user_profile(user_id, new_data)
            return updated_profile
            
        except Exception as e:
            logger.error(f"Profile update failed for user {user_id}: {str(e)}")
            return None

# Real-time Event Processor
class RealTimeProcessor:
    """Handles real-time processing of behavioral events"""
    
    def __init__(self):
        self.anomaly_engine = AnomalyDetectionEngine()
        self.profile_manager = ProfileManager()
        self.active_sessions = {}
        self.session_locks = defaultdict(Lock)  # Per-session locks
        
        # Event buffers for batch processing
        self.event_buffers = defaultdict(list)
        self.buffer_locks = defaultdict(Lock)
        
        # Statistics
        self.stats = {
            'events_processed': 0,
            'anomalies_detected': 0,
            'sessions_active': 0,
            'last_reset': datetime.utcnow()
        }
    
    async def process_behavioral_event(self, event_data: Dict) -> Dict[str, Any]:
        """Process incoming behavioral events in real-time"""
        try:
            user_id = event_data.get('user_id')
            session_id = event_data.get('session_id')
            event_type = event_data.get('type')
            
            if not all([user_id, session_id, event_type]):
                return self._error_response('Missing required fields: user_id, session_id, type')
            
            # Initialize or get session
            session = self._get_or_create_session(user_id, session_id, event_data)
            
            # Process event based on type
            processed = await self._process_event_by_type(session, event_data)
            if not processed:
                return self._error_response('Failed to process event')
            
            # Update statistics
            self.stats['events_processed'] += 1
            
            # Check if we should perform anomaly detection
            detection_result = await self._check_anomaly_detection(session_id, session)
            
            return {
                'status': 'success',
                'session_id': session_id,
                'events_in_session': len(session['keystrokes']) + len(session['mouse']),
                **detection_result
            }
            
        except Exception as e:
            logger.error(f"Event processing failed: {str(e)}")
            return self._error_response(f'Processing failed: {str(e)}')
    
    def _get_or_create_session(self, user_id: str, session_id: str, event_data: Dict) -> Dict:
        """Get existing session or create new one"""
        if session_id not in self.active_sessions:
            with self.session_locks[session_id]:
                if session_id not in self.active_sessions:  # Double-check pattern
                    self.active_sessions[session_id] = {
                        'user_id': user_id,
                        'keystrokes': [],
                        'mouse': [],
                        'navigation': [],
                        'start_time': time.time(),
                        'last_activity': time.time(),
                        'anomaly_count': 0,
                        'risk_score': 0.0,
                        'metadata': {
                            'ip_address': event_data.get('ip_address'),
                            'user_agent': event_data.get('user_agent'),
                            'device_fingerprint': event_data.get('device_fingerprint')
                        }
                    }
                    self.stats['sessions_active'] += 1
        
        # Update last activity
        self.active_sessions[session_id]['last_activity'] = time.time()
        return self.active_sessions[session_id]
    
    async def _process_event_by_type(self, session: Dict, event_data: Dict) -> bool:
        """Process event based on its type"""
        try:
            event_type = event_data['type']
            user_id = session['user_id']
            session_id = event_data['session_id']
            
            if event_type == 'keystroke':
                keystroke = KeystrokeDynamics(
                    user_id=user_id,
                    session_id=session_id,
                    key=event_data.get('key', ''),
                    key_code=event_data.get('key_code', 0),
                    dwell_time=event_data.get('dwell_time', 0.0),
                    flight_time=event_data.get('flight_time', 0.0),
                    timestamp=event_data.get('timestamp', time.time()),
                    pressure=event_data.get('pressure')
                )
                session['keystrokes'].append(keystroke)
                
                # Store in database asynchronously
                await self._store_keystroke_event(keystroke)
                
            elif event_type == 'mouse':
                # Calculate derived metrics
                velocity = event_data.get('velocity', 0.0)
                acceleration = event_data.get('acceleration', 0.0)
                jerk = event_data.get('jerk', 0.0)
                
                # If not provided, calculate from previous mouse events
                if not velocity and len(session['mouse']) > 0:
                    last_mouse = session['mouse'][-1]
                    dx = event_data.get('x', 0) - last_mouse.x
                    dy = event_data.get('y', 0) - last_mouse.y
                    dt = event_data.get('timestamp', time.time()) - last_mouse.timestamp
                    
                    if dt > 0:
                        distance = np.sqrt(dx*dx + dy*dy)
                        velocity = distance / dt
                
                mouse_event = MouseBehavior(
                    user_id=user_id,
                    session_id=session_id,
                    x=event_data.get('x', 0),
                    y=event_data.get('y', 0),
                    velocity=velocity,
                    acceleration=acceleration,
                    jerk=jerk,
                    timestamp=event_data.get('timestamp', time.time()),
                    event_type=event_data.get('mouse_event_type', 'move'),
                    button=event_data.get('button')
                )
                session['mouse'].append(mouse_event)
                
                # Store in database asynchronously
                await self._store_mouse_event(mouse_event)
                
            elif event_type == 'navigation':
                nav_event = NavigationPattern(
                    user_id=user_id,
                    session_id=session_id,
                    page_url=event_data.get('page_url', ''),
                    action_type=event_data.get('action_type', 'page_load'),
                    time_spent=event_data.get('time_spent', 0.0),
                    scroll_depth=event_data.get('scroll_depth', 0.0),
                    timestamp=event_data.get('timestamp', time.time())
                )
                session['navigation'].append(nav_event)
            
            return True
            
        except Exception as e:
            logger.error(f"Event type processing failed: {str(e)}")
            return False
    
    async def _check_anomaly_detection(self, session_id: str, session: Dict) -> Dict[str, Any]:
        """Check if anomaly detection should be performed"""
        keystroke_count = len(session['keystrokes'])
        mouse_count = len(session['mouse'])
        total_events = keystroke_count + mouse_count
        
        # Perform detection if we have enough events
        if total_events > 10 and (total_events % 20 == 0 or total_events > 100):
            try:
                detection_result = self.anomaly_engine.detect_anomaly(
                    session['user_id'], session
                )
                
                # Update session risk metrics
                session['risk_score'] = detection_result['anomaly_score']
                
                if detection_result['is_anomaly']:
                    session['anomaly_count'] += 1
                    self.stats['anomalies_detected'] += 1
                    
                    # Store anomaly alert
                    await self._store_anomaly_alert(session_id, detection_result)
                    
                    # Emit real-time alert
                    await self._emit_realtime_alert(session_id, detection_result)
                
                return {
                    'anomaly_detected': detection_result['is_anomaly'],
                    'risk_score': detection_result['anomaly_score'],
                    'confidence': detection_result['confidence'],
                    'risk_level': detection_result['risk_level']
                }
                
            except Exception as e:
                logger.error(f"Anomaly detection failed: {str(e)}")
                return {'anomaly_detected': False, 'error': str(e)}
        
        return {'anomaly_detected': False, 'reason': 'insufficient_data'}
    
    async def _store_keystroke_event(self, keystroke: KeystrokeDynamics):
        """Store keystroke event in database (async)"""
        try:
            # In a real implementation, you'd use async database operations
            # For now, we'll add to a queue for batch processing
            with self.buffer_locks['keystrokes']:
                self.event_buffers['keystrokes'].append(keystroke)
                
                # Process buffer if it's full
                if len(self.event_buffers['keystrokes']) >= Config.BATCH_PROCESSING_SIZE:
                    await self._flush_keystroke_buffer()
                    
        except Exception as e:
            logger.error(f"Keystroke storage failed: {str(e)}")
    
    async def _store_mouse_event(self, mouse_event: MouseBehavior):
        """Store mouse event in database (async)"""
        try:
            with self.buffer_locks['mouse']:
                self.event_buffers['mouse'].append(mouse_event)
                
                if len(self.event_buffers['mouse']) >= Config.BATCH_PROCESSING_SIZE:
                    await self._flush_mouse_buffer()
                    
        except Exception as e:
            logger.error(f"Mouse event storage failed: {str(e)}")
    
    async def _flush_keystroke_buffer(self):
        """Flush keystroke events to database"""
        try:
            events = self.event_buffers['keystrokes'][:]
            self.event_buffers['keystrokes'].clear()
            
            # Store in database (would be async in production)
            for event in events:
                db_event = KeystrokeEvent(
                    session_id=event.session_id,
                    key=event.key,
                    key_code=event.key_code,
                    dwell_time=event.dwell_time,
                    flight_time=event.flight_time,
                    pressure=event.pressure,
                    timestamp=datetime.fromtimestamp(event.timestamp)
                )
                db.session.add(db_event)
            
            db.session.commit()
            logger.debug(f"Flushed {len(events)} keystroke events to database")
            
        except Exception as e:
            logger.error(f"Keystroke buffer flush failed: {str(e)}")
    
    async def _flush_mouse_buffer(self):
        """Flush mouse events to database"""
        try:
            events = self.event_buffers['mouse'][:]
            self.event_buffers['mouse'].clear()
            
            for event in events:
                db_event = MouseEvent(
                    session_id=event.session_id,
                    x_position=event.x,
                    y_position=event.y,
                    velocity=event.velocity,
                    acceleration=event.acceleration,
                    jerk=event.jerk,
                    event_type=event.event_type,
                    button=event.button,
                    timestamp=datetime.fromtimestamp(event.timestamp)
                )
                db.session.add(db_event)
            
            db.session.commit()
            logger.debug(f"Flushed {len(events)} mouse events to database")
            
        except Exception as e:
            logger.error(f"Mouse buffer flush failed: {str(e)}")
    
    async def _store_anomaly_alert(self, session_id: str, detection_result: Dict):
        """Store anomaly alert in database"""
        try:
            alert = AnomalyAlert(
                session_id=session_id,
                anomaly_type='behavioral',
                confidence_score=detection_result['confidence'],
                severity=detection_result['risk_level'],
                details=json.dumps(detection_result['details'])
            )
            
            db.session.add(alert)
            db.session.commit()
            
            logger.info(f"Stored anomaly alert for session {session_id}")
            
        except Exception as e:
            logger.error(f"Anomaly alert storage failed: {str(e)}")
    
    async def _emit_realtime_alert(self, session_id: str, detection_result: Dict):
        """Emit real-time alert via WebSocket"""
        try:
            alert_data = {
                'session_id': session_id,
                'anomaly_type': 'behavioral',
                'risk_level': detection_result['risk_level'],
                'confidence': detection_result['confidence'],
                'timestamp': datetime.utcnow().isoformat(),
                'details': detection_result.get('details', {})
            }
            
            # Emit to all connected clients (in production, you'd filter by user/admin)
            await sio.emit('anomaly_alert', alert_data)
            
        except Exception as e:
            logger.error(f"Real-time alert emission failed: {str(e)}")
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Generate standardized error response"""
        return {
            'status': 'error',
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def cleanup_inactive_sessions(self):
        """Clean up inactive sessions"""
        try:
            current_time = time.time()
            inactive_sessions = []
            
            for session_id, session in self.active_sessions.items():
                if current_time - session['last_activity'] > Config.MAX_SESSION_DURATION:
                    inactive_sessions.append(session_id)
            
            for session_id in inactive_sessions:
                # Finalize session in database
                session = BehavioralSession.query.get(session_id)
                if session:
                    session.session_end = datetime.utcnow()
                    session.total_keystrokes = len(self.active_sessions[session_id]['keystrokes'])
                    session.total_mouse_events = len(self.active_sessions[session_id]['mouse'])
                    session.risk_score = self.active_sessions[session_id]['risk_score']
                    session.anomaly_count = self.active_sessions[session_id]['anomaly_count']
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                self.stats['sessions_active'] -= 1
            
            db.session.commit()
            
            if inactive_sessions:
                logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
                
        except Exception as e:
            logger.error(f"Session cleanup failed: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        uptime = datetime.utcnow() - self.stats['last_reset']
        
        return {
            **self.stats,
            'uptime_seconds': uptime.total_seconds(),
            'events_per_second': self.stats['events_processed'] / max(uptime.total_seconds(), 1),
            'anomaly_rate': self.stats['anomalies_detected'] / max(self.stats['events_processed'], 1)
        }

# Initialize global instances
real_time_processor = RealTimeProcessor()

# Background tasks using Celery
@celery.task
def train_user_models(user_id: str):
    """Background task to train ML models for a specific user"""
    logger.info(f"Training models for user {user_id}")
    
    try:
        # Fetch user's historical data
        user = User.query.get(user_id)
        if not user:
            logger.error(f"User {user_id} not found")
            return False
        
        sessions = BehavioralSession.query.filter_by(user_id=user_id).all()
        if not sessions:
            logger.warning(f"No sessions found for user {user_id}")
            return False
        
        # Prepare training data
        training_data = {user_id: {}}
        
        for session in sessions:
            # Get keystrokes
            keystrokes = KeystrokeEvent.query.filter_by(session_id=session.id).all()
            keystroke_objects = []
            for k in keystrokes:
                keystroke_objects.append(KeystrokeDynamics(
                    user_id=user_id,
                    session_id=session.id,
                    key=k.key,
                    key_code=k.key_code,
                    dwell_time=k.dwell_time,
                    flight_time=k.flight_time,
                    timestamp=k.timestamp.timestamp(),
                    pressure=k.pressure
                ))
            
            # Get mouse events
            mouse_events = MouseEvent.query.filter_by(session_id=session.id).all()
            mouse_objects = []
            for m in mouse_events:
                mouse_objects.append(MouseBehavior(
                    user_id=user_id,
                    session_id=session.id,
                    x=m.x_position,
                    y=m.y_position,
                    velocity=m.velocity,
                    acceleration=m.acceleration,
                    jerk=m.jerk,
                    timestamp=m.timestamp.timestamp(),
                    event_type=m.event_type,
                    button=m.button
                ))
            
            training_data[user_id][session.id] = {
                'keystrokes': keystroke_objects,
                'mouse': mouse_objects
            }
        
        # Train anomaly detection models
        anomaly_engine = AnomalyDetectionEngine()
        success = anomaly_engine.train_models(training_data)
        
        if success:
            # Save trained models
            model_path = f'models/user_{user_id}_anomaly_model.pkl'
            os.makedirs('models', exist_ok=True)
            joblib.dump(anomaly_engine, model_path)
            
            # Build and cache user profile
            profile_manager = ProfileManager()
            profile = profile_manager.build_user_profile(user_id, training_data[user_id])
            
            # Update user status
            user.profile_established = True
            db.session.commit()
            
            logger.info(f"Successfully trained models for user {user_id}")
            return True
        else:
            logger.warning(f"Model training failed for user {user_id}")
            return False
            
    except Exception as e:
        logger.error(f"Model training task failed for user {user_id}: {str(e)}")
        return False

@celery.task
def cleanup_old_data():
    """Clean up old behavioral data to manage storage"""
    try:
        # Delete events older than 30 days
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        old_keystrokes = KeystrokeEvent.query.filter(
            KeystrokeEvent.timestamp < cutoff_date
        ).count()
        
        old_mouse_events = MouseEvent.query.filter(
            MouseEvent.timestamp < cutoff_date
        ).count()
        
        # Delete in batches to avoid long locks
        batch_size = 1000
        
        # Clean keystrokes
        while True:
            batch = KeystrokeEvent.query.filter(
                KeystrokeEvent.timestamp < cutoff_date
            ).limit(batch_size).all()
            
            if not batch:
                break
                
            for event in batch:
                db.session.delete(event)
            db.session.commit()
        
        # Clean mouse events
        while True:
            batch = MouseEvent.query.filter(
                MouseEvent.timestamp < cutoff_date
            ).limit(batch_size).all()
            
            if not batch:
                break
                
            for event in batch:
                db.session.delete(event)
            db.session.commit()
        
        logger.info(f"Cleaned up {old_keystrokes} keystroke events and "
                   f"{old_mouse_events} mouse events older than 30 days")
        
        return True
        
    except Exception as e:
        logger.error(f"Data cleanup task failed: {str(e)}")
        return False

@celery.task
def generate_user_report(user_id: str):
    """Generate comprehensive behavioral analysis report for user"""
    try:
        logger.info(f"Generating report for user {user_id}")
        
        user = User.query.get(user_id)
        if not user:
            return None
        
        # Get user profile
        profile_manager = ProfileManager()
        profile = profile_manager.get_cached_profile(user_id)
        
        # Get session statistics
        sessions = BehavioralSession.query.filter_by(user_id=user_id).all()
        total_sessions = len(sessions)
        flagged_sessions = len([s for s in sessions if s.is_flagged])
        avg_risk_score = np.mean([s.risk_score for s in sessions]) if sessions else 0
        
        # Get recent alerts
        recent_alerts = AnomalyAlert.query.join(BehavioralSession).filter(
            BehavioralSession.user_id == user_id,
            AnomalyAlert.created_at >= datetime.utcnow() - timedelta(days=7)
        ).all()
        
        report = {
            'user_id': user_id,
            'username': user.username,
            'report_generated': datetime.utcnow().isoformat(),
            'profile_established': user.profile_established,
            'total_sessions': total_sessions,
            'flagged_sessions': flagged_sessions,
            'flagged_percentage': (flagged_sessions / total_sessions * 100) if total_sessions > 0 else 0,
            'average_risk_score': float(avg_risk_score),
            'recent_alerts': len(recent_alerts),
            'behavioral_profile': profile.to_dict() if profile else None,
            'risk_level': user.risk_level,
            'recommendations': []
        }
        
        # Add recommendations based on analysis
        if flagged_sessions / total_sessions > 0.1 if total_sessions > 0 else False:
            report['recommendations'].append(
                "High number of flagged sessions detected. Consider additional verification."
            )
        
        if avg_risk_score > 0.7:
            report['recommendations'].append(
                "Consistently high risk scores. Monitor user activity closely."
            )
        
        if len(recent_alerts) > 5:
            report['recommendations'].append(
                "Multiple recent anomaly alerts. Investigate recent behavioral changes."
            )
        
        # Store report (in production, you might store this in a reports table)
        if redis_client:
            redis_client.setex(
                f"report:{user_id}",
                timedelta(hours=24),
                json.dumps(report)
            )
        
        logger.info(f"Generated report for user {user_id}")
        return report
        
    except Exception as e:
        logger.error(f"Report generation failed for user {user_id}: {str(e)}")
        return None

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db.session.execute('SELECT 1')
        
        # Check Redis connection
        redis_status = "connected"
        if redis_client:
            try:
                redis_client.ping()
            except:
                redis_status = "disconnected"
        else:
            redis_status = "not_configured"
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'database': 'connected',
            'redis': redis_status,
            'version': '1.0.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/users', methods=['POST'])
def create_user():
    """Create a new user"""
    try:
        data = request.get_json()
        
        if not data or not data.get('username') or not data.get('email'):
            return jsonify({'error': 'Username and email are required'}), 400
        
        # Check if user already exists
        existing_user = User.query.filter(
            (User.username == data['username']) | (User.email == data['email'])
        ).first()
        
        if existing_user:
            return jsonify({'error': 'User with this username or email already exists'}), 409
        
        user = User(
            username=data['username'],
            email=data['email']
        )
        
        db.session.add(user)
        db.session.commit()
        
        logger.info(f"Created user: {user.username} ({user.id})")
        
        return jsonify({
            'user_id': user.id,
            'username': user.username,
            'message': 'User created successfully'
        }), 201
        
    except Exception as e:
        logger.error(f"User creation failed: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'User creation failed'}), 500

@app.route('/api/users/<user_id>', methods=['GET'])
def get_user(user_id):
    """Get user information"""
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify(user.to_dict())
        
    except Exception as e:
        logger.error(f"User retrieval failed: {str(e)}")
        return jsonify({'error': 'User retrieval failed'}), 500

@app.route('/api/sessions', methods=['POST'])
def create_session():
    """Create a new behavioral session"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Verify user exists
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get client information
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')
        
        session = BehavioralSession(
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            device_fingerprint=data.get('device_fingerprint')
        )
        
        db.session.add(session)
        db.session.commit()
        
        # Update user's last active time
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Created session {session.id} for user {user_id}")
        
        return jsonify({
            'session_id': session.id,
            'user_id': user_id,
            'created_at': session.session_start.isoformat()
        }), 201
        
    except Exception as e:
        logger.error(f"Session creation failed: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Session creation failed'}), 500

@app.route('/api/events', methods=['POST'])
async def process_event():
    """Process behavioral events in real-time"""
    try:
        event_data = request.get_json()
        
        if not event_data:
            return jsonify({'error': 'Event data is required'}), 400
        
        # Add client information to event
        event_data['ip_address'] = request.remote_addr
        event_data['user_agent'] = request.headers.get('User-Agent', '')
        
        # Process the event
        result = await real_time_processor.process_behavioral_event(event_data)
        
        if result.get('status') == 'error':
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Event processing failed: {str(e)}")
        return jsonify({'error': 'Event processing failed'}), 500

@app.route('/api/profiles/<user_id>', methods=['GET'])
def get_user_profile(user_id):
    """Get user behavioral profile"""
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        profile_manager = ProfileManager()
        profile = profile_manager.get_cached_profile(user_id)
        
        if not profile:
            return jsonify({'error': 'Profile not found or not established'}), 404
        
        return jsonify(profile.to_dict())
        
    except Exception as e:
        logger.error(f"Profile retrieval failed: {str(e)}")
        return jsonify({'error': 'Profile retrieval failed'}), 500

@app.route('/api/profiles/<user_id>/build', methods=['POST'])
def build_user_profile(user_id):
    """Manually trigger profile building for a user"""
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Trigger background task to build profile
        task = train_user_models.delay(user_id)
        
        return jsonify({
            'message': 'Profile building initiated',
            'task_id': task.id,
            'user_id': user_id
        })
        
    except Exception as e:
        logger.error(f"Profile building initiation failed: {str(e)}")
        return jsonify({'error': 'Profile building failed'}), 500

@app.route('/api/sessions/<session_id>/alerts', methods=['GET'])
def get_session_alerts(session_id):
    """Get alerts for a specific session"""
    try:
        session = BehavioralSession.query.get(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        alerts = AnomalyAlert.query.filter_by(session_id=session_id).order_by(
            AnomalyAlert.created_at.desc()
        ).all()
        
        return jsonify([alert.to_dict() for alert in alerts])
        
    except Exception as e:
        logger.error(f"Alert retrieval failed: {str(e)}")
        return jsonify({'error': 'Alert retrieval failed'}), 500

@app.route('/api/alerts/<alert_id>/resolve', methods=['PUT'])
def resolve_alert(alert_id):
    """Mark an alert as resolved"""
    try:
        alert = AnomalyAlert.query.get(alert_id)
        if not alert:
            return jsonify({'error': 'Alert not found'}), 404
        
        data = request.get_json()
        alert.resolved = True
        alert.resolved_at = datetime.utcnow()
        alert.false_positive = data.get('false_positive', False)
        
        db.session.commit()
        
        return jsonify({
            'message': 'Alert resolved successfully',
            'alert_id': alert_id
        })
        
    except Exception as e:
        logger.error(f"Alert resolution failed: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Alert resolution failed'}), 500

@app.route('/api/dashboard/stats', methods=['GET'])
def dashboard_stats():
    """Get dashboard statistics"""
    try:
        # Database statistics
        total_users = User.query.count()
        active_users = User.query.filter_by(is_active=True).count()
        established_profiles = User.query.filter_by(profile_established=True).count()
        
        # Session statistics
        total_sessions = BehavioralSession.query.count()
        active_sessions = BehavioralSession.query.filter(
            BehavioralSession.session_end.is_(None)
        ).count()
        
        flagged_sessions = BehavioralSession.query.filter_by(is_flagged=True).count()
        
        # Alert statistics
        total_alerts = AnomalyAlert.query.count()
        unresolved_alerts = AnomalyAlert.query.filter_by(resolved=False).count()
        recent_alerts = AnomalyAlert.query.filter(
            AnomalyAlert.created_at >= datetime.utcnow() - timedelta(hours=24)
        ).count()
        
        # Real-time processor statistics
        processor_stats = real_time_processor.get_statistics()
        
        return jsonify({
            'users': {
                'total': total_users,
                'active': active_users,
                'with_profiles': established_profiles,
                'profile_completion_rate': (established_profiles / total_users * 100) if total_users > 0 else 0
            },
            'sessions': {
                'total': total_sessions,
                'active': active_sessions,
                'flagged': flagged_sessions,
                'flagged_percentage': (flagged_sessions / total_sessions * 100) if total_sessions > 0 else 0
            },
            'alerts': {
                'total': total_alerts,
                'unresolved': unresolved_alerts,
                'last_24h': recent_alerts
            },
            'processing': processor_stats,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Dashboard stats retrieval failed: {str(e)}")
        return jsonify({'error': 'Stats retrieval failed'}), 500

@app.route('/api/reports/<user_id>', methods=['GET'])
def get_user_report(user_id):
    """Get comprehensive user analysis report"""
    try:
        # Check if report is cached
        if redis_client:
            cached_report = redis_client.get(f"report:{user_id}")
            if cached_report:
                return jsonify(json.loads(cached_report))
        
        # Generate new report
        task = generate_user_report.delay(user_id)
        
        return jsonify({
            'message': 'Report generation initiated',
            'task_id': task.id,
            'user_id': user_id,
            'check_url': f'/api/tasks/{task.id}'
        })
        
    except Exception as e:
        logger.error(f"Report retrieval failed: {str(e)}")
        return jsonify({'error': 'Report retrieval failed'}), 500

@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get status of background task"""
    try:
        task = celery.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'status': 'Task is waiting to be processed'
            }
        elif task.state == 'PROGRESS':
            response = {
                'state': task.state,
                'status': task.info.get('status', ''),
                'progress': task.info.get('progress', 0)
            }
        elif task.state == 'SUCCESS':
            response = {
                'state': task.state,
                'result': task.result
            }
        else:  # FAILURE
            response = {
                'state': task.state,
                'error': str(task.info)
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Task status retrieval failed: {str(e)}")
        return jsonify({'error': 'Task status retrieval failed'}), 500

# WebSocket Events
@sio.event
def connect(sid, environ):
    """Handle WebSocket connection"""
    logger.info(f'Client {sid} connected')
    sio.emit('connected', {
        'message': 'Connected to fraud detection system',
        'sid': sid,
        'timestamp': datetime.utcnow().isoformat()
    })

@sio.event
def disconnect(sid):
    """Handle WebSocket disconnection"""
    logger.info(f'Client {sid} disconnected')

@sio.event
def join_user_room(sid, data):
    """Join user-specific room for targeted alerts"""
    user_id = data.get('user_id')
    if user_id:
        sio.enter_room(sid, f"user_{user_id}")
        sio.emit('joined_room', {
            'room': f"user_{user_id}",
            'message': 'Subscribed to user alerts'
        })

@sio.event
def leave_user_room(sid, data):
    """Leave user-specific room"""
    user_id = data.get('user_id')
    if user_id:
        sio.leave_room(sid, f"user_{user_id}")
        sio.emit('left_room', {
            'room': f"user_{user_id}",
            'message': 'Unsubscribed from user alerts'
        })

# Periodic cleanup task
def run_periodic_cleanup():
    """Run periodic cleanup tasks"""
    while True:
        try:
            time.sleep(300)  # Run every 5 minutes
            real_time_processor.cleanup_inactive_sessions()
        except Exception as e:
            logger.error(f"Periodic cleanup failed: {str(e)}")

# Initialize database and start background tasks
def initialize_application():
    """Initialize the application"""
    with app.app_context():
        try:
            # Create database tables
            db.create_all()
            logger.info("Database tables created successfully")
            
            # Start periodic cleanup in background thread
            cleanup_thread = threading.Thread(target=run_periodic_cleanup, daemon=True)
            cleanup_thread.start()
            logger.info("Background cleanup thread started")
            
        except Exception as e:
            logger.error(f"Application initialization failed: {str(e)}")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({'error': 'An unexpected error occurred'}), 500


if __name__ == '__main__':

    initialize_application()
    

    from flask_socketio import SocketIOServer
    
    
    logger.info("Starting Behavioral Fraud Detection Backend...")
    logger.info(f"Database: {Config.DATABASE_URL}")
    logger.info(f"Redis: {Config.REDIS_URL}")
    
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,  
        threaded=True
    )