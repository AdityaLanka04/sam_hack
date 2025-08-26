# main.py - Complete Enhanced Fraud Detection with Ollama Agents
import os
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

# FastAPI
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Database
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Boolean, Text, ForeignKey, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.pool import StaticPool

# ML libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
import joblib
from scipy import stats

# Redis and Celery
try:
    import redis.asyncio as aioredis
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

# Agent libraries
try:
    import ollama
    # Try new import first, fall back to deprecated one
    try:
        from langchain_ollama import OllamaLLM as Ollama
    except ImportError:
        from langchain_community.llms import Ollama
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

# Configuration
class Settings:
    app_name = "Behavioral Fraud Detection API with AI Agents"
    version = "2.0.0"
    debug = True
    
    database_url = os.getenv("DATABASE_URL", "sqlite:///./fraud_detection.db")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    celery_broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    
    anomaly_threshold = 0.6
    min_training_samples = 50
    max_session_duration = 3600
    
    # Ollama settings
    ollama_model = "llama3:latest"
    ollama_base_url = "http://localhost:11434"

settings = Settings()

# Early logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database Setup
engine = create_engine(
    settings.database_url,
    poolclass=StaticPool if "sqlite" in settings.database_url else None,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Pydantic Models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime
    is_active: bool
    profile_established: bool
    risk_level: str

class SessionCreate(BaseModel):
    user_id: str
    device_fingerprint: Optional[str] = None

class SessionResponse(BaseModel):
    id: str
    user_id: str
    session_start: datetime
    risk_score: float

class BehavioralEvent(BaseModel):
    type: str
    user_id: str
    session_id: str
    timestamp: float
    
    # Keystroke fields
    key: Optional[str] = None
    key_code: Optional[int] = None
    dwell_time: Optional[float] = None
    flight_time: Optional[float] = None
    
    # Mouse fields
    x: Optional[int] = None
    y: Optional[int] = None
    velocity: Optional[float] = None
    acceleration: Optional[float] = None
    mouse_event_type: Optional[str] = None

class EventResponse(BaseModel):
    status: str
    session_id: str
    events_in_session: int
    anomaly_detected: bool = False
    risk_score: float = 0.0
    risk_level: str = "low"
    agent_analysis: Optional[Dict[str, Any]] = None

# Data Classes
@dataclass
class KeystrokeDynamics:
    user_id: str
    session_id: str
    key: str
    key_code: int
    dwell_time: float
    flight_time: float
    timestamp: float

@dataclass
class MouseBehavior:
    user_id: str
    session_id: str
    x: int
    y: int
    velocity: float
    acceleration: float
    timestamp: float
    event_type: str

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    profile_established = Column(Boolean, default=False)
    risk_level = Column(String(20), default='low')
    
    sessions = relationship("BehavioralSession", back_populates="user")

class BehavioralSession(Base):
    __tablename__ = "behavioral_sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    session_start = Column(DateTime, default=datetime.utcnow)
    session_end = Column(DateTime)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    device_fingerprint = Column(Text)
    risk_score = Column(Float, default=0.0)
    anomaly_count = Column(Integer, default=0)
    is_flagged = Column(Boolean, default=False)
    total_keystrokes = Column(Integer, default=0)
    total_mouse_events = Column(Integer, default=0)
    
    user = relationship("User", back_populates="sessions")
    anomaly_alerts = relationship("AnomalyAlert", back_populates="session")

class AnomalyAlert(Base):
    __tablename__ = "anomaly_alerts"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey('behavioral_sessions.id'), nullable=False)
    anomaly_type = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=False)
    severity = Column(String(20), default='medium')
    details = Column(Text)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("BehavioralSession", back_populates="anomaly_alerts")

class AgentAssessment(Base):
    __tablename__ = "agent_assessments"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey('behavioral_sessions.id'), nullable=False)
    agent_type = Column(String(50), nullable=False)
    assessment_data = Column(Text, nullable=False)
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

# Feature Engineering
class BehavioralFeatureExtractor:
    def __init__(self):
        self.scaler = RobustScaler()
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when no data is available"""
        return {
            'dwell_mean': 0.0,
            'dwell_std': 0.0,
            'flight_mean': 0.0,
            'flight_std': 0.0,
            'typing_speed': 0.0,
            'rhythm_consistency': 0.0,
            'keystroke_entropy': 0.0
        }
    
    def extract_keystroke_features(self, keystroke_data: List[KeystrokeDynamics]) -> Dict[str, float]:
        """Extract keystroke dynamics features"""
        if not keystroke_data or len(keystroke_data) < 2:
            return self._get_default_features()
        
        dwell_times = [k.dwell_time for k in keystroke_data if k.dwell_time > 0]
        flight_times = [k.flight_time for k in keystroke_data if k.flight_time > 0]
        
        if not dwell_times:
            return self._get_default_features()
        
        # Calculate typing speed
        duration = keystroke_data[-1].timestamp - keystroke_data[0].timestamp
        typing_speed = len(keystroke_data) / (duration / 60.0) if duration > 0 else 0
        
        # Calculate rhythm consistency
        rhythm_consistency = 1.0 / (np.std(flight_times) + 1e-6) if flight_times else 0
        
        # Calculate keystroke entropy
        keys = [k.key for k in keystroke_data if k.key]
        if keys:
            unique, counts = np.unique(keys, return_counts=True)
            probabilities = counts / len(keys)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        else:
            entropy = 0.0
        
        return {
            'dwell_mean': np.mean(dwell_times),
            'dwell_std': np.std(dwell_times),
            'flight_mean': np.mean(flight_times) if flight_times else 0,
            'flight_std': np.std(flight_times) if flight_times else 0,
            'typing_speed': typing_speed,
            'rhythm_consistency': rhythm_consistency,
            'keystroke_entropy': entropy
        }
    
    def extract_mouse_features(self, mouse_data: List[MouseBehavior]) -> Dict[str, float]:
        """Extract mouse behavior features"""
        if not mouse_data or len(mouse_data) < 2:
            return {
                'velocity_mean': 0.0, 'velocity_std': 0.0, 'acceleration_mean': 0.0,
                'movement_efficiency': 0.0, 'click_frequency': 0.0, 'movement_area': 0.0
            }
        
        velocities = [m.velocity for m in mouse_data if m.velocity >= 0]
        accelerations = [m.acceleration for m in mouse_data]
        x_positions = [m.x for m in mouse_data]
        y_positions = [m.y for m in mouse_data]
        
        # Movement efficiency
        if len(x_positions) >= 2:
            direct_distance = np.sqrt((x_positions[-1] - x_positions[0])**2 + 
                                    (y_positions[-1] - y_positions[0])**2)
            actual_distance = sum(
                np.sqrt((x_positions[i] - x_positions[i-1])**2 + 
                       (y_positions[i] - y_positions[i-1])**2)
                for i in range(1, len(x_positions))
            )
            efficiency = direct_distance / (actual_distance + 1e-6) if actual_distance > 0 else 1.0
        else:
            efficiency = 1.0
        
        # Click frequency
        clicks = len([m for m in mouse_data if m.event_type == 'click'])
        click_frequency = clicks / len(mouse_data)
        
        # Movement area
        if x_positions and y_positions:
            area = (max(x_positions) - min(x_positions)) * (max(y_positions) - min(y_positions))
        else:
            area = 0.0
        
        return {
            'velocity_mean': np.mean(velocities) if velocities else 0.0,
            'velocity_std': np.std(velocities) if velocities else 0.0,
            'acceleration_mean': np.mean(accelerations) if accelerations else 0.0,
            'movement_efficiency': efficiency,
            'click_frequency': click_frequency,
            'movement_area': area
        }

# Anomaly Detection Engine
class AnomalyDetectionEngine:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.one_class_svm = None
        self.scaler = RobustScaler()
        self.feature_extractor = BehavioralFeatureExtractor()
        self.models_trained = False
        self.baseline_profiles = {}
        self.anomaly_history = defaultdict(deque)
    
    async def train_models(self, training_data: Dict) -> bool:
        """Train anomaly detection models"""
        try:
            feature_matrix = self._prepare_training_features(training_data)
            
            if len(feature_matrix) < settings.min_training_samples:
                logger.warning(f"Insufficient training data: {len(feature_matrix)} samples")
                return False
            
            # Scale and train
            scaled_features = self.scaler.fit_transform(feature_matrix)
            self.isolation_forest.fit(scaled_features)
            
            self.one_class_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
            self.one_class_svm.fit(scaled_features)
            
            self._build_baseline_profiles(training_data)
            self.models_trained = True
            
            logger.info(f"Models trained with {len(feature_matrix)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return False
    
    def _prepare_training_features(self, training_data: Dict) -> np.ndarray:
        """Prepare feature matrix from training data"""
        features = []
        
        for user_id, sessions in training_data.items():
            for session_data in sessions.values():
                keystroke_features = self.feature_extractor.extract_keystroke_features(
                    session_data.get('keystrokes', [])
                )
                mouse_features = self.feature_extractor.extract_mouse_features(
                    session_data.get('mouse', [])
                )
                
                combined = {**keystroke_features, **mouse_features}
                if combined and any(v != 0 for v in combined.values()):
                    features.append(list(combined.values()))
        
        return np.array(features) if features else np.array([])
    
    def _build_baseline_profiles(self, training_data: Dict):
        """Build baseline profiles for each user"""
        for user_id, sessions in training_data.items():
            user_features = []
            
            for session_data in sessions.values():
                keystroke_features = self.feature_extractor.extract_keystroke_features(
                    session_data.get('keystrokes', [])
                )
                mouse_features = self.feature_extractor.extract_mouse_features(
                    session_data.get('mouse', [])
                )
                
                combined = {**keystroke_features, **mouse_features}
                if combined:
                    user_features.append(list(combined.values()))
            
            if user_features:
                feature_matrix = np.array(user_features)
                self.baseline_profiles[user_id] = {
                    'mean': np.mean(feature_matrix, axis=0),
                    'std': np.std(feature_matrix, axis=0)
                }
    
    async def detect_anomaly(self, user_id: str, current_data: Dict) -> Dict[str, Any]:
        """Detect anomalies in current behavioral data"""
        try:
            # Quick rule-based detection for immediate results
            keystrokes = current_data.get('keystrokes', [])
            mouse_data = current_data.get('mouse', [])
            
            # Bot detection based on keystroke patterns
            if keystrokes and len(keystrokes) >= 3:
                recent_keystrokes = keystrokes[-10:]
                
                dwell_times = [k.dwell_time for k in recent_keystrokes if k.dwell_time > 0]
                flight_times = [k.flight_time for k in recent_keystrokes if k.flight_time > 0]
                
                if dwell_times and flight_times:
                    avg_dwell = sum(dwell_times) / len(dwell_times)
                    avg_flight = sum(flight_times) / len(flight_times)
                    
                    # Bot detection - very short timing patterns
                    if avg_dwell < 20 or avg_flight < 15:
                        return {
                            'anomaly_score': 0.95,
                            'is_anomaly': True,
                            'confidence': 0.95,
                            'risk_level': 'critical',
                            'reason': 'bot_detection'
                        }
                    
                    # Consistency check - too uniform is suspicious
                    dwell_consistency = np.std(dwell_times) / (np.mean(dwell_times) + 1e-6)
                    flight_consistency = np.std(flight_times) / (np.mean(flight_times) + 1e-6)
                    
                    if dwell_consistency < 0.1 and flight_consistency < 0.1:
                        return {
                            'anomaly_score': 0.85,
                            'is_anomaly': True,
                            'confidence': 0.8,
                            'risk_level': 'high',
                            'reason': 'too_consistent'
                        }
            
            # Mouse behavior anomaly detection
            if mouse_data and len(mouse_data) >= 5:
                recent_mouse = mouse_data[-10:]
                velocities = [m.velocity for m in recent_mouse if m.velocity >= 0]
                
                if velocities:
                    avg_velocity = sum(velocities) / len(velocities)
                    
                    # Extremely fast or slow mouse movement
                    if avg_velocity > 1000:  # Too fast
                        return {
                            'anomaly_score': 0.8,
                            'is_anomaly': True,
                            'confidence': 0.75,
                            'risk_level': 'high',
                            'reason': 'erratic_mouse'
                        }
                    
                    # Check for linear patterns (bot-like)
                    x_coords = [m.x for m in recent_mouse]
                    y_coords = [m.y for m in recent_mouse]
                    
                    if len(set(x_coords)) <= 2 or len(set(y_coords)) <= 2:
                        return {
                            'anomaly_score': 0.9,
                            'is_anomaly': True,
                            'confidence': 0.85,
                            'risk_level': 'critical',
                            'reason': 'linear_mouse_pattern'
                        }
            
            # ML-based detection if models are trained
            if self.models_trained and (keystrokes or mouse_data):
                ml_result = await self._ml_anomaly_detection(user_id, current_data)
                if ml_result.get('is_anomaly'):
                    return ml_result
            
            # Baseline comparison
            if user_id in self.baseline_profiles:
                baseline_score = self._compare_with_baseline(user_id, current_data)
                if baseline_score > 0.7:
                    return {
                        'anomaly_score': baseline_score,
                        'is_anomaly': True,
                        'confidence': 0.7,
                        'risk_level': 'medium',
                        'reason': 'baseline_deviation'
                    }
            
            return {
                'anomaly_score': 0.1,
                'is_anomaly': False,
                'confidence': 0.9,
                'risk_level': 'low',
                'reason': 'normal_behavior'
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return {
                'anomaly_score': 0.0,
                'is_anomaly': False,
                'confidence': 0.0,
                'risk_level': 'low',
                'reason': 'detection_error'
            }
    
    async def _ml_anomaly_detection(self, user_id: str, current_data: Dict) -> Dict[str, Any]:
        """ML-based anomaly detection"""
        try:
            # Extract features
            keystroke_features = self.feature_extractor.extract_keystroke_features(
                current_data.get('keystrokes', [])
            )
            mouse_features = self.feature_extractor.extract_mouse_features(
                current_data.get('mouse', [])
            )
            
            combined_features = {**keystroke_features, **mouse_features}
            if not combined_features or not any(v != 0 for v in combined_features.values()):
                return {'is_anomaly': False, 'anomaly_score': 0.0}
            
            # Prepare feature vector
            feature_vector = np.array(list(combined_features.values())).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_vector)
            
            # Get predictions
            isolation_score = self.isolation_forest.decision_function(scaled_features)[0]
            isolation_pred = self.isolation_forest.predict(scaled_features)[0]
            
            svm_pred = self.one_class_svm.predict(scaled_features)[0] if self.one_class_svm else 1
            
            # Combine scores
            is_anomaly = isolation_pred == -1 or svm_pred == -1
            anomaly_score = max(0, min(1, (0.5 - isolation_score) * 2))  # Normalize to 0-1
            
            risk_level = 'low'
            if anomaly_score > 0.8:
                risk_level = 'critical'
            elif anomaly_score > 0.6:
                risk_level = 'high'
            elif anomaly_score > 0.4:
                risk_level = 'medium'
            
            return {
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly,
                'confidence': 0.8,
                'risk_level': risk_level,
                'reason': 'ml_detection'
            }
            
        except Exception as e:
            logger.error(f"ML detection failed: {str(e)}")
            return {'is_anomaly': False, 'anomaly_score': 0.0}
    
    def _compare_with_baseline(self, user_id: str, current_data: Dict) -> float:
        """Compare current behavior with user baseline"""
        if user_id not in self.baseline_profiles:
            return 0.0
        
        try:
            # Extract current features
            keystroke_features = self.feature_extractor.extract_keystroke_features(
                current_data.get('keystrokes', [])
            )
            mouse_features = self.feature_extractor.extract_mouse_features(
                current_data.get('mouse', [])
            )
            
            combined_features = {**keystroke_features, **mouse_features}
            if not combined_features:
                return 0.0
            
            features = np.array(list(combined_features.values()))
            baseline = self.baseline_profiles[user_id]
            
            # Calculate z-scores
            z_scores = []
            for i, value in enumerate(features):
                if i < len(baseline['std']) and baseline['std'][i] > 0:
                    z_score = abs(value - baseline['mean'][i]) / baseline['std'][i]
                    z_scores.append(z_score)
            
            if not z_scores:
                return 0.0
            
            # Count anomalous features (z-score > 2)
            anomalous_features = sum(1 for z in z_scores if z > 2.0)
            return anomalous_features / len(z_scores)
            
        except Exception as e:
            logger.error(f"Baseline comparison failed: {str(e)}")
            return 0.0

# AI Security Agent
class SecurityAgent:
    def __init__(self):
        if not AGENTS_AVAILABLE:
            self.available = False
            logger.warning("Agent libraries not available. Install: pip install ollama langchain langchain-community")
            return
            
        try:
            self.llm = Ollama(model="llama3:latest", base_url="http://localhost:11434")
            self.available = True
            logger.info("SecurityAgent initialized with llama3:latest")
        except Exception as e:
            logger.warning(f"SecurityAgent initialization failed: {e}")
            self.available = False
    
    async def analyze_behavioral_risk(self, session_data: Dict) -> Dict[str, Any]:
        """Analyze behavioral data using AI agent"""
        if not self.available:
            return {"error": "Agent not available", "fallback": True}
        
        try:
            behavioral_summary = self._prepare_behavioral_summary(session_data)
            
            prompt = f"""You are a cybersecurity expert analyzing user behavioral patterns for fraud detection.

Behavioral Data:
{json.dumps(behavioral_summary, indent=2)}

Analyze this data for fraud indicators:
1. Keystroke timing patterns (normal human dwell: 80-150ms, flight: 50-200ms)
2. Mouse movement patterns (velocity, acceleration, linearity)
3. Overall behavioral consistency

Provide assessment in JSON format:
{{"risk_level": "low", "confidence": 0.8, "primary_concerns": ["issues"], "reasoning": "explanation", "recommendations": ["actions"]}}

Respond ONLY with valid JSON."""
            
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            
            try:
                analysis = json.loads(response.strip())
                analysis["agent_model"] = "llama3:latest"
                analysis["timestamp"] = datetime.utcnow().isoformat()
                return analysis
            except json.JSONDecodeError:
                return {
                    "risk_level": "unknown",
                    "confidence": 0.5,
                    "reasoning": response[:200],
                    "agent_model": "llama3:latest",
                    "parsing_error": True
                }
                
        except Exception as e:
            logger.error(f"Agent analysis failed: {str(e)}")
            return {"error": str(e), "agent_model": "llama3:latest"}
    
    def _prepare_behavioral_summary(self, session_data: Dict) -> Dict[str, Any]:
        keystrokes = session_data.get('keystrokes', [])
        mouse_events = session_data.get('mouse', [])
        
        summary = {
            "session_stats": {
                "keystroke_count": len(keystrokes),
                "mouse_event_count": len(mouse_events),
                "session_duration": time.time() - session_data.get('start_time', time.time()),
                "current_risk_score": session_data.get('risk_score', 0.0),
                "anomaly_count": session_data.get('anomaly_count', 0)
            }
        }
        
        if keystrokes:
            recent = keystrokes[-10:]
            dwell_times = [k.dwell_time for k in recent if hasattr(k, 'dwell_time') and k.dwell_time > 0]
            flight_times = [k.flight_time for k in recent if hasattr(k, 'flight_time') and k.flight_time > 0]
            
            if dwell_times and flight_times:
                summary["keystroke_patterns"] = {
                    "avg_dwell_time": sum(dwell_times) / len(dwell_times),
                    "avg_flight_time": sum(flight_times) / len(flight_times),
                    "dwell_std": np.std(dwell_times),
                    "consistency_ratio": np.std(dwell_times) / (np.mean(dwell_times) + 1e-6)
                }
        
        if mouse_events:
            recent_mouse = mouse_events[-10:]
            velocities = [m.velocity for m in recent_mouse if hasattr(m, 'velocity') and m.velocity >= 0]
            
            if velocities:
                summary["mouse_patterns"] = {
                    "avg_velocity": sum(velocities) / len(velocities),
                    "velocity_std": np.std(velocities),
                    "max_velocity": max(velocities),
                    "movement_count": len(recent_mouse)
                }
        
        return summary

# Real-time Processor
class RealTimeProcessor:
    def __init__(self):
        self.anomaly_engine = AnomalyDetectionEngine()
        self.active_sessions = {}
        self.stats = {
            'events_processed': 0,
            'anomalies_detected': 0,
            'sessions_active': 0,
            'last_reset': datetime.utcnow()
        }
    
    async def process_behavioral_event(self, event_data: Dict, db: Session) -> Dict[str, Any]:
        """Process behavioral events in real-time"""
        try:
            user_id = event_data.get('user_id')
            session_id = event_data.get('session_id')
            event_type = event_data.get('type')
            
            if not all([user_id, session_id, event_type]):
                raise HTTPException(status_code=400, detail="Missing required fields")
            
            # Get or create session
            session = self._get_or_create_session(user_id, session_id)
            
            # Process event
            if event_type == 'keystroke':
                keystroke = KeystrokeDynamics(
                    user_id=user_id,
                    session_id=session_id,
                    key=event_data.get('key', ''),
                    key_code=event_data.get('key_code', 0),
                    dwell_time=event_data.get('dwell_time', 0.0),
                    flight_time=event_data.get('flight_time', 0.0),
                    timestamp=event_data.get('timestamp', time.time())
                )
                session['keystrokes'].append(keystroke)
                
            elif event_type == 'mouse':
                mouse_event = MouseBehavior(
                    user_id=user_id,
                    session_id=session_id,
                    x=event_data.get('x', 0),
                    y=event_data.get('y', 0),
                    velocity=event_data.get('velocity', 0.0),
                    acceleration=event_data.get('acceleration', 0.0),
                    timestamp=event_data.get('timestamp', time.time()),
                    event_type=event_data.get('mouse_event_type', 'move')
                )
                session['mouse'].append(mouse_event)
            
            # Update stats
            self.stats['events_processed'] += 1
            
            # Check for anomaly detection
            detection_result = await self._check_anomaly_detection(session_id, session, db)
            
            return {
                'status': 'success',
                'session_id': session_id,
                'events_in_session': len(session['keystrokes']) + len(session['mouse']),
                **detection_result
            }
            
        except Exception as e:
            logger.error(f"Event processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _get_or_create_session(self, user_id: str, session_id: str) -> Dict:
        """Get or create session"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                'user_id': user_id,
                'keystrokes': [],
                'mouse': [],
                'start_time': time.time(),
                'last_activity': time.time(),
                'anomaly_count': 0,
                'risk_score': 0.0
            }
            self.stats['sessions_active'] += 1
        
        self.active_sessions[session_id]['last_activity'] = time.time()
        return self.active_sessions[session_id]
    
    async def _check_anomaly_detection(self, session_id: str, session: Dict, db: Session) -> Dict[str, Any]:
        """Check if we should perform anomaly detection"""
        total_events = len(session['keystrokes']) + len(session['mouse'])
        
        if total_events > 10 and (total_events % 20 == 0):
            try:
                detection_result = await self.anomaly_engine.detect_anomaly(
                    session['user_id'], session
                )
                
                session['risk_score'] = detection_result['anomaly_score']
                
                if detection_result['is_anomaly']:
                    session['anomaly_count'] += 1
                    self.stats['anomalies_detected'] += 1
                    
                    # Store alert
                    await self._store_anomaly_alert(session_id, detection_result, db)
                
                return {
                    'anomaly_detected': detection_result['is_anomaly'],
                    'risk_score': detection_result['anomaly_score'],
                    'confidence': detection_result.get('confidence', 0.0),
                    'risk_level': detection_result['risk_level']
                }
                
            except Exception as e:
                logger.error(f"Anomaly detection failed: {str(e)}")
                return {'anomaly_detected': False, 'risk_score': 0.0, 'risk_level': 'low'}
        
        return {
            'anomaly_detected': False, 
            'risk_score': session.get('risk_score', 0.0),
            'risk_level': 'low',
            'reason': 'insufficient_data'
        }
    
    async def _store_anomaly_alert(self, session_id: str, detection_result: Dict, db: Session):
        """Store anomaly alert"""
        try:
            alert = AnomalyAlert(
                session_id=session_id,
                anomaly_type='behavioral',
                confidence_score=detection_result.get('confidence', 0.0),
                severity=detection_result['risk_level'],
                details=json.dumps(detection_result)
            )
            
            db.add(alert)
            db.commit()
            
            # Broadcast via WebSocket
            await websocket_manager.broadcast_alert({
                'session_id': session_id,
                'risk_level': detection_result['risk_level'],
                'confidence': detection_result.get('confidence', 0.0),
                'timestamp': datetime.utcnow().isoformat(),
                'reason': detection_result.get('reason', 'unknown')
            })
            
        except Exception as e:
            logger.error(f"Alert storage failed: {str(e)}")
            db.rollback()

# Enhanced Real-time Processor with Agent Integration
class EnhancedRealTimeProcessor(RealTimeProcessor):
    def __init__(self):
        super().__init__()
        self.security_agent = SecurityAgent() if AGENTS_AVAILABLE else None
        self.stats['agent_analyses'] = 0
    
    async def process_behavioral_event_with_agent(self, event_data: Dict, db: Session) -> Dict[str, Any]:
        """Process events with AI agent analysis"""
        
        # First run the standard processing
        standard_result = await super().process_behavioral_event(event_data, db)
        
        session_id = event_data.get('session_id')
        session_data = self.active_sessions.get(session_id, {})
        
        # Run AI agent analysis periodically or on anomalies
        total_events = len(session_data.get('keystrokes', [])) + len(session_data.get('mouse', []))
        agent_analysis = None
        
        if (total_events > 0 and total_events % 25 == 0) or standard_result.get('anomaly_detected'):
            if self.security_agent and self.security_agent.available:
                try:
                    agent_analysis = await self.security_agent.analyze_behavioral_risk(session_data)
                    self.stats['agent_analyses'] += 1
                    
                    # Store agent assessment
                    await self._store_agent_assessment(session_id, agent_analysis, db)
                    
                except Exception as e:
                    logger.error(f"Agent analysis failed: {str(e)}")
                    agent_analysis = {"error": "Agent analysis failed"}
        
        if agent_analysis:
            standard_result['agent_analysis'] = agent_analysis
        
        return standard_result
    
    async def _store_agent_assessment(self, session_id: str, assessment: Dict, db: Session):
        """Store AI agent assessment"""
        try:
            agent_assessment = AgentAssessment(
                session_id=session_id,
                agent_type='security_agent',
                assessment_data=json.dumps(assessment),
                confidence=assessment.get('confidence', 0.0)
            )
            
            db.add(agent_assessment)
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to store agent assessment: {str(e)}")
            db.rollback()

# WebSocket Manager
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast_alert(self, alert_data: Dict):
        if not self.active_connections:
            return
        
        message = {'type': 'anomaly_alert', 'data': alert_data}
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"WebSocket send failed: {str(e)}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

# Initialize global instances
enhanced_processor = EnhancedRealTimeProcessor()
websocket_manager = WebSocketManager()

# Celery setup
if CELERY_AVAILABLE:
    celery_app = Celery(
        "fraud_detection",
        broker=settings.celery_broker_url,
        backend=settings.celery_broker_url
    )

    @celery_app.task
    def train_user_models(user_id: str):
        """Background task to train models"""
        logger.info(f"Training models for user {user_id}")
        
        try:
            db = SessionLocal()
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                return False
            
            # Get training data
            sessions = db.query(BehavioralSession).filter(BehavioralSession.user_id == user_id).all()
            
            if sessions:
                # Build training data from active sessions
                training_data = {user_id: {}}
                for session in sessions:
                    if session.id in enhanced_processor.active_sessions:
                        training_data[user_id][session.id] = enhanced_processor.active_sessions[session.id]
                
                # Train models
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(enhanced_processor.anomaly_engine.train_models(training_data))
                
                if success:
                    user.profile_established = True
                    db.commit()
                    logger.info(f"Models trained for user {user_id}")
                
                db.close()
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# App Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Enhanced Behavioral Fraud Detection Backend...")
    
    # Test Ollama connection
    if AGENTS_AVAILABLE:
        try:
            ollama.list()
            logger.info("Ollama connection successful")
        except Exception as e:
            logger.warning(f"Ollama connection failed: {e}")
            logger.warning("Make sure Ollama is running: 'ollama serve'")
    else:
        logger.warning("Agent libraries not available")
    
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Real-time behavioral fraud detection with AI agents",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/")
async def root():
    return {
        "message": "Enhanced Behavioral Fraud Detection API with AI Agents",
        "version": settings.version,
        "docs": "/docs",
        "agent_model": settings.ollama_model,
        "agents_available": AGENTS_AVAILABLE
    }

@app.get("/health")
async def health_check():
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        
        # Test Ollama connection
        ollama_status = "disconnected"
        if AGENTS_AVAILABLE:
            try:
                ollama.list()
                ollama_status = "connected"
            except:
                ollama_status = "disconnected"
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected",
            "ollama": ollama_status,
            "agents_available": AGENTS_AVAILABLE,
            "version": settings.version,
            "active_sessions": len(enhanced_processor.active_sessions),
            "events_processed": enhanced_processor.stats['events_processed']
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/api/users", response_model=UserResponse)
async def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    try:
        existing = db.query(User).filter(
            (User.username == user_data.username) | (User.email == user_data.email)
        ).first()
        
        if existing:
            raise HTTPException(status_code=409, detail="User already exists")
        
        user = User(username=user_data.username, email=user_data.email)
        db.add(user)
        db.commit()
        db.refresh(user)
        
        logger.info(f"Created user: {user.username}")
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            created_at=user.created_at,
            is_active=user.is_active,
            profile_established=user.profile_established,
            risk_level=user.risk_level
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"User creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="User creation failed")

@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(session_data: SessionCreate, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.id == session_data.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        session = BehavioralSession(
            user_id=session_data.user_id,
            ip_address="127.0.0.1",
            device_fingerprint=session_data.device_fingerprint
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        logger.info(f"Created session {session.id} for user {session_data.user_id}")
        
        return SessionResponse(
            id=session.id,
            user_id=session.user_id,
            session_start=session.session_start,
            risk_score=session.risk_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Session creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/events", response_model=EventResponse)
async def process_event(
    event_data: BehavioralEvent,
    db: Session = Depends(get_db)
):
    """Process behavioral events (original endpoint)"""
    try:
        event_dict = event_data.dict()
        result = await enhanced_processor.process_behavioral_event(event_dict, db)
        return EventResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Event processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Event processing failed")

@app.post("/api/events/enhanced", response_model=EventResponse)
async def process_event_with_agent(
    event_data: BehavioralEvent,
    db: Session = Depends(get_db)
):
    """Process behavioral events with AI agent analysis"""
    try:
        event_dict = event_data.dict()
        result = await enhanced_processor.process_behavioral_event_with_agent(event_dict, db)
        return EventResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced event processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Enhanced event processing failed")

@app.get("/api/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        created_at=user.created_at,
        is_active=user.is_active,
        profile_established=user.profile_established,
        risk_level=user.risk_level
    )

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(BehavioralSession).filter(BehavioralSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get real-time data if available
    real_time_data = enhanced_processor.active_sessions.get(session_id, {})
    
    return {
        "id": session.id,
        "user_id": session.user_id,
        "session_start": session.session_start.isoformat(),
        "risk_score": session.risk_score,
        "anomaly_count": session.anomaly_count,
        "is_flagged": session.is_flagged,
        "real_time": {
            "keystrokes_count": len(real_time_data.get('keystrokes', [])),
            "mouse_events_count": len(real_time_data.get('mouse', [])),
            "current_risk_score": real_time_data.get('risk_score', 0.0),
            "anomaly_count": real_time_data.get('anomaly_count', 0)
        }
    }

@app.get("/api/sessions/{session_id}/alerts")
async def get_session_alerts(session_id: str, db: Session = Depends(get_db)):
    session = db.query(BehavioralSession).filter(BehavioralSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    alerts = db.query(AnomalyAlert).filter(
        AnomalyAlert.session_id == session_id
    ).order_by(AnomalyAlert.created_at.desc()).all()
    
    return [
        {
            "id": alert.id,
            "session_id": alert.session_id,
            "anomaly_type": alert.anomaly_type,
            "confidence_score": alert.confidence_score,
            "severity": alert.severity,
            "details": json.loads(alert.details) if alert.details else {},
            "resolved": alert.resolved,
            "created_at": alert.created_at.isoformat()
        } for alert in alerts
    ]

@app.post("/api/profiles/{user_id}/build")
async def build_user_profile(user_id: str, db: Session = Depends(get_db)):
    """Trigger profile building"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if CELERY_AVAILABLE:
        task = train_user_models.delay(user_id)
        return {
            "message": "Profile building initiated",
            "task_id": task.id,
            "user_id": user_id
        }
    else:
        return {
            "message": "Profile building not available (Celery not installed)",
            "user_id": user_id
        }

@app.get("/api/dashboard/stats")
async def enhanced_dashboard_stats(db: Session = Depends(get_db)):
    try:
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.is_active == True).count()
        total_sessions = db.query(BehavioralSession).count()
        total_alerts = db.query(AnomalyAlert).count()
        unresolved_alerts = db.query(AnomalyAlert).filter(AnomalyAlert.resolved == False).count()
        
        # Agent assessments count
        total_agent_assessments = 0
        if AGENTS_AVAILABLE:
            try:
                total_agent_assessments = db.query(AgentAssessment).count()
            except:
                pass  # Table might not exist yet
        
        processor_stats = enhanced_processor.stats.copy()
        uptime = datetime.utcnow() - processor_stats['last_reset']
        
        return {
            "users": {
                "total": total_users,
                "active": active_users
            },
            "sessions": {
                "total": total_sessions,
                "active": len(enhanced_processor.active_sessions)
            },
            "alerts": {
                "total": total_alerts,
                "unresolved": unresolved_alerts
            },
            "processing": {
                **processor_stats,
                "uptime_seconds": uptime.total_seconds(),
                "events_per_second": processor_stats['events_processed'] / max(uptime.total_seconds(), 1)
            },
            "ai_agent": {
                "available": AGENTS_AVAILABLE,
                "total_assessments": total_agent_assessments,
                "agent_analyses_count": processor_stats.get('agent_analyses', 0),
                "agent_ready": enhanced_processor.security_agent.available if enhanced_processor.security_agent else False,
                "model": settings.ollama_model
            },
            "dependencies": {
                "agents": AGENTS_AVAILABLE,
                "celery": CELERY_AVAILABLE
            }
        }
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Stats retrieval failed")

# AI Agent endpoints
@app.get("/api/test/agent-status")
async def get_agent_status():
    """Check AI agent status"""
    if not AGENTS_AVAILABLE:
        return {
            "agent_available": False,
            "error": "Agent libraries not installed",
            "install_command": "pip install ollama langchain langchain-community"
        }
    
    agent = enhanced_processor.security_agent
    
    # Test agent with a simple query
    test_result = None
    if agent and agent.available:
        try:
            test_data = {"session_stats": {"test": "connection"}}
            test_result = await agent.analyze_behavioral_risk(test_data)
        except Exception as e:
            test_result = {"error": str(e)}
    
    return {
        "agent_available": agent.available if agent else False,
        "model": settings.ollama_model,
        "ollama_base_url": settings.ollama_base_url,
        "test_result": test_result,
        "stats": {
            "total_analyses": enhanced_processor.stats.get('agent_analyses', 0)
        }
    }

@app.post("/api/agent/analyze-session/{session_id}")
async def analyze_session_with_agent(session_id: str, db: Session = Depends(get_db)):
    """Get AI agent analysis of a session"""
    
    if not AGENTS_AVAILABLE or not enhanced_processor.security_agent:
        raise HTTPException(status_code=503, detail="AI agent not available")
    
    session_data = enhanced_processor.active_sessions.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Active session not found")
    
    try:
        assessment = await enhanced_processor.security_agent.analyze_behavioral_risk(session_data)
        enhanced_processor.stats['agent_analyses'] += 1
        
        # Store the assessment
        await enhanced_processor._store_agent_assessment(session_id, assessment, db)
        
        return {
            "session_id": session_id,
            "assessment": assessment,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent analysis failed: {str(e)}")

@app.get("/api/agent/assessments/{session_id}")
async def get_agent_assessments(session_id: str, db: Session = Depends(get_db)):
    """Get all AI agent assessments for a session"""
    
    if not AGENTS_AVAILABLE:
        return []
    
    try:
        assessments = db.query(AgentAssessment).filter(
            AgentAssessment.session_id == session_id
        ).order_by(AgentAssessment.created_at.desc()).all()
        
        return [
            {
                "id": assessment.id,
                "session_id": assessment.session_id,
                "agent_type": assessment.agent_type,
                "assessment_data": json.loads(assessment.assessment_data),
                "confidence": assessment.confidence,
                "created_at": assessment.created_at.isoformat()
            } for assessment in assessments
        ]
    except Exception as e:
        logger.error(f"Failed to get agent assessments: {str(e)}")
        return []

# Testing endpoints
@app.post("/api/test/generate-events")
async def generate_test_events(
    user_id: str,
    session_id: str,
    event_count: int = 50,
    anomalous: bool = False,
    use_agent: bool = False,
    db: Session = Depends(get_db)
):
    """Generate test events with optional AI agent analysis"""
    try:
        session = db.query(BehavioralSession).filter(BehavioralSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        events_created = 0
        current_time = time.time()
        
        # Choose endpoint based on use_agent parameter
        endpoint_func = process_event_with_agent if use_agent and AGENTS_AVAILABLE else process_event
        
        # Generate keystroke events
        keystroke_count = event_count // 2
        for i in range(keystroke_count):
            if anomalous:
                dwell_time = np.random.uniform(5, 15)  # Bot-like
                flight_time = np.random.uniform(1, 10)
            else:
                dwell_time = np.random.uniform(80, 150)  # Normal
                flight_time = np.random.uniform(50, 200)
            
            event_data = BehavioralEvent(
                type="keystroke",
                user_id=user_id,
                session_id=session_id,
                key=chr(ord('a') + (i % 26)),
                key_code=65 + (i % 26),
                dwell_time=dwell_time,
                flight_time=flight_time,
                timestamp=current_time + i * 0.1
            )
            
            await endpoint_func(event_data, db)
            events_created += 1
        
        # Generate mouse events
        mouse_count = event_count - keystroke_count
        x, y = 500, 500
        for i in range(mouse_count):
            if anomalous:
                if i % 5 == 0:
                    dx, dy = 10, 0  # Linear patterns
                else:
                    dx, dy = np.random.randint(-200, 200, 2)
                velocity = np.random.uniform(800, 1500)
            else:
                dx, dy = np.random.randint(-30, 30, 2)
                velocity = np.random.uniform(50, 300)
            
            x = max(0, min(1920, x + dx))
            y = max(0, min(1080, y + dy))
            
            event_data = BehavioralEvent(
                type="mouse",
                user_id=user_id,
                session_id=session_id,
                x=x,
                y=y,
                velocity=velocity,
                acceleration=np.random.uniform(-100, 100),
                mouse_event_type='move',
                timestamp=current_time + keystroke_count * 0.1 + i * 0.05
            )
            
            await endpoint_func(event_data, db)
            events_created += 1
        
        return {
            "message": f"Generated {events_created} test events",
            "anomalous": anomalous,
            "agent_analysis": use_agent and AGENTS_AVAILABLE,
            "session_id": session_id,
            "keystroke_events": keystroke_count,
            "mouse_events": mouse_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Test generation failed")

@app.post("/api/test/reset-session/{session_id}")
async def reset_session(session_id: str):
    """Reset a session's real-time data"""
    if session_id in enhanced_processor.active_sessions:
        del enhanced_processor.active_sessions[session_id]
        enhanced_processor.stats['sessions_active'] = max(0, enhanced_processor.stats['sessions_active'] - 1)
        return {"message": f"Session {session_id} reset successfully"}
    else:
        raise HTTPException(status_code=404, detail="Active session not found")

@app.get("/api/test/sessions")
async def list_active_sessions():
    """List all active sessions"""
    sessions = []
    for session_id, data in enhanced_processor.active_sessions.items():
        sessions.append({
            "session_id": session_id,
            "user_id": data['user_id'],
            "start_time": data['start_time'],
            "keystrokes": len(data['keystrokes']),
            "mouse_events": len(data['mouse']),
            "risk_score": data['risk_score'],
            "anomaly_count": data['anomaly_count']
        })
    
    return {
        "active_sessions": sessions,
        "total_count": len(sessions)
    }

# Admin endpoints
@app.post("/api/admin/reset-stats")
async def reset_stats():
    """Reset processing statistics"""
    enhanced_processor.stats = {
        'events_processed': 0,
        'anomalies_detected': 0,
        'agent_analyses': 0,
        'sessions_active': len(enhanced_processor.active_sessions),
        'last_reset': datetime.utcnow()
    }
    return {"message": "Statistics reset successfully"}

@app.get("/api/admin/models/status")
async def model_status():
    """Get ML model status"""
    engine = enhanced_processor.anomaly_engine
    return {
        "models_trained": engine.models_trained,
        "baseline_profiles_count": len(engine.baseline_profiles),
        "isolation_forest_trained": engine.isolation_forest is not None,
        "one_class_svm_trained": engine.one_class_svm is not None,
        "scaler_fitted": hasattr(engine.scaler, 'mean_') and engine.scaler.mean_ is not None,
        "agents_available": AGENTS_AVAILABLE,
        "agent_ready": enhanced_processor.security_agent.available if enhanced_processor.security_agent else False
    }

# WebSocket endpoint
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket_manager.connect(websocket)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": f"Connected for user {user_id}",
            "timestamp": datetime.utcnow().isoformat(),
            "agent_enabled": enhanced_processor.security_agent.available if enhanced_processor.security_agent else False
        })
        
        while True:
            data = await websocket.receive_json()
            
            if data.get('type') == 'ping':
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
            elif data.get('type') == 'get_agent_analysis':
                session_id = data.get('session_id')
                if session_id and session_id in enhanced_processor.active_sessions:
                    if enhanced_processor.security_agent and enhanced_processor.security_agent.available:
                        try:
                            analysis = await enhanced_processor.security_agent.analyze_behavioral_risk(
                                enhanced_processor.active_sessions[session_id]
                            )
                            await websocket.send_json({
                                "type": "agent_analysis",
                                "session_id": session_id,
                                "analysis": analysis
                            })
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Agent analysis failed: {str(e)}"
                            })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "AI agent not available"
                        })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {str(e)}")
    finally:
        websocket_manager.disconnect(websocket)

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: HTTPException):
    logger.error(f"Internal server error: {exc.detail}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.utcnow().isoformat()}
    )

# Run the app
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    logger.info("Starting Enhanced Behavioral Fraud Detection Backend with AI Agents...")
    
    # Check dependencies
    if not AGENTS_AVAILABLE:
        logger.warning("Agent libraries not available. Install with: pip install ollama langchain langchain-community")
    
    if not CELERY_AVAILABLE:
        logger.warning("Celery not available. Background tasks disabled. Install with: pip install redis celery")
    
    uvicorn.run(
        app,  # Direct reference to app variable
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )