# app.py - Clean FastAPI Behavioral Fraud Detection Backend
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
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.pool import StaticPool

# Redis and Celery
import redis.asyncio as aioredis
from celery import Celery

# ML libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
import joblib
from scipy import stats

# Configuration
class Settings:
    app_name = "Behavioral Fraud Detection API"
    version = "1.0.0"
    debug = True
    
    database_url = os.getenv("DATABASE_URL", "sqlite:///./fraud_detection.db")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    celery_broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    
    anomaly_threshold = 0.6
    min_training_samples = 50
    max_session_duration = 3600

settings = Settings()

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

# Feature Engineering
class BehavioralFeatureExtractor:
    def __init__(self):
        self.scaler = RobustScaler()
    
    def extract_keystroke_features(self, keystroke_data: List[KeystrokeDynamics]) -> Dict[str, float]:
        """Extract keystroke dynamics features"""
        if not keystroke_data or len(keystroke_data) < 2:
            return {
                'dwell_mean': 0.0, 'dwell_std': 0.0, 'flight_mean': 0.0, 'flight_std': 0.0,
                'typing_speed': 0.0, 'rhythm_consistency': 0.0, 'keystroke_entropy': 0.0
            }
        
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
        if not self.models_trained:
            return {
                'anomaly_score': 0.0,
                'is_anomaly': False,
                'confidence': 0.0,
                'risk_level': 'low'
            }
        
        try:
            # Extract features
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
                    'risk_level': 'low'
                }
            
            # Prepare for prediction
            feature_vector = np.array(list(combined_features.values())).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_vector)
            
            # Get predictions
            isolation_score = self.isolation_forest.decision_function(scaled_features)[0]
            isolation_anomaly = self.isolation_forest.predict(scaled_features)[0] == -1
            
            svm_score = 0.0
            svm_anomaly = False
            if self.one_class_svm:
                svm_score = self.one_class_svm.decision_function(scaled_features)[0]
                svm_anomaly = self.one_class_svm.predict(scaled_features)[0] == -1
            
            # Baseline comparison
            baseline_score = self._compare_with_baseline(user_id, feature_vector[0])
            
            # Combine scores
            final_score = (abs(isolation_score) * 0.4 + abs(svm_score) * 0.3 + baseline_score * 0.3)
            is_anomaly = isolation_anomaly or svm_anomaly or baseline_score > 0.7
            confidence = min(abs(final_score), 1.0)
            
            # Risk level
            if not is_anomaly and final_score < 0.3:
                risk_level = 'low'
            elif final_score < 0.6:
                risk_level = 'medium'
            elif final_score < 0.8:
                risk_level = 'high'
            else:
                risk_level = 'critical'
            
            # Update history
            self.anomaly_history[user_id].append(final_score)
            if len(self.anomaly_history[user_id]) > 100:
                self.anomaly_history[user_id].popleft()
            
            return {
                'anomaly_score': float(final_score),
                'is_anomaly': bool(is_anomaly),
                'confidence': float(confidence),
                'risk_level': risk_level
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return {
                'anomaly_score': 0.0,
                'is_anomaly': False,
                'confidence': 0.0,
                'risk_level': 'low'
            }
    
    def _compare_with_baseline(self, user_id: str, features: np.ndarray) -> float:
        """Compare with user baseline"""
        if user_id not in self.baseline_profiles:
            return 0.0
        
        baseline = self.baseline_profiles[user_id]
        z_scores = []
        
        for i, value in enumerate(features):
            if i < len(baseline['std']) and baseline['std'][i] > 0:
                z_score = abs(value - baseline['mean'][i]) / baseline['std'][i]
                z_scores.append(z_score)
        
        if not z_scores:
            return 0.0
        
        anomalous_features = sum(1 for z in z_scores if z > 2.0)
        return anomalous_features / len(z_scores)

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
                    'confidence': detection_result['confidence'],
                    'risk_level': detection_result['risk_level']
                }
                
            except Exception as e:
                logger.error(f"Anomaly detection failed: {str(e)}")
                return {'anomaly_detected': False}
        
        return {'anomaly_detected': False, 'reason': 'insufficient_data'}
    
    async def _store_anomaly_alert(self, session_id: str, detection_result: Dict, db: Session):
        """Store anomaly alert"""
        try:
            alert = AnomalyAlert(
                session_id=session_id,
                anomaly_type='behavioral',
                confidence_score=detection_result['confidence'],
                severity=detection_result['risk_level'],
                details=json.dumps(detection_result)
            )
            
            db.add(alert)
            db.commit()
            
            # Broadcast via WebSocket
            await websocket_manager.broadcast_alert({
                'session_id': session_id,
                'risk_level': detection_result['risk_level'],
                'confidence': detection_result['confidence'],
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Alert storage failed: {str(e)}")
            db.rollback()

# WebSocket Manager
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Connect WebSocket"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect WebSocket"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast_alert(self, alert_data: Dict):
        """Broadcast alert to all connections"""
        if not self.active_connections:
            return
        
        message = {'type': 'anomaly_alert', 'data': alert_data}
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

# Initialize global instances
real_time_processor = RealTimeProcessor()
websocket_manager = WebSocketManager()

# Celery setup
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
        
        # Get training data (simplified for hackathon)
        sessions = db.query(BehavioralSession).filter(BehavioralSession.user_id == user_id).all()
        
        if sessions:
            # Build training data from active sessions
            training_data = {user_id: {}}
            for session in sessions:
                if session.id in real_time_processor.active_sessions:
                    training_data[user_id][session.id] = real_time_processor.active_sessions[session.id]
            
            # Train models
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(real_time_processor.anomaly_engine.train_models(training_data))
            
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
    logger.info("Starting Behavioral Fraud Detection Backend...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Real-time behavioral fraud detection using ML",
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

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Behavioral Fraud Detection API",
        "version": settings.version,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check"""
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected",
            "version": settings.version
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/api/users", response_model=UserResponse)
async def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Create a new user"""
    try:
        # Check if exists
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
        raise HTTPException(status_code=500, detail="User creation failed")

@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(
    session_data: SessionCreate, 
    request: Request,
    db: Session = Depends(get_db)
):
    """Create a new session"""
    try:
        user = db.query(User).filter(User.id == session_data.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get client info
        client_ip = request.client.host if hasattr(request, 'client') else "unknown"
        
        session = BehavioralSession(
            user_id=session_data.user_id,
            ip_address=client_ip,
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
        raise HTTPException(status_code=500, detail="Session creation failed")

@app.post("/api/events", response_model=EventResponse)
async def process_event(
    event_data: BehavioralEvent,
    db: Session = Depends(get_db)
):
    """Process behavioral events"""
    try:
        event_dict = event_data.dict()
        result = await real_time_processor.process_behavioral_event(event_dict, db)
        return EventResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Event processing failed")

@app.get("/api/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, db: Session = Depends(get_db)):
    """Get user information"""
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

@app.post("/api/profiles/{user_id}/build")
async def build_user_profile(user_id: str, db: Session = Depends(get_db)):
    """Trigger profile building"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    task = train_user_models.delay(user_id)
    
    return {
        "message": "Profile building initiated",
        "task_id": task.id,
        "user_id": user_id
    }

@app.get("/api/sessions/{session_id}/alerts")
async def get_session_alerts(session_id: str, db: Session = Depends(get_db)):
    """Get alerts for session"""
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
            "resolved": alert.resolved,
            "created_at": alert.created_at.isoformat()
        } for alert in alerts
    ]

@app.get("/api/dashboard/stats")
async def dashboard_stats(db: Session = Depends(get_db)):
    """Get dashboard statistics"""
    try:
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.is_active == True).count()
        total_sessions = db.query(BehavioralSession).count()
        total_alerts = db.query(AnomalyAlert).count()
        
        processor_stats = real_time_processor.stats.copy()
        uptime = datetime.utcnow() - processor_stats['last_reset']
        
        return {
            "users": {
                "total": total_users,
                "active": active_users
            },
            "sessions": {
                "total": total_sessions,
                "active": len(real_time_processor.active_sessions)
            },
            "alerts": {
                "total": total_alerts
            },
            "processing": {
                **processor_stats,
                "uptime_seconds": uptime.total_seconds(),
                "events_per_second": processor_stats['events_processed'] / max(uptime.total_seconds(), 1)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Stats retrieval failed")

# WebSocket endpoint
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket for real-time updates"""
    await websocket_manager.connect(websocket)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": f"Connected for user {user_id}",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        while True:
            data = await websocket.receive_json()
            
            if data.get('type') == 'ping':
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        pass
    finally:
        websocket_manager.disconnect(websocket)

# Testing endpoint
@app.post("/api/test/generate-events")
async def generate_test_events(
    user_id: str,
    session_id: str,
    event_count: int = 50,
    anomalous: bool = False,
    db: Session = Depends(get_db)
):
    """Generate test events for demo"""
    try:
        session = db.query(BehavioralSession).filter(BehavioralSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        events_created = 0
        current_time = time.time()
        
        # Generate keystroke events
        for i in range(event_count // 2):
            if anomalous:
                dwell_time = np.random.uniform(5, 15)  # Very short (bot-like)
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
            
            await process_event(event_data, db)
            events_created += 1
        
        # Generate mouse events
        x, y = 500, 500
        for i in range(event_count // 2):
            if anomalous:
                dx, dy = np.random.randint(-200, 200, 2)  # Erratic
                velocity = np.random.uniform(800, 1500)
            else:
                dx, dy = np.random.randint(-30, 30, 2)  # Normal
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
                timestamp=current_time + (event_count // 2) * 0.1 + i * 0.05
            )
            
            await process_event(event_data, db)
            events_created += 1
        
        return {
            "message": f"Generated {events_created} test events",
            "anomalous": anomalous,
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Test generation failed")

# Run the app
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )