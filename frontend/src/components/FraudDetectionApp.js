import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { 
  AlertTriangle, 
  TrendingUp, 
  Users, 
  Shield, 
  Cpu, 
  Zap, 
  Mouse, 
  Keyboard, 
  Eye 
} from 'lucide-react';

// Mock API functions
const mockAPI = {
  get: async (url) => {
    await new Promise(resolve => setTimeout(resolve, 500));
    
    if (url === '/api/dashboard/stats') {
      return {
        data: {
          users: { total: 125, active: 8 },
          sessions: { active: 3, total: 47 },
          processing: { 
            events_processed: 15420, 
            events_per_second: 12.4, 
            anomalies_detected: 4 
          },
          alerts: { unresolved: 2 },
          ai_agent: {
            agent_ready: true,
            model: 'llama3.2:3b',
            agent_analyses_count: 23
          }
        }
      };
    }
    
    if (url === '/api/test/sessions') {
      return {
        data: {
          active_sessions: [
            {
              session_id: 'sess_abc123def456',
              user_id: 'user_789xyz',
              keystrokes: 145,
              mouse_events: 289,
              risk_score: 0.72,
              anomaly_count: 2
            },
            {
              session_id: 'sess_def456ghi789',
              user_id: 'user_456abc',
              keystrokes: 89,
              mouse_events: 156,
              risk_score: 0.34,
              anomaly_count: 0
            }
          ]
        }
      };
    }
    
    if (url === '/health') {
      return {
        data: {
          status: 'healthy',
          database: 'connected',
          ollama: 'connected'
        }
      };
    }
    
    return { data: [] };
  },
  
  post: async (url, data) => {
    await new Promise(resolve => setTimeout(resolve, 300));
    return { data: { success: true } };
  }
};

// Inline styles for reliable rendering
const styles = {
  container: {
    minHeight: '100vh',
    backgroundColor: '#f8fafc',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
  },
  header: {
    backgroundColor: '#1e40af',
    color: 'white',
    padding: '2rem 0'
  },
  headerContent: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '0 2rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  headerTitle: {
    fontSize: '2.5rem',
    fontWeight: '300',
    textTransform: 'uppercase',
    letterSpacing: '0.2em',
    margin: 0
  },
  headerSubtitle: {
    fontSize: '1rem',
    opacity: '0.8',
    marginTop: '0.5rem'
  },
  main: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '2rem'
  },
  sectionTitle: {
    fontSize: '1.8rem',
    fontWeight: '300',
    color: '#1e40af',
    textAlign: 'center',
    textTransform: 'uppercase',
    letterSpacing: '0.15em',
    marginBottom: '2rem'
  },
  grid: {
    display: 'grid',
    gap: '2rem',
    marginBottom: '3rem'
  },
  gridCols2: {
    gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))'
  },
  gridCols4: {
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))'
  },
  card: {
    backgroundColor: 'white',
    border: '2px solid #e2e8f0',
    padding: '2rem',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    transition: 'all 0.3s ease'
  },
  cardHover: {
    borderColor: '#1e40af',
    boxShadow: '0 4px 12px rgba(30,64,175,0.1)'
  },
  metricCard: {
    textAlign: 'center',
    padding: '2rem'
  },
  metricIcon: {
    width: '3rem',
    height: '3rem',
    margin: '0 auto 1rem',
    color: '#64748b'
  },
  metricTitle: {
    fontSize: '0.75rem',
    fontWeight: 'bold',
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    color: '#64748b',
    marginBottom: '0.5rem'
  },
  metricValue: {
    fontSize: '2.5rem',
    fontWeight: '300',
    color: '#1e293b',
    lineHeight: '1'
  },
  metricSubtitle: {
    fontSize: '0.75rem',
    color: '#64748b',
    marginTop: '0.5rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em'
  },
  button: {
    backgroundColor: '#1e40af',
    color: 'white',
    border: '2px solid #1e40af',
    padding: '0.75rem 1.5rem',
    fontSize: '0.875rem',
    fontWeight: 'bold',
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    outline: 'none'
  },
  buttonSecondary: {
    backgroundColor: 'white',
    color: '#1e40af',
    border: '2px solid #1e40af'
  },
  buttonDanger: {
    backgroundColor: '#dc2626',
    borderColor: '#dc2626'
  },
  input: {
    width: '100%',
    padding: '0.75rem',
    border: '2px solid #e2e8f0',
    fontSize: '1rem',
    textAlign: 'center',
    outline: 'none',
    transition: 'border-color 0.3s ease'
  },
  statusIndicator: {
    width: '1rem',
    height: '1rem',
    borderRadius: '50%',
    display: 'inline-block',
    marginLeft: '0.5rem'
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse'
  },
  tableHeader: {
    borderBottom: '2px solid #e2e8f0',
    padding: '1rem',
    textAlign: 'left',
    fontSize: '0.875rem',
    fontWeight: 'bold',
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    color: '#64748b'
  },
  tableCell: {
    padding: '1rem',
    borderBottom: '1px solid #f1f5f9',
    fontSize: '0.875rem'
  },
  riskBadge: {
    display: 'inline-block',
    padding: '0.25rem 0.75rem',
    fontSize: '0.75rem',
    fontWeight: 'bold',
    textTransform: 'uppercase',
    borderRadius: '0.25rem',
    border: '2px solid'
  },
  riskLow: {
    backgroundColor: '#dcfce7',
    color: '#166534',
    borderColor: '#bbf7d0'
  },
  riskMedium: {
    backgroundColor: '#fef3c7',
    color: '#92400e',
    borderColor: '#fde68a'
  },
  riskHigh: {
    backgroundColor: '#fee2e2',
    color: '#991b1b',
    borderColor: '#fecaca'
  },
  footer: {
    backgroundColor: '#1e40af',
    color: 'white',
    textAlign: 'center',
    padding: '2rem'
  }
};

const MetricCard = ({ title, value, icon: Icon, subtitle, accent = false }) => (
  <div style={{
    ...styles.card,
    ...styles.metricCard,
    ...(accent ? { borderColor: '#1e40af', boxShadow: '0 4px 12px rgba(30,64,175,0.1)' } : {})
  }}>
    <Icon style={{
      ...styles.metricIcon,
      ...(accent ? { color: '#1e40af' } : {})
    }} />
    <div style={styles.metricTitle}>{title}</div>
    <div style={{
      ...styles.metricValue,
      ...(accent ? { color: '#1e40af' } : {})
    }}>{value}</div>
    {subtitle && <div style={styles.metricSubtitle}>{subtitle}</div>}
  </div>
);

const Button = ({ children, variant = 'primary', onClick, disabled, style = {} }) => {
  const getButtonStyle = () => {
    const baseStyle = { ...styles.button };
    if (variant === 'secondary') {
      return { ...baseStyle, ...styles.buttonSecondary };
    }
    if (variant === 'danger') {
      return { ...baseStyle, ...styles.buttonDanger };
    }
    return baseStyle;
  };

  return (
    <button
      style={{ ...getButtonStyle(), ...style, ...(disabled ? { opacity: 0.5, cursor: 'not-allowed' } : {}) }}
      onClick={onClick}
      disabled={disabled}
      onMouseEnter={(e) => {
        if (!disabled) {
          if (variant === 'secondary') {
            e.target.style.backgroundColor = '#1e40af';
            e.target.style.color = 'white';
          } else {
            e.target.style.backgroundColor = 'white';
            e.target.style.color = '#1e40af';
          }
        }
      }}
      onMouseLeave={(e) => {
        if (!disabled) {
          if (variant === 'secondary') {
            e.target.style.backgroundColor = 'white';
            e.target.style.color = '#1e40af';
          } else {
            e.target.style.backgroundColor = '#1e40af';
            e.target.style.color = 'white';
          }
        }
      }}
    >
      {children}
    </button>
  );
};

const Card = ({ children, title }) => (
  <div style={styles.card}>
    {title && (
      <h3 style={{
        fontSize: '1.25rem',
        fontWeight: 'bold',
        color: '#1e40af',
        textAlign: 'center',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        marginBottom: '1.5rem'
      }}>{title}</h3>
    )}
    {children}
  </div>
);

const BehaviorCapture = ({ userId, sessionId, onEventCaptured }) => {
  const canvasRef = useRef(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [eventCount, setEventCount] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !isCapturing) return;

    const handleMouseMove = async (e) => {
      setEventCount(prev => prev + 1);
      if (onEventCaptured) onEventCaptured();
    };

    const handleKeyDown = async (e) => {
      setEventCount(prev => prev + 1);
      if (onEventCaptured) onEventCaptured();
    };

    canvas.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      canvas.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isCapturing, userId, sessionId, onEventCaptured]);

  return (
    <Card title="Behavior Capture">
      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '2rem', marginBottom: '1.5rem' }}>
          <div>
            <div style={styles.metricTitle}>Events Captured</div>
            <div style={{ ...styles.metricValue, fontSize: '2rem' }}>{eventCount}</div>
          </div>
          <Button
            variant={isCapturing ? 'danger' : 'primary'}
            onClick={() => setIsCapturing(!isCapturing)}
          >
            {isCapturing ? 'Stop Capture' : 'Start Capture'}
          </Button>
        </div>
      </div>
      
      <div style={{ 
        border: '4px solid #d1d5db', 
        backgroundColor: '#f9fafb', 
        aspectRatio: '16/9',
        position: 'relative',
        ...(isCapturing ? { borderColor: '#1e40af', backgroundColor: '#eff6ff' } : {})
      }}>
        <canvas
          ref={canvasRef}
          width={800}
          height={450}
          style={{ 
            width: '100%', 
            height: '100%', 
            cursor: isCapturing ? 'crosshair' : 'default',
            display: 'block'
          }}
        />
        {!isCapturing && (
          <div style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'column'
          }}>
            <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: '#64748b', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
              Click "Start Capture" to Begin
            </div>
            <div style={{ fontSize: '0.875rem', color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Move mouse and type to generate behavioral data
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

const FraudDetectionApp = () => {
  const [stats, setStats] = useState(null);
  const [currentUser, setCurrentUser] = useState(null);
  const [currentSession, setCurrentSession] = useState(null);
  const [newUserData, setNewUserData] = useState({ username: '', email: '' });
  const [isCreateUserOpen, setIsCreateUserOpen] = useState(false);
  const [activeSessions, setActiveSessions] = useState([]);
  const [healthStatus, setHealthStatus] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      const [statsResponse, sessionsResponse, healthResponse] = await Promise.all([
        mockAPI.get('/api/dashboard/stats'),
        mockAPI.get('/api/test/sessions'),
        mockAPI.get('/health')
      ]);
      
      setStats(statsResponse.data);
      setActiveSessions(sessionsResponse.data.active_sessions);
      setHealthStatus(healthResponse.data);
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const createUser = async () => {
    if (!newUserData.username || !newUserData.email) {
      alert('Please fill in all fields');
      return;
    }

    const response = await mockAPI.post('/api/users', newUserData);
    setCurrentUser({ ...newUserData, id: 'user_' + Math.random().toString(36).substr(2, 9) });
    setNewUserData({ username: '', email: '' });
    setIsCreateUserOpen(false);
  };

  const createSession = async () => {
    if (!currentUser) {
      alert('Please create a user first');
      return;
    }

    const sessionId = 'sess_' + Math.random().toString(36).substr(2, 9);
    setCurrentSession({ id: sessionId, risk_score: 0.15 });
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
            <Shield size={64} />
            <div>
              <h1 style={styles.headerTitle}>Behavioral Fraud Detection</h1>
              <p style={styles.headerSubtitle}>Real-time Security Monitoring System</p>
            </div>
          </div>
          
          {healthStatus && (
            <div style={{ textAlign: 'right' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '0.75rem', marginBottom: '0.5rem' }}>
                <div style={{
                  ...styles.statusIndicator,
                  backgroundColor: healthStatus.status === 'healthy' ? '#22c55e' : '#ef4444'
                }}></div>
                <span style={{ fontSize: '1rem', fontWeight: '500', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                  {healthStatus.ollama === 'connected' ? 'AI Agent Ready' : 'AI Agent Offline'}
                </span>
              </div>
              <p style={{ fontSize: '0.875rem', opacity: '0.8' }}>System Status: Active</p>
            </div>
          )}
        </div>
      </header>

      <div style={styles.main}>
        {/* System Overview */}
        {stats && (
          <section>
            <h2 style={styles.sectionTitle}>System Overview</h2>
            <div style={{ ...styles.grid, ...styles.gridCols4 }}>
              <MetricCard
                title="Total Users"
                value={stats.users.total}
                icon={Users}
                subtitle={`${stats.users.active} Active Users`}
              />
              <MetricCard
                title="Active Sessions"
                value={stats.sessions.active}
                icon={TrendingUp}
                subtitle={`${stats.sessions.total} Total Sessions`}
                accent={stats.sessions.active > 0}
              />
              <MetricCard
                title="Events Processed"
                value={stats.processing.events_processed.toLocaleString()}
                icon={Zap}
                subtitle={`${stats.processing.events_per_second.toFixed(1)} Events/sec`}
              />
              <MetricCard
                title="Anomalies Detected"
                value={stats.processing.anomalies_detected}
                icon={AlertTriangle}
                subtitle={`${stats.alerts.unresolved} Unresolved`}
                accent={stats.processing.anomalies_detected > 0}
              />
            </div>
          </section>
        )}

        {/* AI Agent Status */}
        {stats && stats.ai_agent && (
          <section>
            <div style={{
              ...styles.card,
              borderColor: '#1e40af',
              borderWidth: '4px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '3rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
                <Cpu size={80} color="#1e40af" />
                <div>
                  <h3 style={{ fontSize: '2rem', fontWeight: '300', color: '#1e40af', textTransform: 'uppercase', letterSpacing: '0.15em', marginBottom: '1rem' }}>
                    AI Security Agent
                  </h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '3rem', fontSize: '1rem' }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={styles.metricTitle}>Status</div>
                      <div style={{ fontWeight: 'bold', color: '#1e40af', fontSize: '1.25rem' }}>
                        {stats.ai_agent.agent_ready ? 'Ready' : 'Offline'}
                      </div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={styles.metricTitle}>Model</div>
                      <div style={{ fontWeight: 'bold', color: '#1e40af', fontSize: '1.25rem', fontFamily: 'monospace' }}>
                        {stats.ai_agent.model}
                      </div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={styles.metricTitle}>Analyses</div>
                      <div style={{ fontWeight: 'bold', color: '#1e40af', fontSize: '1.25rem' }}>
                        {stats.ai_agent.agent_analyses_count}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div style={{
                ...styles.statusIndicator,
                width: '3rem',
                height: '3rem',
                border: '4px solid',
                backgroundColor: stats.ai_agent.agent_ready ? '#22c55e' : '#ef4444',
                borderColor: stats.ai_agent.agent_ready ? '#16a34a' : '#dc2626'
              }}></div>
            </div>
          </section>
        )}

        {/* User and Session Management */}
        <section>
          <h2 style={styles.sectionTitle}>Session Management</h2>
          <div style={{ ...styles.grid, ...styles.gridCols2 }}>
            <Card title="User Management">
              {!currentUser ? (
                <div style={{ textAlign: 'center' }}>
                  <Button
                    onClick={() => setIsCreateUserOpen(true)}
                    style={{ marginBottom: '2rem' }}
                  >
                    Create New User
                  </Button>
                  
                  {isCreateUserOpen && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                      <input
                        type="text"
                        placeholder="Username"
                        value={newUserData.username}
                        onChange={(e) => setNewUserData({...newUserData, username: e.target.value})}
                        style={styles.input}
                      />
                      <input
                        type="email"
                        placeholder="Email Address"
                        value={newUserData.email}
                        onChange={(e) => setNewUserData({...newUserData, email: e.target.value})}
                        style={styles.input}
                      />
                      <div style={{ display: 'flex', gap: '1rem' }}>
                        <Button onClick={createUser} style={{ flex: 1 }}>
                          Create User
                        </Button>
                        <Button variant="secondary" onClick={() => setIsCreateUserOpen(false)} style={{ flex: 1 }}>
                          Cancel
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ textAlign: 'center' }}>
                  <div style={{ backgroundColor: '#dcfce7', border: '2px solid #bbf7d0', padding: '1.5rem', marginBottom: '2rem' }}>
                    <div style={{ fontWeight: 'bold', color: '#166534', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.5rem' }}>
                      Active User
                    </div>
                    <div style={{ color: '#15803d', fontSize: '1.5rem', fontWeight: '500' }}>{currentUser.username}</div>
                    <div style={{ color: '#16a34a', fontSize: '0.875rem', fontFamily: 'monospace', marginTop: '0.25rem' }}>
                      {currentUser.email}
                    </div>
                  </div>
                  <Button variant="secondary" onClick={() => setCurrentUser(null)}>
                    Switch User
                  </Button>
                </div>
              )}
            </Card>

            <Card title="Session Control">
              {!currentSession ? (
                <div style={{ textAlign: 'center' }}>
                  <Button
                    onClick={createSession}
                    disabled={!currentUser}
                    style={{ marginBottom: '1.5rem' }}
                  >
                    Create Session
                  </Button>
                  {!currentUser && (
                    <p style={{ color: '#64748b', fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                      Create user first
                    </p>
                  )}
                </div>
              ) : (
                <div style={{ textAlign: 'center' }}>
                  <div style={{ backgroundColor: '#dbeafe', border: '2px solid #bfdbfe', padding: '1.5rem', marginBottom: '2rem' }}>
                    <div style={{ fontWeight: 'bold', color: '#1e40af', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.5rem' }}>
                      Active Session
                    </div>
                    <div style={{ color: '#1e40af', fontFamily: 'monospace', fontSize: '1rem' }}>{currentSession.id}</div>
                    <div style={{ color: '#3b82f6', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                      Risk Score: {currentSession.risk_score}
                    </div>
                  </div>
                  <Button variant="danger" onClick={() => setCurrentSession(null)}>
                    End Session
                  </Button>
                </div>
              )}
            </Card>
          </div>
        </section>

        {/* Behavior Tools */}
        {currentUser && currentSession && (
          <section>
            <h2 style={styles.sectionTitle}>Behavior Analysis Tools</h2>
            <div style={{ ...styles.grid, ...styles.gridCols2 }}>
              <BehaviorCapture
                userId={currentUser.id}
                sessionId={currentSession.id}
              />
              <Card title="Test Generator">
                <div style={{ textAlign: 'center' }}>
                  <div style={styles.metricTitle}>Generate synthetic behavioral data for testing</div>
                  <Button style={{ marginTop: '2rem' }}>
                    Generate Test Data
                  </Button>
                </div>
              </Card>
            </div>
          </section>
        )}

        {/* Active Sessions Table */}
        <section>
          <h2 style={styles.sectionTitle}>Active Sessions Monitor</h2>
          <Card>
            {activeSessions.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '4rem 0' }}>
                <div style={{ fontSize: '0.875rem', color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Create a session to begin monitoring
                </div>
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={styles.table}>
                  <thead>
                    <tr>
                      <th style={styles.tableHeader}>Session</th>
                      <th style={styles.tableHeader}>User</th>
                      <th style={{ ...styles.tableHeader, textAlign: 'center' }}>Events</th>
                      <th style={{ ...styles.tableHeader, textAlign: 'center' }}>Risk Score</th>
                      <th style={{ ...styles.tableHeader, textAlign: 'center' }}>Anomalies</th>
                      <th style={{ ...styles.tableHeader, textAlign: 'center' }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeSessions.map((session) => {
                      const getRiskStyle = (score) => {
                        if (score > 0.7) return { ...styles.riskBadge, ...styles.riskHigh };
                        if (score > 0.4) return { ...styles.riskBadge, ...styles.riskMedium };
                        return { ...styles.riskBadge, ...styles.riskLow };
                      };

                      return (
                        <tr key={session.session_id} style={{ 
                          borderBottom: '1px solid #f1f5f9',
                          transition: 'background-color 0.3s ease'
                        }}>
                          <td style={{ ...styles.tableCell, fontFamily: 'monospace', fontSize: '1rem' }}>
                            {session.session_id.slice(-8)}
                          </td>
                          <td style={{ ...styles.tableCell, fontFamily: 'monospace', fontSize: '1rem' }}>
                            {session.user_id.slice(-8)}
                          </td>
                          <td style={{ ...styles.tableCell, textAlign: 'center' }}>
                            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '1.5rem' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <Keyboard size={20} color="#64748b" />
                                <span style={{ fontSize: '1rem', fontWeight: '500' }}>{session.keystrokes}</span>
                              </div>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <Mouse size={20} color="#64748b" />
                                <span style={{ fontSize: '1rem', fontWeight: '500' }}>{session.mouse_events}</span>
                              </div>
                            </div>
                          </td>
                          <td style={{ ...styles.tableCell, textAlign: 'center' }}>
                            <span style={getRiskStyle(session.risk_score)}>
                              {session.risk_score.toFixed(2)}
                            </span>
                          </td>
                          <td style={{ ...styles.tableCell, textAlign: 'center' }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                              {session.anomaly_count > 0 && (
                                <AlertTriangle size={20} color="#ef4444" />
                              )}
                              <span style={{ fontSize: '1rem', fontWeight: '500' }}>{session.anomaly_count}</span>
                            </div>
                          </td>
                          <td style={{ ...styles.tableCell, textAlign: 'center' }}>
                            <Button
                              variant="secondary"
                              onClick={() => alert(`Monitoring session ${session.session_id.slice(-8)}`)}
                              style={{ padding: '0.5rem' }}
                            >
                              <Eye size={16} />
                            </Button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </section>

        {/* Analytics Charts */}
        <section>
          <h2 style={styles.sectionTitle}>Real-time Analytics</h2>
          <div style={{ ...styles.grid, ...styles.gridCols2 }}>
            <Card title="Risk Score Trends">
              <div style={{ height: '320px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={activeSessions.map((session, index) => ({
                    session: session.session_id.slice(-8),
                    risk: session.risk_score,
                    index
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis 
                      dataKey="session" 
                      axisLine={false}
                      tickLine={false}
                      tick={{ fontSize: 12, fill: '#6b7280' }}
                    />
                    <YAxis 
                      domain={[0, 1]} 
                      axisLine={false}
                      tickLine={false}
                      tick={{ fontSize: 12, fill: '#6b7280' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1e40af', 
                        border: 'none', 
                        color: 'white',
                        fontWeight: 'bold',
                        borderRadius: '4px'
                      }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="risk" 
                      stroke="#1e40af" 
                      strokeWidth={3}
                      dot={{ fill: '#1e40af', strokeWidth: 2, r: 6 }}
                      activeDot={{ r: 8, stroke: '#1e40af', strokeWidth: 2 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <Card title="Event Distribution">
              <div style={{ height: '320px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={activeSessions.map((session, index) => ({
                    session: session.session_id.slice(-8),
                    keystrokes: session.keystrokes,
                    mouse: session.mouse_events
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis 
                      dataKey="session" 
                      axisLine={false}
                      tickLine={false}
                      tick={{ fontSize: 12, fill: '#6b7280' }}
                    />
                    <YAxis 
                      axisLine={false}
                      tickLine={false}
                      tick={{ fontSize: 12, fill: '#6b7280' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1e40af', 
                        border: 'none', 
                        color: 'white',
                        fontWeight: 'bold',
                        borderRadius: '4px'
                      }}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="keystrokes" 
                      stackId="1" 
                      stroke="#10b981" 
                      fill="#10b981" 
                      fillOpacity={0.7}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="mouse" 
                      stackId="1" 
                      stroke="#1e40af" 
                      fill="#1e40af"
                      fillOpacity={0.7}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>
        </section>

        {/* System Status */}
        <section>
          <h2 style={styles.sectionTitle}>System Status</h2>
          <Card title="Infrastructure Health">
            {healthStatus && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '3rem', marginBottom: '3rem' }}>
                <div style={{ textAlign: 'center', padding: '1.5rem', border: '2px solid #e5e7eb' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                    <span style={{ fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.1em', color: '#374151' }}>
                      Backend Status
                    </span>
                    <div style={{
                      ...styles.statusIndicator,
                      backgroundColor: healthStatus.status === 'healthy' ? '#22c55e' : '#ef4444'
                    }}></div>
                  </div>
                  <p style={{ fontSize: '2rem', fontWeight: '300', color: '#1e40af', textTransform: 'uppercase' }}>
                    {healthStatus.status}
                  </p>
                </div>
                
                <div style={{ textAlign: 'center', padding: '1.5rem', border: '2px solid #e5e7eb' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                    <span style={{ fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.1em', color: '#374151' }}>
                      Database
                    </span>
                    <div style={{
                      ...styles.statusIndicator,
                      backgroundColor: healthStatus.database === 'connected' ? '#22c55e' : '#ef4444'
                    }}></div>
                  </div>
                  <p style={{ fontSize: '2rem', fontWeight: '300', color: '#1e40af', textTransform: 'uppercase' }}>
                    {healthStatus.database}
                  </p>
                </div>
                
                <div style={{ textAlign: 'center', padding: '1.5rem', border: '2px solid #e5e7eb' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                    <span style={{ fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.1em', color: '#374151' }}>
                      AI Agent
                    </span>
                    <div style={{
                      ...styles.statusIndicator,
                      backgroundColor: healthStatus.ollama === 'connected' ? '#22c55e' : '#ef4444'
                    }}></div>
                  </div>
                  <p style={{ fontSize: '2rem', fontWeight: '300', color: '#1e40af', textTransform: 'uppercase' }}>
                    {healthStatus.ollama}
                  </p>
                </div>
              </div>
            )}
            
            <div style={{ backgroundColor: '#f8fafc', border: '2px solid #e5e7eb', padding: '2rem' }}>
              <h4 style={{ fontWeight: 'bold', color: '#1e40af', marginBottom: '1.5rem', textTransform: 'uppercase', letterSpacing: '0.1em', textAlign: 'center' }}>
                Quick Start Guide
              </h4>
              <ol style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <li style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <span style={{ backgroundColor: '#1e40af', color: 'white', width: '2rem', height: '2rem', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '0.875rem' }}>1</span>
                  <span style={{ color: '#374151', fontWeight: '500', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Create a user account</span>
                </li>
                <li style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <span style={{ backgroundColor: '#1e40af', color: 'white', width: '2rem', height: '2rem', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '0.875rem' }}>2</span>
                  <span style={{ color: '#374151', fontWeight: '500', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Start a new session</span>
                </li>
                <li style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <span style={{ backgroundColor: '#1e40af', color: 'white', width: '2rem', height: '2rem', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '0.875rem' }}>3</span>
                  <span style={{ color: '#374151', fontWeight: '500', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Use behavior capture or test generator</span>
                </li>
                <li style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <span style={{ backgroundColor: '#1e40af', color: 'white', width: '2rem', height: '2rem', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '0.875rem' }}>4</span>
                  <span style={{ color: '#374151', fontWeight: '500', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Monitor real-time anomaly detection</span>
                </li>
                <li style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <span style={{ backgroundColor: '#1e40af', color: 'white', width: '2rem', height: '2rem', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '0.875rem' }}>5</span>
                  <span style={{ color: '#374151', fontWeight: '500', textTransform: 'uppercase', letterSpacing: '0.05em' }}>View AI agent analysis results</span>
                </li>
              </ol>
            </div>
          </Card>
        </section>
      </div>

      {/* Footer */}
      <footer style={styles.footer}>
      </footer>
    </div>
  );
};

export default FraudDetectionApp; 