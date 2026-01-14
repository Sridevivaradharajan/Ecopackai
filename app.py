import joblib 
import traceback 
import numpy as np 
import pandas as pd 
import sys
import json 
from werkzeug.utils import secure_filename 
from flask import Flask, request, jsonify, render_template, redirect, url_for 
from functools import wraps 
from flask_cors import CORS 
from authlib.integrations.flask_client import OAuth 
import psycopg2 
from psycopg2.extras import RealDictCursor 
from pathlib import Path
import bcrypt 
import jwt 
import datetime 
import os 
import re 
from dotenv import load_dotenv 
 
# -------------------------------------------------- 
# LOAD ENV 
# -------------------------------------------------- 
load_dotenv() 
 
# -------------------------------------------------- 
# APP INIT (MUST BE FIRST) 
# -------------------------------------------------- 
app = Flask(__name__) 
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key") 
 
# -------------------------------------------------- 
# CONFIG 
# -------------------------------------------------- 
app.config['JWT_SECRET'] = os.getenv("JWT_SECRET", "jwt-secret") 
app.config['UPLOAD_FOLDER'] = 'uploads' 
 
# -------------------------------------------------- 
# CORS 
# -------------------------------------------------- 
CORS(
    app,
    supports_credentials=True,
    origins=[
        "http://localhost:3000",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "https://ecopack-yngt.onrender.com",
    ],
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    expose_headers=["Set-Cookie"]
)
# -------------------------------------------------- 
# DATABASE 
# -------------------------------------------------- 
DATABASE_URL = os.getenv("DATABASE_URL") 
if not DATABASE_URL: 
    raise ValueError("DATABASE_URL is not set") 
 
# -------------------------------------------------- 
# OAUTH (AFTER app is created) 
# -------------------------------------------------- 
 
oauth = OAuth(app) 
 
oauth.register( 
    name='google', 
    client_id=os.getenv('GOOGLE_CLIENT_ID'), 
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'), 
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration', 
    client_kwargs={ 
        'scope': 'openid email profile' 
    } 
) 
 
# -------------------------------------------------- 
# GOOGLE LOGIN 
# -------------------------------------------------- 
# Replace your existing Google OAuth routes with these fixed versions

@app.route('/auth/google')
def google_login():
    # Force HTTPS for production (Render)
    redirect_uri = url_for('google_callback', _external=True, _scheme='https')
    print(f"[Google Login] Redirect URI: {redirect_uri}")
    return oauth.google.authorize_redirect(redirect_uri)

@app.route('/auth/google/callback')
def google_callback():
    print("\n[Google Callback] Starting...")
    
    try:
        token = oauth.google.authorize_access_token()
        print(f"[Google Callback] Token received: {bool(token)}")
    except Exception as e:
        print(f"[Google Callback] Error during authorization: {e}")
        return redirect('/?error=google_auth_failed')
    
    user_info = token.get('userinfo')
    if not user_info:
        try:
            user_info = oauth.google.get('userinfo').json()
        except Exception as e:
            print(f"[Google Callback] Error fetching userinfo: {e}")
            return redirect('/?error=google_auth_failed')
    
    email = user_info.get("email")
    if not email:
        print("[Google Callback] No email in user_info")
        return redirect('/?error=no_email')
    
    print(f"[Google Callback] Email: {email}")
    
    # Check if user exists or create new one
    conn = get_db_connection()
    if not conn:
        print("[Google Callback] Database connection failed")
        return redirect('/?error=database_connection_failed')
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Check if user exists
        cur.execute('SELECT id FROM users WHERE email = %s', (email,))
        user = cur.fetchone()
        
        if user:
            user_id = user['id']
            print(f"[Google Callback] Existing user found: {user_id}")
            # Update last login
            cur.execute('UPDATE users SET last_login = %s WHERE id = %s', 
                       (datetime.datetime.now(), user_id))
        else:
            # Register new user (Google login)
            full_name = user_info.get('name', email.split('@')[0])
            dummy_hash = bcrypt.hashpw(os.urandom(16), bcrypt.gensalt()).decode('utf-8')
            
            print(f"[Google Callback] Creating new user: {email}")
            cur.execute(
                'INSERT INTO users (email, password_hash, full_name) VALUES (%s, %s, %s) RETURNING id',
                (email, dummy_hash, full_name)
            )
            user_id = cur.fetchone()['id']
            cur.execute('INSERT INTO user_analytics (user_id) VALUES (%s)', (user_id,))
            print(f"[Google Callback] New user created: {user_id}")
        
        conn.commit()
    except Exception as e:
        print(f"[Google Callback] Database error: {e}")
        conn.rollback()
        return redirect('/?error=database_error')
    finally:
        cur.close()
        conn.close()
    
    # Create JWT token
    payload = {
        "user_id": user_id,
        "email": email,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }
    
    jwt_token = jwt.encode(
        payload,
        app.config['JWT_SECRET'],
        algorithm="HS256"
    )
    
    print(f"[Google Callback] JWT created for user {user_id}")
    
    # Redirect to dashboard with cookie
    dashboard_url = '/dashboard'
    response = redirect(dashboard_url)
    
    # FIXED: Use Lax for same-site redirect (not cross-origin)
    # samesite='None' is only needed for cross-origin requests
    # Since we're redirecting to our own domain, use 'Lax'
    response.set_cookie(
        'token',
        jwt_token,
        max_age=86400,  # 24 hours
        httponly=True,
        samesite='Lax',  # Changed from 'None' - we're on same domain
        secure=True,  # Required for Render (HTTPS)
        path='/',
        domain=None  # Let browser set it automatically
    )
    
    print(f"[Google Callback] Cookie set with secure=True, samesite=Lax")
    print(f"[Google Callback] JWT Token (first 20 chars): {jwt_token[:20]}...")
    print(f"[Google Callback] Redirecting to: {dashboard_url}")
    print(f"[Google Callback] Host: {request.host}")
    print(f"[Google Callback] User ID: {user_id}")
    
    return response
class PredictionValidator:
    """Professional validation system for packaging predictions"""
    
    MATERIAL_COST_RANGES = {
        'glass': {'min': 0.5, 'max': 4.0, 'typical_max': 2.5},
        'metal': {'min': 0.8, 'max': 5.0, 'typical_max': 3.5},
        'aluminium': {'min': 1.0, 'max': 6.0, 'typical_max': 4.0},
        'aluminum': {'min': 1.0, 'max': 6.0, 'typical_max': 4.0},
        'plastic': {'min': 0.1, 'max': 2.0, 'typical_max': 1.2},
        'cardboard': {'min': 0.05, 'max': 1.5, 'typical_max': 0.8},
        'paper': {'min': 0.05, 'max': 1.5, 'typical_max': 0.8},
    }
    
    MATERIAL_CO2_RANGES = {
        'glass': {'min': 0.1, 'max': 2.0, 'typical_max': 1.2},
        'metal': {'min': 0.3, 'max': 3.0, 'typical_max': 2.0},
        'aluminium': {'min': 0.5, 'max': 4.0, 'typical_max': 2.5},
        'aluminum': {'min': 0.5, 'max': 4.0, 'typical_max': 2.5},
        'plastic': {'min': 0.05, 'max': 1.5, 'typical_max': 0.8},
        'cardboard': {'min': 0.01, 'max': 0.5, 'typical_max': 0.3},
        'paper': {'min': 0.01, 'max': 0.5, 'typical_max': 0.3},
    }
    
    @classmethod
    def validate_prediction(cls, cost_pred, co2_pred, product_dict, verbose=True):
        """Validate predictions against material-specific bounds"""
        material = str(product_dict.get('material', 'plastic')).lower()
        weight = float(product_dict.get('weight_measured', 50))
        
        # Get material ranges
        cost_ranges = cls.MATERIAL_COST_RANGES.get(material, cls.MATERIAL_COST_RANGES['plastic'])
        co2_ranges = cls.MATERIAL_CO2_RANGES.get(material, cls.MATERIAL_CO2_RANGES['plastic'])
        
        # Calculate bounds
        expected_cost_max = cost_ranges['typical_max'] * weight
        expected_co2_max = co2_ranges['typical_max'] * weight
        
        # Validation flags
        cost_status = 'valid'
        co2_status = 'valid'
        severity = 'none'
        
        if cost_pred > cost_ranges['max'] * weight:
            cost_status = 'high'
            severity = 'warning'
        elif cost_pred > expected_cost_max:
            cost_status = 'elevated'
            severity = 'info'
            
        if co2_pred > co2_ranges['max'] * weight:
            co2_status = 'high'
            severity = 'warning' if severity != 'warning' else severity
        elif co2_pred > expected_co2_max:
            co2_status = 'elevated'
            severity = 'info' if severity == 'none' else severity
        
        if verbose:
            if severity == 'none':
                print(f"[✓] Prediction: Cost=₹{cost_pred:.2f} | CO2={co2_pred:.2f}")
            elif severity == 'info':
                print(f"[ℹ] Prediction: Cost=₹{cost_pred:.2f} | CO2={co2_pred:.2f}")
                print(f"    Material: {material} ({weight}g) - Above typical range but within bounds")
            else:
                print(f"[⚠] Prediction: Cost=₹{cost_pred:.2f} | CO2={co2_pred:.2f}")
                print(f"    Material: {material} ({weight}g) - Unusually high for this material")
        
        return True, {'cost_status': cost_status, 'co2_status': co2_status, 'severity': severity}
     
class MLModelManager:
    """
    PRODUCTION ML MODEL MANAGER
    Matches training pipeline EXACTLY from notebook
    """
    
    def __init__(self):
        self.cost_model = None
        self.co2_model = None
        self.label_encoders = None
        self.cost_features = None
        self.co2_features = None
        
        # Material to parent_material mapping (CRITICAL)
        self.parent_material_map = {
            # Plastics
            'plastic': 'plastic',
            'pet': 'plastic',
            'hdpe': 'plastic',
            'ldpe': 'plastic',
            'pp': 'plastic',
            'ps': 'plastic',
            'pvc': 'plastic',
            'bioplastic': 'plastic',
            
            # Metals
            'metal': 'metal',
            'aluminium': 'metal',
            'aluminum': 'metal',
            'tinplate': 'metal',
            'steel': 'metal',
            
            # Paper-based
            'paper': 'paper-or-cardboard',
            'cardboard': 'paper-or-cardboard',
            'paperboard': 'paper-or-cardboard',
            
            # Glass
            'glass': 'glass',
            
            # Composites
            'composite': 'plastic',  # Most composites are plastic-based
            'wood': 'unknown',
            
            # Fallback
            'unknown': 'unknown'
        }
    
    def _infer_parent_material(self, material):
        """
        Infer parent_material from material using training logic
        EXACT COPY from notebook preprocessing
        """
        material_lower = str(material).lower().strip()
        
        # Direct lookup
        if material_lower in self.parent_material_map:
            return self.parent_material_map[material_lower]
        
        # Pattern matching (for variants)
        if 'plastic' in material_lower or 'pe' in material_lower or 'pp' in material_lower:
            return 'plastic'
        elif 'metal' in material_lower or 'alumin' in material_lower or 'steel' in material_lower:
            return 'metal'
        elif 'paper' in material_lower or 'cardboard' in material_lower:
            return 'paper-or-cardboard'
        elif 'glass' in material_lower:
            return 'glass'
        else:
            return 'unknown'
    
    def _standardize_recycling(self, value):
        """EXACT standardization from training notebook"""
        if not value or pd.isna(value):
            return 'Recyclable'
        
        val_str = str(value).lower().strip()
        
        # Comprehensive mapping from notebook
        recycling_map = {
            # Positive recycling
            'recycle': 'Recyclable',
            'recyclable': 'Recyclable',
            'yes': 'Recyclable',
            'widely recyclable': 'Recyclable',
            'recycle glass': 'Recyclable',
            'recycle paper': 'Recyclable',
            'recycle plastic': 'Recyclable',
            'recycle metal': 'Recyclable',
            
            # Negative
            'discard': 'Not Recyclable',
            'not recyclable': 'Not Recyclable',
            'no': 'Not Recyclable',
            
            # Special categories
            'compost': 'Compost',
            'compostable': 'Compost',
            'reuse': 'Reusable',
            'reusable': 'Reusable',
            'deposit': 'Deposit Return',
            'deposit return': 'Deposit Return',
            'return': 'Return to Store',
            'return to store': 'Return to Store',
            
            # Conditional
            'recycle with conditions': 'Recycle with Conditions',
            'recycle if clean': 'Recycle with Conditions'
        }
        
        return recycling_map.get(val_str, 'Recyclable')
    
    def _standardize_shape(self, value):
        """EXACT standardization from training notebook"""
        if not value or pd.isna(value):
            return 'bottle'
        
        val_str = str(value).lower().strip()
        
        # Comprehensive mapping from notebook
        shape_map = {
            # Compound/descriptive names -> standard
            'individual-bag': 'bag',
            'plastic-bag': 'bag',
            'resealable-bag': 'bag',
            'rectangular-bag': 'bag',
            'transparent-bag': 'bag',
            
            'food-can': 'can',
            'drink-can': 'can',
            
            'bottle-cap': 'cap',
            'pouring-cap': 'cap',
            
            'twist-off-lid': 'lid',
            'rectangular-lid': 'lid',
            
            'pizza-box': 'box',
            'rectangular-box': 'box',
            'corrugated-cardboard': 'box',
            'egg-carton': 'box',
            
            'rectangular': 'box',
            'square': 'box',
            'round': 'container',
            'cone': 'container',
            
            'individual-pot': 'pot',
            'small-dish': 'tray',
            'multi-cell-tray': 'tray',
            
            'small-bottle': 'bottle',
            'glass-jar': 'jar',
            'reusable-glass': 'jar',
            
            'seal-film': 'film',
            'foil-wrap': 'film',
            'cellophane-wrap': 'film',
            
            # Standard shapes (keep as-is)
            'bottle': 'bottle',
            'box': 'box',
            'bag': 'bag',
            'can': 'can',
            'jar': 'jar',
            'pouch': 'pouch',
            'tray': 'tray',
            'tube': 'tube',
            'container': 'container',
            'wrapper': 'wrapper',
            'pot': 'pot',
            'cap': 'cap',
            'lid': 'lid',
            'seal': 'seal',
            'film': 'film',
            'sleeve': 'sleeve'
        }
        
        return shape_map.get(val_str, val_str)
    
    def load_models(self):
        """Load all model files with comprehensive validation"""
        try:
            print("="*80)
            print("LOADING ML MODELS - PRODUCTION PIPELINE")
            print("="*80)
            
            model_dir = Path('models')
            required_files = {
                'cost_model': model_dir / 'final_cost_model.pkl',
                'co2_model': model_dir / 'final_co2_model.pkl',
                'encoders': model_dir / 'label_encoders.pkl',
                'cost_features': model_dir / 'cost_features.txt',
                'co2_features': model_dir / 'co2_features.txt'
            }
            
            # Check file existence
            missing = [str(f) for f in required_files.values() if not f.exists()]
            if missing:
                print(f"[ERROR] Missing files: {missing}")
                print("[WARNING] Running in DEMO mode")
                return False
            
            # Load models
            self.cost_model = joblib.load(required_files['cost_model'])
            self.co2_model = joblib.load(required_files['co2_model'])
            self.label_encoders = joblib.load(required_files['encoders'])
            
            # Load feature lists (PRESERVES EXACT ORDER)
            with open(required_files['cost_features'], 'r') as f:
                self.cost_features = [line.strip() for line in f if line.strip()]
            
            with open(required_files['co2_features'], 'r') as f:
                self.co2_features = [line.strip() for line in f if line.strip()]
            
            print(f"[OK] Cost model: {len(self.cost_features)} features")
            print(f"[OK] CO2 model: {len(self.co2_features)} features")
            print(f"[OK] Encoders: {len(self.label_encoders)} categories")
            
            # Validate pipeline
            return self._validate_feature_pipeline()
            
        except Exception as e:
            print(f"[ERROR] Loading failed: {e}")
            traceback.print_exc()
            return False
    
    def _validate_feature_pipeline(self):
        """Comprehensive validation with actual test case"""
        try:
            print("\n" + "="*80)
            print("VALIDATING FEATURE PIPELINE")
            print("="*80)
            
            # Test input (matches notebook validation)
            test_input = {
                'product_quantity': 500,
                'weight_measured': 50,
                'weight_capacity': 600,
                'number_of_units': 1,
                'recyclability_percent': 70,
                'material': 'plastic',
                'shape': 'bottle',
                'strength': 'Medium',
                'recycling': 'Recyclable',
                'food_group': 'fruit-juices'
            }
            
            # Add parent_material
            test_input['parent_material'] = self._infer_parent_material(test_input['material'])
            
            print("\n[Test 1] Feature Generation")
            features = self.engineer_features(test_input)
            print(f"  [OK] Generated {len(features)} features")
            
            # Validate cost features
            print("\n[Test 2] Cost Feature Validation")
            missing_cost = [f for f in self.cost_features if f not in features]
            if missing_cost:
                print(f"  [CRITICAL] Missing {len(missing_cost)} cost features:")
                for feat in missing_cost[:10]:
                    print(f"    - {feat}")
                return False
            print(f"  [OK] All {len(self.cost_features)} cost features present")
            
            # Validate CO2 features
            print("\n[Test 3] CO2 Feature Validation")
            missing_co2 = [f for f in self.co2_features if f not in features]
            if missing_co2:
                print(f"  [CRITICAL] Missing {len(missing_co2)} CO2 features:")
                for feat in missing_co2[:10]:
                    print(f"    - {feat}")
                return False
            print(f"  [OK] All {len(self.co2_features)} CO2 features present")
            
            # Test predictions
            print("\n[Test 4] Prediction Test")
            cost_pred, co2_pred, _ = self.predict(test_input)
            
            if cost_pred is None or co2_pred is None:
                print("  [CRITICAL] Prediction failed")
                return False
            
            print(f"  [RESULT] Cost: ₹{cost_pred:.2f}")
            print(f"  [RESULT] CO2: {co2_pred:.2f}")
            
            # Sanity checks
            if not (1 <= cost_pred <= 500):
                print(f"  [WARNING] Unusual cost: ₹{cost_pred:.2f}")
            if not (0.1 <= co2_pred <= 500):
                print(f"  [WARNING] Unusual CO2: {co2_pred:.2f}")
            
            print("\n" + "="*80)
            print("✓ VALIDATION PASSED - Pipeline matches training")
            print("="*80)
            return True
            
        except Exception as e:
            print(f"\n[CRITICAL] Validation failed: {e}")
            traceback.print_exc()
            return False
    
    def _encode_categorical(self, value, column_name):
        """
        Encode categorical with standardization
        CRITICAL: Standardize BEFORE encoding
        """
        if column_name not in self.label_encoders:
            print(f"[ERROR] No encoder for {column_name}")
            return 0
        
        encoder = self.label_encoders[column_name]
        available_classes = list(encoder.classes_)
        
        # STEP 1: STANDARDIZE (CRITICAL)
        if column_name == 'recycling':
            value = self._standardize_recycling(value)
        elif column_name == 'shape':
            value = self._standardize_shape(value)
        
        value_str = str(value).strip()
        
        # STEP 2: EXACT MATCH
        if value_str in available_classes:
            return int(encoder.transform([value_str])[0])
        
        # STEP 3: CASE-INSENSITIVE MATCH
        value_lower = value_str.lower()
        for cls in available_classes:
            if str(cls).lower() == value_lower:
                return int(encoder.transform([str(cls)])[0])
        
        # STEP 4: SAFE FALLBACK
        if available_classes:
            print(f"[WARNING] Unknown {column_name}='{value}', using: '{available_classes[0]}'")
            return int(encoder.transform([available_classes[0]])[0])
        
        return 0
    
    def engineer_features(self, product_dict):
        """
        EXACT FEATURE ENGINEERING from training notebook
        Must match line-by-line to ensure correct predictions
        """
        features = {}
        
        # ================================================================
        # RAW NUMERIC FEATURES
        # ================================================================
        features['product_quantity'] = float(product_dict.get('product_quantity', 500))
        features['weight_measured'] = float(product_dict.get('weight_measured', 50))
        features['weight_capacity'] = float(product_dict.get('weight_capacity', 600))
        features['recyclability_percent'] = float(product_dict.get('recyclability_percent', 70))
        features['number_of_units'] = int(product_dict.get('number_of_units', 1))
        
        # ================================================================
        # CATEGORICAL ENCODINGS
        # ================================================================
        material = str(product_dict.get('material', 'plastic')).lower().strip()
        shape = str(product_dict.get('shape', 'bottle')).lower().strip()
        strength = str(product_dict.get('strength', 'Medium')).strip()
        
        # CRITICAL: Use centralized inference
        parent_material = product_dict.get('parent_material')
        if not parent_material or str(parent_material).lower() in ['', 'nan', 'none', 'unknown']:
            parent_material = self._infer_parent_material(material)
        
        categorical_fields = {
            'food_group': product_dict.get('food_group', 'fruit-juices'),
            'material': material,
            'parent_material': parent_material,
            'recycling': product_dict.get('recycling', 'Recyclable'),
            'shape': shape,
            'strength': strength
        }
        
        for field_name, value in categorical_fields.items():
            encoded_name = f'{field_name}_encoded'
            features[encoded_name] = self._encode_categorical(value, field_name)
        
        # ================================================================
        # POLYNOMIAL FEATURES
        # ================================================================
        weight = features['weight_measured']
        
        features['weight_squared'] = weight ** 2
        features['weight_log'] = np.log1p(weight)
        features['weight_sqrt'] = np.sqrt(weight)
        
        # ================================================================
        # INTERACTION FEATURES
        # ================================================================
        capacity = features['weight_capacity']
        material_enc = features['material_encoded']
        parent_mat_enc = features['parent_material_encoded']
        shape_enc = features['shape_encoded']
        
        # Weight x Capacity
        features['capacity_weight_ratio'] = capacity / (weight + 0.01)
        features['capacity_weight_prod'] = capacity * weight
        
        # Material x Weight
        features['material_weight'] = material_enc * weight
        features['material_weight_sq'] = material_enc * (weight ** 2)
        
        # Parent Material x Weight
        features['parent_mat_weight'] = parent_mat_enc * weight
        
        # Shape x Weight
        features['shape_weight'] = shape_enc * weight
        
        # ================================================================
        # COST-SPECIFIC DERIVED FEATURES
        # ================================================================
        product_qty = features['product_quantity']
        features['packaging_ratio'] = weight / (product_qty + 1)
        
        recyc_pct = features['recyclability_percent']
        features['recyclability_score'] = recyc_pct / 100
        features['non_recyclable_penalty'] = 100 - recyc_pct
        
        return features
    
    def predict(self, product_dict):
        """
        Generate predictions with EXACT feature order
        Returns: (cost_pred, co2_pred, features_dict)
        """
        try:
            if self.cost_model is None or self.co2_model is None:
                print("[ERROR] Models not loaded")
                return None, None, None
            
            # Generate ALL features
            features = self.engineer_features(product_dict)
            
            # ============================================================
            # COST PREDICTION - EXACT ORDER
            # ============================================================
            X_cost = np.array([features[f] for f in self.cost_features]).reshape(1, -1)
            cost_pred = float(self.cost_model.predict(X_cost)[0])
            
            # ============================================================
            # CO2 PREDICTION - EXACT ORDER + LOG TRANSFORM
            # ============================================================
            X_co2 = np.array([features[f] for f in self.co2_features]).reshape(1, -1)
            co2_pred_log = self.co2_model.predict(X_co2)[0]
            co2_pred = float(np.expm1(co2_pred_log))
            
            # ============================================================
            # VALIDATION
            # ============================================================
            is_valid, diagnostics = PredictionValidator.validate_prediction(
                cost_pred=cost_pred,
                co2_pred=co2_pred,
                product_dict=product_dict,
                verbose=True
            )
            
            return cost_pred, co2_pred, features
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            traceback.print_exc()
            return None, None, None
  
ml_manager = MLModelManager()  
  
# ============================================================================  
# DATABASE (unchanged)  
# ============================================================================  
  
def get_db_connection():  
    try:  
        return psycopg2.connect(DATABASE_URL, sslmode='prefer')  
    except Exception as e:  
        print(f"Database error: {e}")  
        return None  
  
def init_db():  
    conn = get_db_connection()  
    if conn:  
        try:  
            cur = conn.cursor()  
              
            cur.execute('''  
                CREATE TABLE IF NOT EXISTS users (  
                    id SERIAL PRIMARY KEY,  
                    email VARCHAR(255) UNIQUE NOT NULL,  
                    password_hash VARCHAR(255) NOT NULL,  
                    full_name VARCHAR(255),  
                    organization VARCHAR(255),  
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  
                    last_login TIMESTAMP  
                )  
            ''')  
              
            cur.execute('''  
                CREATE TABLE IF NOT EXISTS recommendations_history (  
                    id SERIAL PRIMARY KEY,  
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,  
                    product_details JSONB NOT NULL,  
                    current_packaging JSONB,  
                    recommendations JSONB NOT NULL,  
                    cost_savings DECIMAL(10, 2),  
                    co2_reduction DECIMAL(10, 2),  
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  
                )  
            ''')  
              
            cur.execute('''  
                CREATE TABLE IF NOT EXISTS user_analytics (  
                    id SERIAL PRIMARY KEY,  
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,  
                    total_recommendations INTEGER DEFAULT 0,  
                    total_cost_savings DECIMAL(15, 2) DEFAULT 0,  
                    total_co2_reduction DECIMAL(15, 2) DEFAULT 0,  
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP  
                )  
            ''')  
              
            cur.execute('''  
                CREATE TABLE IF NOT EXISTS bulk_uploads (  
                    id SERIAL PRIMARY KEY,  
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,  
                    filename VARCHAR(255),  
                    total_rows INTEGER,  
                    processed_rows INTEGER,  
                    results JSONB,  
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  
                )  
            ''')  
              
            conn.commit()  
            cur.close()  
            conn.close()  
            print("✓ Database initialized")  
        except Exception as e:  
            print(f"✗ Database init error: {e}")  
  
# ============================================================================  
# JWT (unchanged)  
# ============================================================================  
  
def token_required(f): 
    @wraps(f) 
    def decorated(*args, **kwargs): 
        token = None 
 
        # 1️⃣ Check Authorization header 
        auth_header = request.headers.get('Authorization') 
        if auth_header and auth_header.startswith("Bearer "): 
            token = auth_header.split(" ")[1] 
 
        # 2️⃣ Fallback to cookie (IMPORTANT) 
        if not token: 
            token = request.cookies.get('token') 
 
        if not token: 
            return jsonify({"message": "Token is missing"}), 401 
 
        try: 
            data = jwt.decode( 
                token, 
                app.config['JWT_SECRET'], 
                algorithms=["HS256"] 
            ) 
            current_user_id = data['user_id'] 
        except jwt.ExpiredSignatureError: 
            return jsonify({"message": "Token expired"}), 401 
        except jwt.InvalidTokenError: 
            return jsonify({"message": "Invalid token"}), 401 
 
        return f(current_user_id, *args, **kwargs) 
    return decorated

# ============================================================================  
# HELPER FUNCTIONS
# ============================================================================  

def validate_email(email):
    """Validate email format"""
    EMAIL_REGEX = r'^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?@[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z]{2,})+$'
    return re.match(EMAIL_REGEX, email) is not None

def validate_password(password):
    """
    Validate password strength
    Returns: (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_alternatives_v2(product_details, num_alternatives=5, current_cost=None, current_co2=None):
    """
    Generate EXACTLY num_alternatives packaging alternatives
    
    Improvements:
    - Dynamic scaling based on requested count
    - Broader material exploration
    - Multi-strategy approach
    - Guaranteed minimum results
    """
    
    print(f"\n{'='*80}")
    print(f"GENERATING {num_alternatives} ALTERNATIVES")
    print(f"{'='*80}")
    
    current_material = product_details.get('material', 'plastic')
    current_shape = product_details.get('shape', 'bottle')
    current_weight = float(product_details.get('weight_measured', 50))
    
    print(f"Current: {current_material} {current_shape} ({current_weight}g)")
    
    # ========================================================================
    # EXPANDED MATERIAL & SHAPE OPTIONS
    # ========================================================================
    
    all_materials = ['aluminium', 'cardboard', 'glass', 'metal', 'paper', 'plastic', 'steel']
    
    shape_material_compatibility = {
        'bottle': ['plastic', 'glass', 'aluminium', 'cardboard', 'metal'],
        'box': ['cardboard', 'paper', 'plastic', 'metal'],
        'bag': ['paper', 'plastic', 'aluminium'],
        'can': ['aluminium', 'metal', 'steel', 'plastic'],
        'jar': ['glass', 'plastic', 'metal'],
        'pouch': ['plastic', 'paper', 'aluminium'],
        'tray': ['cardboard', 'plastic', 'aluminium', 'paper'],
        'tube': ['plastic', 'aluminium', 'cardboard', 'metal'],
        'container': ['plastic', 'glass', 'cardboard', 'metal', 'paper'],
        'wrapper': ['paper', 'plastic', 'aluminium'],
        'film': ['plastic', 'aluminium', 'paper'],
        'lid': ['plastic', 'metal', 'aluminium', 'cardboard'],
        'cap': ['plastic', 'metal', 'aluminium'],
    }
    
    alternative_shapes = {
        'bottle': ['jar', 'can', 'pouch', 'container'],
        'box': ['bag', 'pouch', 'tray', 'container'],
        'bag': ['pouch', 'box', 'wrapper'],
        'can': ['bottle', 'jar', 'container'],
        'jar': ['bottle', 'can', 'container'],
        'pouch': ['bag', 'box', 'wrapper'],
        'tray': ['box', 'container', 'lid'],
        'container': ['box', 'jar', 'bottle'],
        'wrapper': ['bag', 'pouch', 'film'],
    }
    
    # ========================================================================
    # MULTI-STRATEGY CANDIDATE GENERATION
    # ========================================================================
    
    candidates = []
    tested_configs = set()
    
    base_shape = current_shape if current_shape in shape_material_compatibility else 'container'
    compatible_materials = shape_material_compatibility.get(base_shape, all_materials)
    
    # STRATEGY 1: Same shape, ALL compatible materials
    print(f"\n[Strategy 1] Same shape ({base_shape}), all materials")
    for material in compatible_materials:
        if material == current_material:
            continue
        
        config_key = f"{material}_{base_shape}"
        if config_key in tested_configs:
            continue
        tested_configs.add(config_key)
        
        alt = product_details.copy()
        alt['material'] = material
        alt['shape'] = base_shape
        alt['parent_material'] = ml_manager._infer_parent_material(material)
        
        # Adjust strength
        if material in ['glass', 'metal', 'aluminium', 'steel']:
            alt['strength'] = 'High'
        elif material in ['cardboard', 'paper']:
            alt['strength'] = 'Medium' if current_weight < 100 else 'Low'
        else:
            alt['strength'] = product_details.get('strength', 'Medium')
        
        # Adjust recyclability
        recyclability_map = {
            'paper': 95, 'cardboard': 95, 'glass': 100,
            'aluminium': 100, 'metal': 100, 'steel': 100,
            'plastic': 70
        }
        alt['recyclability_percent'] = recyclability_map.get(material, 70)
        
        candidates.append(alt)
        print(f"  ✓ Added: {material} {base_shape}")
    
    # STRATEGY 2: Alternative shapes with multiple materials
    if base_shape in alternative_shapes and len(candidates) < num_alternatives * 3:
        print(f"\n[Strategy 2] Alternative shapes")
        for alt_shape in alternative_shapes[base_shape]:
            shape_materials = shape_material_compatibility.get(alt_shape, all_materials)
            
            for material in shape_materials[:4]:  # Top 4 materials per shape
                if material == current_material and alt_shape == current_shape:
                    continue
                
                config_key = f"{material}_{alt_shape}"
                if config_key in tested_configs:
                    continue
                tested_configs.add(config_key)
                
                alt = product_details.copy()
                alt['material'] = material
                alt['shape'] = alt_shape
                alt['parent_material'] = ml_manager._infer_parent_material(material)
                
                # Adjust properties
                if material in ['glass', 'metal', 'aluminium', 'steel']:
                    alt['strength'] = 'High'
                elif material in ['cardboard', 'paper']:
                    alt['strength'] = 'Medium' if current_weight < 100 else 'Low'
                else:
                    alt['strength'] = product_details.get('strength', 'Medium')
                
                recyclability_map = {
                    'paper': 95, 'cardboard': 95, 'glass': 100,
                    'aluminium': 100, 'metal': 100, 'steel': 100,
                    'plastic': 70
                }
                alt['recyclability_percent'] = recyclability_map.get(material, 70)
                
                candidates.append(alt)
                print(f"  ✓ Added: {material} {alt_shape}")
                
                # EARLY EXIT if we have enough candidates
                if len(candidates) >= num_alternatives * 3:
                    break
            
            if len(candidates) >= num_alternatives * 3:
                break
    
    print(f"\n✓ Generated {len(candidates)} total candidates")
    
    # ========================================================================
    # PREDICT & SCORE ALL CANDIDATES
    # ========================================================================
    
    scored_candidates = []
    
    for alt in candidates:
        alt_cost, alt_co2, _ = ml_manager.predict(alt)
        
        if alt_cost and alt_co2:
            cost_savings = (current_cost - alt_cost) if current_cost else 0
            co2_reduction = (current_co2 - alt_co2) if current_co2 else 0
            
            cost_savings_pct = (cost_savings / current_cost * 100) if current_cost and current_cost > 0 else 0
            co2_reduction_pct = (co2_reduction / current_co2 * 100) if current_co2 and current_co2 > 0 else 0
            recyclability_improvement = alt['recyclability_percent'] - product_details.get('recyclability_percent', 70)
            
            # Weighted scoring
            overall_score = (
                co2_reduction_pct * 0.50 +
                cost_savings_pct * 0.30 +
                recyclability_improvement * 0.20
            )
            
            scored_candidates.append({
                'config': alt,
                'cost': alt_cost,
                'co2': alt_co2,
                'cost_savings': cost_savings,
                'co2_reduction': co2_reduction,
                'cost_savings_pct': cost_savings_pct,
                'co2_reduction_pct': co2_reduction_pct,
                'overall_score': overall_score,
                'material': alt['material'],
                'shape': alt['shape'],
            })
    
    # Sort by score
    scored_candidates.sort(key=lambda x: x['overall_score'], reverse=True)
    
    # ========================================================================
    # RETURN EXACTLY num_alternatives
    # ========================================================================
    
    # Filter: Keep ANY with positive score OR top alternatives
    improvements = [c for c in scored_candidates if c['overall_score'] > -15]
    
    # GUARANTEE MINIMUM RESULTS
    if len(improvements) < num_alternatives:
        print(f"⚠ Only {len(improvements)} improvements found")
        # Add more even if not optimal
        improvements = scored_candidates[:num_alternatives * 2]
    
    result = []
    for i, candidate in enumerate(improvements[:num_alternatives], 1):
        print(f"\n#{i}. {candidate['material']} {candidate['shape']}")
        print(f"    Score: {candidate['overall_score']:.1f}")
        print(f"    Cost: ₹{candidate['cost']:.2f} ({candidate['cost_savings_pct']:+.1f}%)")
        print(f"    CO₂: {candidate['co2']:.2f} ({candidate['co2_reduction_pct']:+.1f}%)")
        result.append(candidate['config'])
    
    # If still not enough, duplicate top alternatives
    while len(result) < num_alternatives:
        print(f"⚠ Duplicating top alternative to reach {num_alternatives}")
        if result:
            result.append(result[0].copy())
        else:
            result.append(product_details.copy())
    
    print(f"\n✓ Returning {len(result)} alternatives (requested: {num_alternatives})")
    
    return result

def calculate_score(cost_savings, co2_reduction):
    """
    Calculate improvement score based on cost savings and CO2 reduction
    
    Args:
        cost_savings (float): Cost savings in currency
        co2_reduction (float): CO2 reduction in units
    
    Returns:
        float: Combined improvement score
    """
    # Weighted score: 60% cost, 40% environmental impact
    cost_score = cost_savings * 0.6
    co2_score = co2_reduction * 0.4
    return round(cost_score + co2_score, 2)

def save_recommendation(user_id, product_details, current_packaging, recommendations):
    """
    Save recommendation to database
    
    Args:
        user_id (int): User ID
        product_details (dict): Product configuration
        current_packaging (dict): Current packaging metrics
        recommendations (list): List of recommendations
    """
    try:
        conn = get_db_connection()
        if not conn:
            print("⚠️ Could not save recommendation - DB connection failed")
            return False
            
        cur = conn.cursor()
        
        # Calculate totals
        total_cost_savings = sum(r.get('cost_savings', 0) for r in recommendations)
        total_co2_reduction = sum(r.get('co2_reduction', 0) for r in recommendations)
        
        # Insert recommendation history
        cur.execute('''
            INSERT INTO recommendations_history 
            (user_id, product_details, current_packaging, recommendations, cost_savings, co2_reduction)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (
            user_id,
            json.dumps(product_details),
            json.dumps(current_packaging),
            json.dumps(recommendations),
            total_cost_savings,
            total_co2_reduction
        ))
        
        # Update user analytics
        cur.execute('''
            UPDATE user_analytics 
            SET total_recommendations = total_recommendations + 1,
                total_cost_savings = total_cost_savings + %s,
                total_co2_reduction = total_co2_reduction + %s,
                last_updated = %s
            WHERE user_id = %s
        ''', (total_cost_savings, total_co2_reduction, datetime.datetime.now(), user_id))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error saving recommendation: {e}")
        if conn:
            conn.rollback()
            cur.close()
            conn.close()
        return False
 
# ============================================================================  
# ROUTES (unchanged except /api/materials and /api/features)  
# ============================================================================  
  
@app.route('/')  
def index():  
    return render_template('index.html')  
  
@app.route('/dashboard')
def dashboard():
    print("=== Dashboard Route Hit ===")
    token = request.cookies.get('token')
    print(f"Token from cookie: {token}")
    
    if not token:
        print("No token found in cookies!")
        print(f"All cookies: {request.cookies}")
        return redirect('/?error=no_token')

    try:
        decoded = jwt.decode(token, app.config['JWT_SECRET'], algorithms=["HS256"])
        print(f"Token decoded successfully: {decoded}")
    except jwt.ExpiredSignatureError:
        print("Token expired!")
        return redirect('/?error=token_expired')
    except jwt.InvalidTokenError as e:
        print(f"Invalid token: {e}")
        return redirect('/?error=invalid_token')

    user_id = decoded.get('user_id')
    print(f"Rendering dashboard for user_id: {user_id}")

    return render_template('dashboard.html', user_id=user_id)

@app.route('/api/register', methods=['POST'])  
def register():  
    try:  
        data = request.get_json()  
        email = data.get('email', '').strip()  
        password = data.get('password', '')  
        full_name = data.get('full_name', '').strip()  
          
        if not validate_email(email):  
            return jsonify({'error': 'Invalid email'}), 400  
          
        valid, msg = validate_password(password)  
        if not valid:  
            return jsonify({'error': msg}), 400  
          
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')  
          
        conn = get_db_connection()  
        if not conn:  
            return jsonify({'error': 'Database connection failed'}), 503  
        cur = conn.cursor(cursor_factory=RealDictCursor)  
          
        try:  
            cur.execute(  
                'INSERT INTO users (email, password_hash, full_name) VALUES (%s, %s, %s) RETURNING id',  
                (email, password_hash, full_name)  
            )  
            user_id = cur.fetchone()['id']  
            cur.execute('INSERT INTO user_analytics (user_id) VALUES (%s)', (user_id,))  
            conn.commit()  
            return jsonify({'message': 'Registered successfully', 'user_id': user_id}), 201  
        except psycopg2.IntegrityError:  
            conn.rollback()  
            return jsonify({'error': 'Email already registered'}), 409  
        finally:  
            cur.close()  
            conn.close()  
    except Exception as e:  
        return jsonify({'error': str(e)}), 500  
  
@app.route('/api/login', methods=['POST'])  
def api_login():  
    try:  
        data = request.get_json()
        email = data.get('email', '').strip()  
        password = data.get('password', '')  
          
        conn = get_db_connection()  
        if not conn: 
            return jsonify({'error': 'Database connection failed'}), 503
 
        cur = conn.cursor(cursor_factory=RealDictCursor)  
        cur.execute('SELECT * FROM users WHERE email = %s', (email,))  
        user = cur.fetchone()  
          
        if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            cur.close()
            conn.close()
            return jsonify({'error': 'Invalid email or password'}), 401
          
        cur.execute('UPDATE users SET last_login = %s WHERE id = %s', (datetime.datetime.now(), user['id']))  
        conn.commit()  
          
        token = jwt.encode({  
            'user_id': user['id'],  
            'email': user['email'],  
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)  
        }, app.config['JWT_SECRET'], algorithm='HS256')  
          
        cur.close()  
        conn.close()  
    
        # RETURN JSON RESPONSE
        response = jsonify({
            'success': True,
            'message': 'Login successful',
            'token': token
        })
        
        # SET COOKIE
        response.set_cookie(
            'token',
            token,
            max_age=86400,  # 24 hours
            httponly=True,
            samesite='Lax',
            secure=True,  # Required for HTTPS (Render)
            path='/'
        )
        
        return response, 200
    
    except Exception as e:  
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/info', methods=['GET'])
@token_required
def get_user_info(current_user_id):
    """Get current authenticated user's information"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 503
        
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('SELECT id, email, full_name, organization FROM users WHERE id = %s', (current_user_id,))
        user = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'user': {
                'id': user['id'],
                'email': user['email'],
                'full_name': user['full_name'],
                'organization': user['organization']
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    response = jsonify({'success': True, 'message': 'Logged out successfully'})
    response.set_cookie('token', '', expires=0, path='/')  # Clear the cookie
    return response, 200
 
@app.route('/api/recommend', methods=['POST'])  
@token_required  
def get_recommendations(current_user_id):  
    try:  
        data = request.get_json()  
        
        # ========================================================================
        # STEP 1: BUILD COMPLETE INPUT (EXACT MATCH TO TRAINING)
        # ========================================================================
        product_details = {  
            'product_quantity': float(data.get('product_quantity', 500)),  
            'weight_measured': float(data.get('weight_measured', 50)),  
            'weight_capacity': float(data.get('weight_capacity', 600)),  
            'number_of_units': int(data.get('number_of_units', 1)),  
            'recyclability_percent': float(data.get('recyclability_percent', 70)), 
            
            # Categorical fields (CRITICAL: Must match training exactly)
            'food_group': data.get('food_group', 'fruit-juices'),  
            'material': data.get('material', 'plastic').lower().strip(),  
            'shape': data.get('shape', 'bottle').lower().strip(),  
            'strength': data.get('strength', 'Medium').strip(),  
            'recycling': data.get('recycling', 'Recyclable'),
            
            # Parent material (inferred if missing)
            'parent_material': data.get('parent_material', ''),
        }
        
        # CRITICAL: Infer parent_material if not provided
        if not product_details['parent_material'] or product_details['parent_material'] in ['', 'nan', 'none', 'unknown']:
            product_details['parent_material'] = ml_manager._infer_parent_material(
                product_details['material']
            )
        
        num_units = product_details['number_of_units']
        num_alternatives_requested = int(data.get('number_of_alternatives', 5))  # ✅ Read from request
        
        # ========================================================================
        # STEP 2: ML PREDICTION (PER-UNIT COST)
        # ========================================================================
        print(f"\n[Prediction Request]")
        print(f"  Material: {product_details['material']}")
        print(f"  Shape: {product_details['shape']}")
        print(f"  Weight: {product_details['weight_measured']}g")
        print(f"  Units: {num_units}")
        print(f"  Alternatives requested: {num_alternatives_requested}")
        
        cost_per_unit, current_co2, features_used = ml_manager.predict(product_details)
        
        # ✅ VALIDATION: Ensure prediction succeeded
        if cost_per_unit is None or current_co2 is None:
            print("\n❌ PREDICTION FAILED")
            print(f"   Input: {product_details}")
            return jsonify({
                'error': 'Model prediction failed. Check server logs for details.',
                'debug': {
                    'input': product_details,
                    'cost_result': cost_per_unit,
                    'co2_result': current_co2
                }
            }), 500
        
        # ========================================================================
        # STEP 3: VOLUME DISCOUNT CALCULATION
        # ========================================================================
        def calculate_volume_discount(units):
            if units <= 10: return 0.00
            elif units <= 50: return 0.08
            elif units <= 200: return 0.15
            elif units <= 1000: return 0.22
            else: return 0.30
        
        discount_rate = calculate_volume_discount(num_units)
        discount_multiplier = 1 - discount_rate
        current_cost = cost_per_unit * num_units * discount_multiplier
        
        print(f"\n[Cost Calculation]")
        print(f"  Per-unit cost: ₹{cost_per_unit:.4f}")
        print(f"  Quantity: {num_units}")
        print(f"  Volume discount: {discount_rate*100:.0f}%")
        print(f"  TOTAL cost: ₹{current_cost:.2f}")
        
        # ========================================================================
        # STEP 4: GENERATE ALTERNATIVES (DYNAMIC COUNT)
        # ========================================================================
        alternatives = generate_alternatives_v2(
            product_details, 
            num_alternatives=num_alternatives_requested, 
            current_cost=current_cost,
            current_co2=current_co2
        )
        
        print(f"\n[Alternatives Generated: {len(alternatives)}]")
        
        # ========================================================================
        # STEP 5: PREDICT FOR EACH ALTERNATIVE
        # ========================================================================
        recommendations = []  
        
        for alt in alternatives:  
            alt_cost_per_unit, alt_co2, _ = ml_manager.predict(alt)  
            
            if alt_cost_per_unit and alt_co2:
                # Apply SAME volume discount to alternative
                alt_total_cost = alt_cost_per_unit * num_units * discount_multiplier
                
                cost_savings = current_cost - alt_total_cost  
                co2_reduction = current_co2 - alt_co2
                
                recommendations.append({  
                    # Packaging specs
                    'material': alt['material'],  
                    'parent_material': alt.get('parent_material', ''),
                    'shape': alt['shape'],  
                    'strength': alt['strength'],  
                    'recycling': alt.get('recycling', 'Recyclable'),
                    'food_group': alt.get('food_group', ''),
                    
                    # Physical properties
                    'product_quantity': alt.get('product_quantity', 0),
                    'weight_measured': alt.get('weight_measured', 0),
                    'weight_capacity': alt.get('weight_capacity', 0),
                    'number_of_units': alt.get('number_of_units', 1),
                    'recyclability_percent': alt.get('recyclability_percent', 0),
                    
                    # Predictions (Frontend must read these fields)
                    'predicted_cost_per_unit': round(alt_cost_per_unit, 2),
                    'predicted_cost': round(alt_total_cost, 2),  
                    'predicted_co2': round(alt_co2, 2),
                    
                    # Savings
                    'cost_savings': round(cost_savings, 2),  
                    'co2_reduction': round(co2_reduction, 2),  
                    'improvement_score': calculate_score(cost_savings, co2_reduction)
                })
        
        # Sort by improvement score
        recommendations.sort(key=lambda x: x['improvement_score'], reverse=True)
        
        # ENSURE EXACT COUNT (trim or warn)
        if len(recommendations) > num_alternatives_requested:
            recommendations = recommendations[:num_alternatives_requested]
        elif len(recommendations) < num_alternatives_requested:
            print(f"⚠ WARNING: Only found {len(recommendations)}/{num_alternatives_requested} alternatives")
        
        # ========================================================================
        # STEP 6: SAVE TO DATABASE
        # ========================================================================
        if recommendations:  
            save_recommendation(
                current_user_id, 
                product_details,
                {'cost': current_cost, 'co2': current_co2},
                recommendations[:5]  # Save top 5 to history
            )
        
        # ========================================================================
        # STEP 7: RESPONSE (Frontend must read these fields)
        # ========================================================================
        response_data = {
            'current_packaging': {  
                # Ensure these field names match frontend expectations
                'cost_per_unit': round(cost_per_unit, 2),   
                'total_cost': round(current_cost, 2),  
                'quantity': num_units,                         
                'volume_discount_pct': round(discount_rate * 100, 1),  
                'co2': round(current_co2, 2),
                'co2_label': 'CO₂ Impact Index', 
                
                # Input echo (for verification)
                'material': product_details['material'], 
                'shape': product_details['shape'], 
                'strength': product_details['strength'], 
                'recyclability_percent': product_details['recyclability_percent']
            },
            
            # RETURN EXACT COUNT REQUESTED
            'recommendations': recommendations,
            
            # Debug info
            'debug': {
                'alternatives_requested': num_alternatives_requested,
                'alternatives_returned': len(recommendations),
                'model_used': 'XGBoost (Optuna Optimized)',
                'features_used': len(features_used) if features_used else 0
            }
        }
        
        print(f"\n✓ SUCCESS:")
        print(f"   Current cost: ₹{current_cost:.2f}")
        print(f"   Alternatives: {len(recommendations)}/{num_alternatives_requested}")
        
        return jsonify(response_data), 200
        
    except Exception as e:  
        print(f"\n ERROR in /api/recommend:")
        print(f"   {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }), 500
  
@app.route('/api/compare', methods=['POST'])  
@token_required  
def compare_materials(current_user_id):  
    try:  
        data = request.get_json()  
        materials = data.get('materials', [])  
          
        if len(materials) < 2:  
            return jsonify({'error': 'Select at least 2 materials'}), 400  
          
        base = {  
            'product_quantity': float(data.get('product_quantity', 500)),  
            'weight_measured': float(data.get('weight_measured', 50)),  
            'weight_capacity': float(data.get('weight_capacity', 600)),  
            'shape': data.get('shape', 'bottle'),  
            'strength': data.get('strength', 'Medium'),  
            'food_group': data.get('food_group', 'fruit-juices'),  
            'recyclability_percent': 70,  
            'number_of_units': 1,  
            'recycling': 'Recyclable'  
        }  
          
        comparisons = []  
        for material in materials:  
            product = base.copy()  
            product['material'] = material  
            product['parent_material'] = material  
              
            cost, co2, _ = ml_manager.predict(product)  
            if cost and co2:  
                comparisons.append({  
                    'material': material,  
                    'cost': round(cost, 2),  
                    'co2': round(co2, 2)  
                })  
          
        return jsonify({'comparisons': comparisons}), 200  
    except Exception as e: 
        return jsonify({'error': str(e)}), 500 
 
@app.route('/api/bulk-upload', methods=['POST'])  
@token_required  
def bulk_upload(current_user_id):  
    try:  
        if 'file' not in request.files:  
            return jsonify({'error': 'No file uploaded'}), 400  
          
        file = request.files['file']  
        if file.filename == '' or not allowed_file(file.filename):  
            return jsonify({'error': 'Invalid file'}), 400  
          
        filename = secure_filename(file.filename)  
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  
        file.save(filepath)  
          
        if filename.endswith('.csv'):  
            df = pd.read_csv(filepath)  
        else:  
            df = pd.read_excel(filepath)  
          
        results = []  
        for idx, row in df.iterrows(): 
            try: 
                # Extract material first for parent_material inference
                material = str(row.get('material', 'plastic')).lower().strip()
                
                # Infer parent_material if not provided (CRITICAL FIX)
                parent_material = str(row.get('parent_material', '')).lower().strip()
                if not parent_material or parent_material in ['', 'nan', 'none']:
                    # Use same inference logic as recommendation route
                    if material in ['aluminium', 'aluminum', 'metal', 'steel']:
                        parent_material = 'metal'
                    elif material in ['cardboard', 'paper', 'paperboard']:
                        parent_material = 'paper-or-cardboard'
                    elif material == 'glass':
                        parent_material = 'glass'
                    elif 'plastic' in material:
                        parent_material = 'plastic'
                    else:
                        parent_material = 'unknown'
                
                # Build COMPLETE input data dictionary
                input_data = { 
                    # Numeric fields
                    'product_quantity': float(row.get('product_quantity', 500)), 
                    'weight_measured': float(row.get('weight_measured', 50)), 
                    'weight_capacity': float(row.get('weight_capacity', 600)), 
                    'number_of_units': int(row.get('number_of_units', 1)), 
                    'recyclability_percent': float(row.get('recyclability_percent', 70)),
                    
                    # Categorical fields (ALL 6 required by models)
                    'material': material,
                    'parent_material': parent_material,  # CRITICAL: Added this
                    'shape': str(row.get('shape', 'bottle')).lower().strip(), 
                    'strength': str(row.get('strength', 'Medium')).strip(),  # Keep capitalization
                    'food_group': str(row.get('food_group', 'fruit-juices')).lower().strip(), 
                    'recycling': str(row.get('recycling', 'Recyclable')).strip(),  # Keep capitalization
                } 
                 
                # Get predictions using ml_manager 
                cost_pred, co2_pred, _ = ml_manager.predict(input_data) 
 
                # Handle prediction failures 
                if cost_pred is None or co2_pred is None: 
                    results.append({ 
                        'row': idx + 1, 
                        'material': input_data['material'], 
                        'parent_material': input_data['parent_material'],  # Include in response
                        'shape': input_data['shape'], 
                        'strength': input_data['strength'], 
                        'food_group': input_data['food_group'], 
                        'recycling': input_data['recycling'], 
                        'product_quantity': input_data['product_quantity'], 
                        'weight_measured': input_data['weight_measured'], 
                        'weight_capacity': input_data['weight_capacity'], 
                        'number_of_units': input_data['number_of_units'], 
                        'recyclability_percent': input_data['recyclability_percent'], 
                        'cost': '-', 
                        'co2': '-', 
                        'status': 'error', 
                        'error': 'Prediction failed - check input data format' 
                    }) 
                    continue 
 
                # Return BOTH input data AND predictions 
                results.append({ 
                    'row': idx + 1, 
                    # Input fields (echo back with inferred parent_material)
                    'material': input_data['material'], 
                    'parent_material': input_data['parent_material'],  # Show inferred value
                    'shape': input_data['shape'], 
                    'strength': input_data['strength'], 
                    'food_group': input_data['food_group'], 
                    'recycling': input_data['recycling'], 
                    'product_quantity': input_data['product_quantity'], 
                    'weight_measured': input_data['weight_measured'], 
                    'weight_capacity': input_data['weight_capacity'], 
                    'number_of_units': input_data['number_of_units'], 
                    'recyclability_percent': input_data['recyclability_percent'], 
                    # Predictions 
                    'cost': round(float(cost_pred), 2), 
                    'co2': round(float(co2_pred), 2), 
                    'status': 'success' 
                }) 
                 
            except Exception as e: 
                results.append({ 
                    'row': idx + 1, 
                    'material': str(row.get('material', 'N/A')), 
                    'status': 'error', 
                    'error': f'Processing error: {str(e)}' 
                }) 
          
        conn = get_db_connection()  
        if not conn: 
            return jsonify({
                'error': 'Database connection failed, results not saved', 
                'results': results
            }), 503 
 
        cur = conn.cursor()  
        cur.execute('''  
            INSERT INTO bulk_uploads (user_id, filename, total_rows, processed_rows, results)  
            VALUES (%s, %s, %s, %s, %s)  
        ''', (current_user_id, filename, len(df), len(results), json.dumps(results)))  
        conn.commit()  
        cur.close()  
        conn.close()  
          
        os.remove(filepath)  
          
        return jsonify({  
            'message': 'Bulk processing complete',  
            'total_rows': len(df),  
            'processed': len(results),  
            'results': results  
        }), 200  
    except Exception as e:  
        return jsonify({'error': str(e)}), 500
  
@app.route('/api/analytics/user', methods=['GET'])  
@token_required  
def get_user_analytics(current_user_id):  
    try:  
        conn = get_db_connection()  
        if not conn: 
             return jsonify({'error': 'Database connection failed'}), 503 

        cur = conn.cursor(cursor_factory=RealDictCursor)  
          
        cur.execute('SELECT * FROM user_analytics WHERE user_id = %s', (current_user_id,))  
        analytics = cur.fetchone()  
          
        cur.execute('''  
            SELECT 
                id,
                product_details,
                current_packaging,
                recommendations,
                cost_savings, 
                co2_reduction, 
                created_at
            FROM recommendations_history  
            WHERE user_id = %s  
            ORDER BY created_at ASC
        ''', (current_user_id,))  
        history = cur.fetchall()

        # Format history data properly
        formatted_history = []
        for record in history:
            formatted_history.append({
                'id': record['id'],
                'product_details': record['product_details'],  # Already JSONB, should be dict
                'current_packaging': record['current_packaging'],
                'recommendations': record['recommendations'],
                'cost_savings': float(record['cost_savings']) if record['cost_savings'] else 0,
                'co2_reduction': float(record['co2_reduction']) if record['co2_reduction'] else 0,
                'created_at': record['created_at'].isoformat() if record['created_at'] else None
            })
          
        cur.close()  
        conn.close()  
          
        return jsonify({  
            'analytics': dict(analytics) if analytics else {},  
            'history': formatted_history,  # Use formatted data
            'recent': formatted_history[-10:]  # Last 10
        }), 200 
    except Exception as e:  
        return jsonify({'error': str(e)}), 500
     
@app.route('/api/history', methods=['GET'])
@token_required
def get_user_history(current_user_id):
    """
    Get complete recommendation history - OPTIMIZED for clean table display
    Returns structured data ready for frontend rendering
    """
    try:
        print(f"\n[History API] Request from user {current_user_id}")
        
        conn = get_db_connection()
        if not conn:
            print("[History API] Database connection failed")
            return jsonify({'error': 'Database connection failed'}), 503
        
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get all recommendations with structured data
        cur.execute('''
            SELECT 
                id,
                product_details,
                current_packaging,
                recommendations,
                cost_savings,
                co2_reduction,
                created_at
            FROM recommendations_history
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 100
        ''', (current_user_id,))
        
        history_records = cur.fetchall()
        
        print(f"[History API] Found {len(history_records)} records")
        
        cur.close()
        conn.close()
        
        # Format the response with clean structure
        result = []
        for record in history_records:
            try:
                # Extract product details from JSONB
                product_details = record['product_details'] if record['product_details'] else {}
                current_pkg = record['current_packaging'] if record['current_packaging'] else {}
                recommendations = record['recommendations'] if record['recommendations'] else []
                
                # Find best recommendation (first one, already sorted by improvement_score)
                best_rec = recommendations[0] if recommendations else None
                
                result.append({
                    'id': record['id'],
                    'created_at': record['created_at'].isoformat() if record['created_at'] else None,
                    
                    # Product specifications
                    'product': {
                        'material': product_details.get('material', 'N/A'),
                        'shape': product_details.get('shape', 'N/A'),
                        'weight_measured': product_details.get('weight_measured', 0),
                        'weight_capacity': product_details.get('weight_capacity', 0),
                        'recyclability_percent': product_details.get('recyclability_percent', 0),
                        'strength': product_details.get('strength', 'N/A'),
                        'food_group': product_details.get('food_group', 'N/A'),
                        'product_quantity': product_details.get('product_quantity', 1)
                    },
                    
                    # Current packaging metrics
                    'current': {
                        'cost': float(current_pkg.get('cost', 0)),
                        'co2': float(current_pkg.get('co2', 0))
                    },
                    
                    # Best alternative (for summary)
                    'best_alternative': {
                        'material': best_rec.get('material', 'N/A') if best_rec else 'N/A',
                        'shape': best_rec.get('shape', 'N/A') if best_rec else 'N/A',
                        'predicted_cost': float(best_rec.get('predicted_cost', 0)) if best_rec else 0,
                        'predicted_co2': float(best_rec.get('predicted_co2', 0)) if best_rec else 0,
                        'cost_savings': float(best_rec.get('cost_savings', 0)) if best_rec else 0,
                        'co2_reduction': float(best_rec.get('co2_reduction', 0)) if best_rec else 0,
                        'improvement_score': float(best_rec.get('improvement_score', 0)) if best_rec else 0
                    },
                    
                    # Overall impact
                    'total_cost_savings': float(record['cost_savings']) if record['cost_savings'] else 0.0,
                    'total_co2_reduction': float(record['co2_reduction']) if record['co2_reduction'] else 0.0,
                    
                    # All recommendations (for expandable view)
                    'all_recommendations': recommendations
                })
            except Exception as e:
                print(f"[History API] Error processing record {record.get('id')}: {e}")
                continue
        
        print(f"[History API] Successfully formatted {len(result)} records")
        
        return jsonify({
            'success': True,
            'history': result,
            'count': len(result)
        }), 200
        
    except Exception as e:
        print(f"[History API] Error fetching history: {e}")
        print(f"[History API] Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500
# ============================================================================  
# CORRECTED: /api/materials - Returns ONLY fields models actually use 
# ============================================================================  
  
@app.route('/api/materials', methods=['GET'])  
def get_materials(): 
    try:  
        # Define ONLY the 6 categorical fields used by models 
        REQUIRED_FIELDS = ['material', 'parent_material', 'shape', 'strength', 'food_group', 'recycling'] 
         
        if ml_manager.label_encoders is None:  
            # Fallback categories (if models not loaded)  
            options = {  
                'material': ['plastic', 'cardboard', 'glass', 'metal', 'paper', 'aluminium'],  
                'parent_material': ['plastic', 'glass', 'metal', 'paper-or-cardboard'], 
                'shape': ['bottle', 'box', 'bag', 'can', 'jar', 'pouch', 'tray', 'tube'],  
                'strength': ['Low', 'Medium', 'High', 'Very High'],  
                'food_group': ['fruit-juices', 'biscuits-and-cakes', 'dairy-desserts', 'bread', 'cheese'],  
                'recycling': ['Recyclable', 'Not Recyclable', 'Compost', 'Reusable', 'Deposit Return']  
            }  
            return jsonify({'options': options, 'source': 'fallback'}), 200  
          
        # Get ACTUAL categories from trained encoders - ONLY the 6 fields 
        options = {}  
        for col in REQUIRED_FIELDS:  
            if col in ml_manager.label_encoders:  
                options[col] = sorted(list(ml_manager.label_encoders[col].classes_))  
            else: 
                print(f"⚠️ WARNING: Expected encoder '{col}' not found in label_encoders") 
          
        return jsonify({ 
            'options': options,  
            'source': 'label_encoders', 
            'fields_count': len(options) 
        }), 200  
    except Exception as e:  
        return jsonify({'error': str(e)}), 500 
 
 
# ============================================================================  
# CORRECTED: /api/features - Only show fields models actually use 
# ============================================================================  
  
@app.route('/api/features', methods=['GET'])  
def get_features():  
    """  
    Returns available categories from the trained label encoders  
    ✅ ONLY return the 6 fields that models actually use (from feature files) 
    ❌ categories_tags and countries_tags are NOT in cost_features.txt or co2_features.txt 
    """  
    try: 
        # Define ONLY the 6 categorical fields used by models 
        REQUIRED_FIELDS = ['material', 'parent_material', 'shape', 'strength', 'food_group', 'recycling'] 
         
        if ml_manager.label_encoders is None:  
            return jsonify({  
                'features': {  
                    'material': ['plastic', 'cardboard', 'glass', 'metal', 'paper', 'aluminium'],  
                    'parent_material': ['plastic', 'glass', 'metal', 'paper-or-cardboard'],  
                    'shape': ['bottle', 'box', 'bag', 'can', 'jar', 'pouch', 'tray', 'tube'],  
                    'strength': ['Low', 'Medium', 'High', 'Very High'],  
                    'recycling': ['Recyclable', 'Not Recyclable', 'Compost', 'Reusable'],  
                    'food_group': ['fruit-juices', 'biscuits-and-cakes', 'dairy-desserts', 'bread', 'cheese']  
                },  
                'source': 'fallback', 
                'note': 'Only showing 6 fields used by models' 
            }), 200  
          
        # Extract ONLY the 6 required fields from label encoders 
        features = {}  
        for column_name in REQUIRED_FIELDS: 
            if column_name in ml_manager.label_encoders: 
                features[column_name] = sorted(list(ml_manager.label_encoders[column_name].classes_)) 
            else: 
                print(f"⚠️ WARNING: Expected encoder '{column_name}' not found") 
          
        return jsonify({  
            'features': features,  
            'source': 'label_encoders',  
            'model_info': {  
                'cost_features_total': len(ml_manager.cost_features) if ml_manager.cost_features else 0,  
                'co2_features_total': len(ml_manager.co2_features) if ml_manager.co2_features else 0, 
                'categorical_features_used': len(features), 
                'fields_used': list(features.keys()), 
                'note': 'Only encoders actually referenced in cost_features.txt and co2_features.txt' 
            }  
        }), 200  
    except Exception as e:  
        return jsonify({'error': str(e)}), 500  
  
# ============================================================================  
# CORRECTED: DEBUG ENDPOINT - Only show fields models actually use 
# ============================================================================  
  
@app.route('/api/debug/features', methods=['GET'])  
def debug_features():  
    """Debug endpoint to view label encoder features in HTML format - ONLY fields used by models"""  
    try: 
        # Define ONLY the 6 categorical fields used by models 
        REQUIRED_FIELDS = ['material', 'parent_material', 'shape', 'strength', 'food_group', 'recycling'] 
         
        html = """  
        <!DOCTYPE html>  
        <html lang="en">  
        <head>  
            <meta charset="UTF-8">  
            <meta name="viewport" content="width=device-width, initial-scale=1.0">  
            <title>🔍 EcoPackAI - Debug Features</title>  
            <style>  
                * { margin: 0; padding: 0; box-sizing: border-box; }  
                body {  
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;  
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  
                    padding: 2rem;  
                    min-height: 100vh;  
                }  
                .container {  
                    max-width: 1200px;  
                    margin: 0 auto;  
                    background: white;  
                    border-radius: 20px;  
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);  
                    overflow: hidden;  
                }  
                .header {  
                    background: linear-gradient(135deg, #394648 0%, #4a5a5c 100%);  
                    color: white;  
                    padding: 2rem;  
                    text-align: center;  
                }  
                .header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }  
                .status {  
                    display: inline-block;  
                    padding: 0.5rem 1.5rem;  
                    border-radius: 50px;  
                    font-weight: bold;  
                    margin-top: 1rem;  
                }  
                .status.loaded { background: #10b981; }  
                .status.fallback { background: #f59e0b; }  
                .content { padding: 2rem; }  
                .feature-card {  
                    background: #f9fafb;  
                    border-left: 5px solid #69995D;  
                    border-radius: 12px;  
                    padding: 1.5rem;  
                    margin-bottom: 1.5rem;  
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);  
                }  
                .feature-title {  
                    font-size: 1.5rem;  
                    font-weight: bold;  
                    color: #394648;  
                    margin-bottom: 1rem;  
                }  
                .feature-values {  
                    display: flex;  
                    flex-wrap: wrap;  
                    gap: 0.75rem;  
                }  
                .value-badge {  
                    background: #69995D;  
                    color: white;  
                    padding: 0.5rem 1rem;  
                    border-radius: 8px;  
                    font-weight: 600;  
                    font-size: 0.9rem;  
                }  
                .count {  
                    background: #CBAC88;  
                    color: #394648;  
                    padding: 0.25rem 0.75rem;  
                    border-radius: 50px;  
                    font-size: 0.85rem;  
                    font-weight: bold;  
                }  
                .info-box {  
                    background: #e0f2fe;  
                    border-left: 4px solid #0284c7;  
                    padding: 1rem;  
                    border-radius: 8px;  
                    margin-bottom: 2rem;  
                }  
                .warning-box { 
                    background: #fef3c7; 
                    border-left: 4px solid #f59e0b; 
                    padding: 1rem; 
                    border-radius: 8px; 
                    margin-bottom: 2rem; 
                } 
            </style>  
        </head>  
        <body>  
            <div class="container">  
                <div class="header">  
                    <h1>Label Encoder Features</h1>  
                    <p>EcoPackAI ML Model Feature Categories</p>  
        """  
          
        if ml_manager.label_encoders is None:  
            html += '<span class="status fallback">FALLBACK MODE</span>'  
            features = {  
                'material': ['plastic', 'cardboard', 'glass', 'metal', 'paper', 'aluminium'],  
                'parent_material': ['plastic', 'glass', 'metal', 'paper-or-cardboard'],  
                'shape': ['bottle', 'box', 'bag', 'can', 'jar', 'pouch', 'tray', 'tube'],  
                'strength': ['Low', 'Medium', 'High', 'Very High'],  
                'recycling': ['Recyclable', 'Not Recyclable', 'Compost'],  
                'food_group': ['fruit-juices', 'biscuits-and-cakes', 'dairy-desserts']  
            }  
        else:  
            html += '<span class="status loaded">✅ LOADED FROM ENCODERS</span>'  
            # ONLY extract the 6 fields used by models 
            features = {} 
            for col in REQUIRED_FIELDS: 
                if col in ml_manager.label_encoders: 
                    features[col] = sorted(list(ml_manager.label_encoders[col].classes_)) 
          
        html += f"""  
                </div>  
                <div class="content">  
                    <div class="info-box">  
                        <strong>ℹ️ Model Features:</strong> Showing ONLY the 6 categorical fields used by cost_features.txt and co2_features.txt 
                        <br><strong>Fields:</strong> {', '.join(REQUIRED_FIELDS)} 
                    </div>  
        """   
        # Corrected the string concatenation error from the user provided code which had >'
        
        for name, values in features.items(): 
            html += f""" 
            <div class="feature-card"> 
                <div class="feature-title"> 
                    {name.replace('_', ' ').title()} 
                    <span class="count">{len(values)} items</span> 
                </div> 
                <div class="feature-values"> 
            """ 
            for val in values: 
                html += f'<span class="value-badge">{val}</span>' 
            html += """ 
                </div> 
            </div> 
            """ 
          
        html += """  
                </div>  
            </div>  
        </body>  
        </html>  
        """  
        return html, 200  
    except Exception as e:  
        return f"<html><body><h1>Error: {str(e)}</h1></body></html>", 500  

_initialized = False

def initialize_app():
    """
    Initialize database and ML models once per worker
    Thread-safe initialization for Gunicorn
    """
    global _initialized
    
    if _initialized:
        return
    
    _initialized = True
    
    print("\n" + "="*60)
    print("   Initializing EcoPackAI Server")
    print("="*60)
    
    # Initialize database
    print("\n[1/3] Initializing database...")
    init_db()
    
    # Load ML models
    print("\n[2/3] Loading ML models...")
    models_loaded = ml_manager.load_models()
    
    if models_loaded:
        print("✓ ML models loaded successfully")
    else:
        print("    WARNING: Running in DEMO mode (models not loaded)")
        print("    Predictions will use fallback calculations")
    
    print("\n[3/3] Server initialization complete")
    print("="*60 + "\n")


# ============================================================================
# CALL INITIALIZATION (Safe for Gunicorn with --preload)
# ============================================================================
# This runs at import time, but the global flag prevents duplicates
# With --preload, models load once before worker fork
initialize_app()


# ============================================================================
# MAIN - LOCAL DEVELOPMENT ONLY
# ============================================================================
if __name__ == '__main__':
    """Run Flask development server (local only)"""
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=True
    )
