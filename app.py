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
    resources={ 
        r"/*": { 
            "origins": [ 
                "http://localhost:3000", 
                "http://localhost:5000", 
                "http://127.0.0.1:5000",
                "https://ecopack-yngt.onrender.com"
            ] 
        } 
    } 
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
 
class MLModelManager:
    """
    Complete ML model manager with robust feature engineering and validation
    
    Features:
    - Exact feature order preservation from training
    - Robust label encoding with fuzzy matching
    - Comprehensive validation pipeline
    - Detailed error reporting
    """
    
    def __init__(self):
        self.cost_model = None
        self.co2_model = None
        self.label_encoders = None
        self.cost_features = None
        self.co2_features = None
        
        # Helper mappings (from notebook)
        self.strength_map = {
            'Low': 1.0, 'Medium': 2.2, 'High': 3.8, 'Very High': 5.5
        }
        
        self.material_cost_map = {
            'plastic': 1.1, 'paper': 0.75, 'glass': 2.3, 'metal': 2.8,
            'cardboard': 0.65, 'wood': 1.4, 'composite': 1.9,
            'aluminium': 3.2, 'aluminum': 3.2, 'steel': 2.8,
            'pe': 1.05, 'pp': 1.1, 'pet': 1.25, 'hdpe': 1.15, 'ldpe': 1.0,
            'plastic 7': 1.1, 'unknown': 1.3
        }
        
        self.shape_complexity = {
            'box': 1.0, 'bag': 0.75, 'bottle': 1.2, 'can': 1.15,
            'jar': 1.3, 'pouch': 0.85, 'tray': 1.05, 'tube': 1.25,
            'container': 1.1, 'wrapper': 0.8, 'packaging': 1.0,
            'lid': 0.6, 'cap': 0.6, 'seal': 0.5, 'film': 0.7,
            'unknown': 1.0
        }
    
    def load_models(self):
        """Load all model files with validation"""
        try:
            print("="*80)
            print("LOADING ML MODELS")
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
            
            # Load feature lists (MAINTAINS ORDER)
            with open(required_files['cost_features'], 'r') as f:
                self.cost_features = [line.strip() for line in f if line.strip()]
            
            with open(required_files['co2_features'], 'r') as f:
                self.co2_features = [line.strip() for line in f if line.strip()]
            
            print(f"[OK] Cost model loaded: {len(self.cost_features)} features")
            print(f"[OK] CO2 model loaded: {len(self.co2_features)} features")
            print(f"[OK] Label encoders: {len(self.label_encoders)} categories")
            
            # CRITICAL: Validate pipeline
            return self._validate_feature_pipeline()
            
        except Exception as e:
            print(f"[ERROR] Error loading models: {e}")
            traceback.print_exc()
            return False
    
    def _validate_feature_pipeline(self):
        """
        Comprehensive validation of feature generation pipeline
        
        Tests:
        1. All features can be generated
        2. Feature order matches training
        3. Encoders work correctly
        4. Predictions are reasonable
        """
        try:
            print("\n" + "="*80)
            print("VALIDATING FEATURE PIPELINE")
            print("="*80)
            
            # Test input (typical product)
            test_input = {
                'product_quantity': 500,
                'weight_measured': 50,
                'weight_capacity': 600,
                'number_of_units': 1,
                'recyclability_percent': 70,
                'material': 'plastic',
                'parent_material': 'plastic',
                'shape': 'bottle',
                'strength': 'Medium',
                'recycling': 'Recyclable',
                'food_group': 'fruit-juices'
            }
            
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
            
            # Validate feature order
            print("\n[Test 4] Feature Order Verification")
            X_cost = np.array([features[f] for f in self.cost_features]).reshape(1, -1)
            X_co2 = np.array([features[f] for f in self.co2_features]).reshape(1, -1)
            print(f"  [OK] Cost array shape: {X_cost.shape}")
            print(f"  [OK] CO2 array shape: {X_co2.shape}")
            
            # Test predictions
            print("\n[Test 5] Prediction Test")
            cost_pred = float(self.cost_model.predict(X_cost)[0])
            co2_log = self.co2_model.predict(X_co2)[0]
            co2_pred = float(np.expm1(co2_log))
            
            print(f"  [RESULT] Cost: Rs.{cost_pred:.2f}")
            print(f"  [RESULT] CO2: {co2_pred:.2f}")
            
            # Sanity checks
            if cost_pred <= 0 or cost_pred > 10000:
                print(f"  [WARNING] Unusual cost prediction: Rs.{cost_pred:.2f}")
            if co2_pred <= 0 or co2_pred > 1000:
                print(f"  [WARNING] Unusual CO2 prediction: {co2_pred:.2f}")
            
            print("\n" + "="*80)
            print("VALIDATION PASSED")
            print("="*80)
            return True
            
        except Exception as e:
            print(f"\n[CRITICAL] Validation failed: {e}")
            traceback.print_exc()
            return False
    
    def _encode_categorical(self, value, column_name):
        """
        Encode categorical values with ROBUST fallback handling
        
        Priority:
        1. Exact match (case-sensitive)
        2. Case-insensitive match
        3. Fuzzy match (e.g., "aluminum" -> "aluminium")
        4. Domain-specific fallback
        5. Most common value (last resort)
        """
        if column_name not in self.label_encoders:
            print(f"[WARNING] No encoder for {column_name}, returning 0")
            return 0
        
        encoder = self.label_encoders[column_name]
        available_classes = list(encoder.classes_)
        
        # Convert to string
        value_str = str(value).strip()
        
        # STEP 1: Exact Match (Case-Sensitive)
        if value_str in available_classes:
            return int(encoder.transform([value_str])[0])
        
        # STEP 2: Case-Insensitive Match
        value_lower = value_str.lower()
        for cls in available_classes:
            if str(cls).lower() == value_lower:
                return int(encoder.transform([str(cls)])[0])
        
        # STEP 3: Fuzzy Mappings (Domain-Specific)
        fuzzy_mappings = {
            'material': {
                'aluminum': 'aluminium',
                'carton': 'cardboard',
                'paperboard': 'cardboard',
                'tin': 'metal',
                'iron': 'metal',
                'pe': 'plastic',
                'pp': 'plastic',
                'pet': 'plastic',
                'hdpe': 'plastic',
                'ldpe': 'plastic',
                'polypropylene': 'plastic',
                'polyethylene': 'plastic',
                'pvc': 'plastic 7'
            },
            'parent_material': {
                'aluminum': 'metal',
                'aluminium': 'metal',
                'steel': 'metal',
                'tin': 'metal',
                'iron': 'metal',
                'cardboard': 'paper-or-cardboard',
                'paperboard': 'paper-or-cardboard',
                'paper': 'paper-or-cardboard',
                'carton': 'paper-or-cardboard'
            },
            'shape': {
                'package': 'packaging',
                'pack': 'packaging',
                'wrapping': 'wrapper',
                'covering': 'wrapper',
                'top': 'cap',
                'cover': 'lid',
                'basket': 'container',
                'dish': 'tray'
            },
            'strength': {
                'weak': 'Low',
                'low': 'Low',
                'strong': 'High',
                'high': 'High',
                'normal': 'Medium',
                'medium': 'Medium',
                'average': 'Medium',
                'very strong': 'Very High',
                'very high': 'Very High',
                'extra strong': 'Very High'
            },
            'recycling': {
                'yes': 'Recyclable',
                'no': 'Not Recyclable',
                'biodegradable': 'Compost',
                'compostable': 'Compost',
                'returnable': 'Return to Store',
                'deposit': 'Deposit Return',
                'reuse': 'Reusable'
            },
            'food_group': {
                'juice': 'fruit-juices',
                'juices': 'fruit-juices',
                'biscuit': 'biscuits-and-cakes',
                'cake': 'biscuits-and-cakes',
                'dessert': 'dairy-desserts',
                'meat': 'meat-other-than-poultry',
                'chicken': 'poultry',
                'fish': 'fish-and-seafood',
                'vegetable': 'vegetables',
                'fruit': 'fruits'
            }
        }
        
        if column_name in fuzzy_mappings:
            mapped = fuzzy_mappings[column_name].get(value_lower)
            if mapped and mapped in available_classes:
                print(f"[INFO] Fuzzy match: {column_name}='{value}' -> '{mapped}'")
                return int(encoder.transform([mapped])[0])
        
        # STEP 4: Domain Fallbacks (most common safe values)
        fallback_map = {
            'material': 'plastic',
            'parent_material': 'plastic',
            'shape': 'bottle',
            'strength': 'Medium',
            'recycling': 'Recyclable',
            'food_group': 'fruit-juices'
        }
        
        fallback = fallback_map.get(column_name)
        if fallback and fallback in available_classes:
            print(f"[WARNING] Using fallback for {column_name}='{value}' -> '{fallback}'")
            return int(encoder.transform([fallback])[0])
        
        # STEP 5: First available class (last resort)
        if available_classes:
            print(f"[ERROR] Could not encode {column_name}='{value}', using first class: '{available_classes[0]}'")
            return int(encoder.transform([available_classes[0]])[0])
        
        print(f"[CRITICAL] No classes available for {column_name}, returning 0")
        return 0
    
    def _infer_parent_material(self, material):
        """Infer parent material from material name"""
        material_lower = str(material).lower()
        
        if material_lower in ['aluminium', 'aluminum', 'metal', 'steel', 'tin', 'iron']:
            return 'metal'
        elif material_lower in ['cardboard', 'paper', 'paperboard', 'carton']:
            return 'paper-or-cardboard'
        elif material_lower == 'glass':
            return 'glass'
        elif 'plastic' in material_lower or material_lower in ['pe', 'pp', 'pet', 'hdpe', 'ldpe', 'pvc']:
            return 'plastic'
        else:
            return 'unknown'
    
    def engineer_features(self, product_dict):
        """
        COMPLETE FEATURE ENGINEERING - Matches notebook exactly
        
        Generates ALL features needed by both Cost and CO2 models
        Returns dict with ALL required features in correct format
        """
        features = {}
        
        # ================================================================
        # STEP 1: RAW NUMERIC FEATURES
        # ================================================================
        features['product_quantity'] = float(product_dict.get('product_quantity', 500))
        features['weight_measured'] = float(product_dict.get('weight_measured', 50))
        features['weight_capacity'] = float(product_dict.get('weight_capacity', 600))
        features['recyclability_percent'] = float(product_dict.get('recyclability_percent', 70))
        features['number_of_units'] = int(product_dict.get('number_of_units', 1))
        
        # ================================================================
        # STEP 2: HELPER FEATURES (Must be calculated FIRST)
        # ================================================================
        
        # Strength numeric mapping
        strength = product_dict.get('strength', 'Medium')
        features['strength_num'] = self.strength_map.get(strength, 2.2)
        
        # Material cost factor (case-insensitive lookup)
        material = product_dict.get('material', 'plastic')
        material_lower = material.lower()
        features['material_cost_factor'] = self.material_cost_map.get(
            material_lower,
            self.material_cost_map.get(material, 1.3)
        )
        
        # Shape complexity (case-insensitive lookup)
        shape = product_dict.get('shape', 'bottle')
        shape_lower = shape.lower()
        features['shape_complexity'] = self.shape_complexity.get(
            shape_lower,
            self.shape_complexity.get(shape, 1.0)
        )
        
        # ================================================================
        # STEP 3: CATEGORICAL ENCODINGS
        # ================================================================
        
        categorical_fields = {
            'food_group': product_dict.get('food_group', 'fruit-juices'),
            'material': material,
            'parent_material': product_dict.get('parent_material') or self._infer_parent_material(material),
            'recycling': product_dict.get('recycling', 'Recyclable'),
            'shape': shape,
            'strength': strength
        }
        
        for field_name, value in categorical_fields.items():
            encoded_name = f'{field_name}_encoded'
            features[encoded_name] = self._encode_categorical(value, field_name)
        
        # ================================================================
        # STEP 4: POLYNOMIAL FEATURES (weight transformations)
        # ================================================================
        weight = features['weight_measured']
        
        features['weight_squared'] = weight ** 2
        features['weight_log'] = np.log1p(weight)
        features['weight_sqrt'] = np.sqrt(weight)
        
        # ================================================================
        # STEP 5: INTERACTION FEATURES
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
        # STEP 6: COST-SPECIFIC DERIVED FEATURES
        # ================================================================
        
        product_qty = features['product_quantity']
        features['packaging_ratio'] = weight / (product_qty + 1)
        
        recyc_pct = features['recyclability_percent']
        features['recyclability_score'] = recyc_pct / 100
        features['non_recyclable_penalty'] = 100 - recyc_pct
        
        return features
    
    def predict(self, product_dict):
        """
        Generate predictions for cost and CO2
        
        CRITICAL: Features are extracted in EXACT order from cost_features.txt / co2_features.txt
        
        Returns:
            cost_pred (float): Predicted packaging cost
            co2_pred (float): Predicted CO2 impact
            features (dict): All engineered features
        """
        try:
            if self.cost_model is None or self.co2_model is None:
                # Fallback prediction
                weight = float(product_dict.get('weight_measured', 50))
                material = product_dict.get('material', 'plastic').lower()
                
                base_cost = self.material_cost_map.get(material, 1.5)
                cost_pred = base_cost * weight * 0.02
                co2_pred = weight * 0.5
                
                print("[WARNING] Using fallback prediction (models not loaded)")
                return cost_pred, co2_pred, {}
            
            # ================================================================
            # STEP 1: Generate ALL features
            # ================================================================
            features = self.engineer_features(product_dict)
            
            # ================================================================
            # STEP 2: COST PREDICTION - Maintain exact feature order
            # ================================================================
            X_cost = []
            missing_cost_features = []
            
            for feat in self.cost_features:
                if feat in features:
                    X_cost.append(features[feat])
                else:
                    missing_cost_features.append(feat)
                    # Intelligent defaults
                    if 'encoded' in feat:
                        X_cost.append(0)
                    elif 'ratio' in feat or 'percent' in feat:
                        X_cost.append(1.0)
                    else:
                        X_cost.append(0)
            
            if missing_cost_features:
                print(f"[CRITICAL] Missing {len(missing_cost_features)} cost features:")
                for feat in missing_cost_features[:5]:
                    print(f"  - {feat}")
                print("[ERROR] This will cause prediction errors!")
            
            X_cost = np.array(X_cost).reshape(1, -1)
            cost_pred = float(self.cost_model.predict(X_cost)[0])
            
            # ================================================================
            # STEP 3: CO2 PREDICTION - Maintain exact feature order
            # ================================================================
            X_co2 = []
            missing_co2_features = []
            
            for feat in self.co2_features:
                if feat in features:
                    X_co2.append(features[feat])
                else:
                    missing_co2_features.append(feat)
                    if 'encoded' in feat:
                        X_co2.append(0)
                    elif 'ratio' in feat or 'percent' in feat:
                        X_co2.append(1.0)
                    else:
                        X_co2.append(0)
            
            if missing_co2_features:
                print(f"[CRITICAL] Missing {len(missing_co2_features)} CO2 features:")
                for feat in missing_co2_features[:5]:
                    print(f"  - {feat}")
                print("[ERROR] This will cause prediction errors!")
            
            X_co2 = np.array(X_co2).reshape(1, -1)
            
            # CRITICAL: Model was trained on log-transformed CO2 values
            co2_pred_log = self.co2_model.predict(X_co2)[0]
            co2_pred = float(np.expm1(co2_pred_log))
            
            # ================================================================
            # VALIDATION: Check if predictions are reasonable
            # ================================================================
            weight = features['weight_measured']
            product_qty = features['product_quantity']
            
            # Expected ranges
            expected_max_cost = weight * product_qty * 5
            expected_max_co2 = weight * product_qty * 0.5
            
            # Validate cost
            if cost_pred < 0:
                print(f"[ERROR] Negative cost: Rs.{cost_pred:.2f}")
            elif cost_pred > expected_max_cost:
                print(f"[ERROR] Cost exceeds limits: Rs.{cost_pred:.2f} (max: Rs.{expected_max_cost:.2f})")
            elif cost_pred > expected_max_cost * 0.5:
                print(f"[WARNING] High cost: Rs.{cost_pred:.2f}")
            
            # Validate CO2
            if co2_pred < 0:
                print(f"[ERROR] Negative CO2: {co2_pred:.2f}")
            elif co2_pred > expected_max_co2:
                print(f"[ERROR] CO2 exceeds limits: {co2_pred:.2f} (max: {expected_max_co2:.2f})")
            elif co2_pred > expected_max_co2 * 0.5:
                print(f"[WARNING] High CO2: {co2_pred:.2f}")
            
            print(f"[OK] Prediction: Cost=Rs.{cost_pred:.2f} | CO2={co2_pred:.2f}")
            
            return cost_pred, co2_pred, features
            
        except Exception as e:
            print(f"[ERROR] Prediction error: {e}")
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

def generate_alternatives(product_details, num_alternatives=5):
    """
    Generate alternative packaging options that IMPROVE sustainability
    Prioritizes: Lower CO₂ > Lower Cost > Similar functionality
    """
    alternatives = []
    
    # Get current metrics for comparison
    current_cost, current_co2, _ = ml_manager.predict(product_details)
    
    if not current_cost or not current_co2:
        print("⚠️ Could not predict current packaging metrics")
        return []
    
    # Define eco-friendly materials (ordered by typical sustainability)
    eco_materials = ['paper', 'cardboard', 'glass', 'plastic', 'aluminium', 'metal']
    
    # Define shapes that work with eco materials
    eco_shapes = ['box', 'bag', 'pouch', 'bottle', 'jar', 'can', 'tray', 'tube']
    
    current_material = product_details.get('material', 'plastic')
    current_shape = product_details.get('shape', 'bottle')
    
    print(f"\n[Generating Alternatives]")
    print(f"Current: {current_material} | Cost: ₹{current_cost:.2f} | CO₂: {current_co2:.2f}")
    
    # Generate and evaluate alternatives
    candidates = []
    
    for material in eco_materials:
        if material == current_material:
            continue
            
        for shape in eco_shapes:
            alt = product_details.copy()
            alt['material'] = material
            alt['shape'] = shape
            
            # Infer parent material
            if material in ['aluminium', 'metal', 'steel']:
                alt['parent_material'] = 'metal'
            elif material in ['cardboard', 'paper']:
                alt['parent_material'] = 'paper-or-cardboard'
            elif material == 'glass':
                alt['parent_material'] = 'glass'
            else:
                alt['parent_material'] = 'plastic'
            
            # Adjust strength based on material
            if material in ['glass', 'metal', 'aluminium', 'steel']:
                alt['strength'] = 'High'
            elif material in ['cardboard', 'paper']:
                alt['strength'] = 'Low'
            else:
                alt['strength'] = product_details.get('strength', 'Medium')
            
            # Predict alternative metrics
            alt_cost, alt_co2, _ = ml_manager.predict(alt)
            
            if not alt_cost or not alt_co2:
                continue
            
            # Calculate improvements
            cost_savings = current_cost - alt_cost
            co2_reduction = current_co2 - alt_co2
            
            # Score: Prioritize CO₂ reduction (70%) and cost savings (30%)
            co2_score = (co2_reduction / current_co2) * 100 if current_co2 > 0 else 0
            cost_score = (cost_savings / current_cost) * 100 if current_cost > 0 else 0
            
            overall_score = (co2_score * 0.7) + (cost_score * 0.3)
            
            candidates.append({
                'config': alt,
                'cost': alt_cost,
                'co2': alt_co2,
                'cost_savings': cost_savings,
                'co2_reduction': co2_reduction,
                'overall_score': overall_score,
                'material': material,
                'shape': shape
            })
            
            print(f"  Tested: {material:10s} {shape:8s} | Cost: ₹{alt_cost:7.2f} (Δ {cost_savings:+7.2f}) | CO₂: {alt_co2:6.2f} (Δ {co2_reduction:+6.2f}) | Score: {overall_score:6.2f}")
    
    # Sort by overall score (higher is better)
    candidates.sort(key=lambda x: x['overall_score'], reverse=True)
    
    # Return top alternatives
    result = []
    for candidate in candidates[:num_alternatives]:
        result.append(candidate['config'])
        print(f"✓ Selected: {candidate['material']} {candidate['shape']} (Score: {candidate['overall_score']:.2f})")
    
    if len(result) == 0:
        print("  No suitable alternatives found - returning diverse options")
        # Fallback: return diverse options even if not improvements
        return [product_details.copy() for _ in range(min(num_alternatives, 3))]
    
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
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
            
        email = data.get('email', '').strip()  
        password = data.get('password', '')  
          
        conn = get_db_connection()  
        if not conn: 
            return redirect('/?error=database_connection_failed')
 
        cur = conn.cursor(cursor_factory=RealDictCursor)  
        cur.execute('SELECT * FROM users WHERE email = %s', (email,))  
        user = cur.fetchone()  
          
        if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            cur.close()
            conn.close()
            return redirect('/?error=invalid_credentials')
          
        cur.execute('UPDATE users SET last_login = %s WHERE id = %s', (datetime.datetime.now(), user['id']))  
        conn.commit()  
          
        token = jwt.encode({  
            'user_id': user['id'],  
            'email': user['email'],  
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)  
        }, app.config['JWT_SECRET'], algorithm='HS256')  
          
        cur.close()  
        conn.close()  
    
        response = redirect('/dashboard')
        response.set_cookie(
            'token',
            token,
            max_age=86400,  # 24 hours in seconds
            httponly=True,
            samesite='Lax',
            secure=False,
            path='/'
        )
        return response
    
    except Exception as e:  
        return redirect(f'/?error={str(e)}')

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
         
        product_details = {  
            'product_quantity': float(data.get('product_quantity', 500)),  
            'weight_measured': float(data.get('weight_measured', 50)),  
            'weight_capacity': float(data.get('weight_capacity', 600)),  
            'number_of_units': int(data.get('number_of_units', 1)),  
            'recyclability_percent': float(data.get('recyclability_percent', 70)), 
             
            # Categorical fields (ONLY the 6 that models use) 
            'food_group': data.get('food_group', 'fruit-juices'),  
            'material': data.get('material', 'plastic'),  
            'parent_material': data.get('parent_material', data.get('material', 'plastic')),  
            'shape': data.get('shape', 'bottle'),  
            'strength': data.get('strength', 'Medium'),  
            'recycling': data.get('recycling', 'Recyclable') 
        }  
          
        current_cost, current_co2, _ = ml_manager.predict(product_details)  
        if current_cost is None:  
            return jsonify({'error': 'Prediction failed'}), 500  
          
        alternatives = generate_alternatives(product_details, data.get('number_of_alternatives', 5))  
        recommendations = []  
          
        for alt in alternatives:  
          alt_cost, alt_co2, _ = ml_manager.predict(alt)  
          if alt_cost and alt_co2:  
              cost_savings = current_cost - alt_cost  
              co2_reduction = current_co2 - alt_co2  
              recommendations.append({  
                  # ALL FIELDS from alternative
                  'material': alt['material'],  
                  'parent_material': alt.get('parent_material', ''),
                  'shape': alt['shape'],  
                  'strength': alt['strength'],  
                  'recycling': alt.get('recycling', ''),
                  'food_group': alt.get('food_group', ''),
                  'product_quantity': alt.get('product_quantity', 0),
                  'weight_measured': alt.get('weight_measured', 0),
                  'weight_capacity': alt.get('weight_capacity', 0),
                  'number_of_units': alt.get('number_of_units', 1),
                  'recyclability_percent': alt.get('recyclability_percent', 0),
                  # Predictions
                  'predicted_cost': round(alt_cost, 2),  
                  'predicted_co2': round(alt_co2, 2),  
                  'cost_savings': round(cost_savings, 2),  
                  'co2_reduction': round(co2_reduction, 2),  
                  'improvement_score': calculate_score(cost_savings, co2_reduction)  
              }) 
          
        recommendations.sort(key=lambda x: x['improvement_score'], reverse=True)  
          
        if recommendations:  
            save_recommendation(current_user_id, product_details,   
                              {'cost': current_cost, 'co2': current_co2},   
                              recommendations[:5])  
          
        return jsonify({  
            'current_packaging': {  
                'cost': round(current_cost, 2),  
                'co2': round(current_co2, 2),  
                'co2_label': 'CO₂ Impact Index', 
                'material': product_details['material'], 
                'shape': product_details['shape'], 
                'strength': product_details['strength'], 
                'recyclability_percent': product_details['recyclability_percent'] 
            },  
            'recommendations': recommendations[:5]  
        }), 200 
    except Exception as e:  
        return jsonify({'error': str(e)}), 500 
  
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
                # Extract input data 
                input_data = { 
                    'product_quantity': float(row.get('product_quantity', 0)), 
                    'weight_measured': float(row.get('weight_measured', 0)), 
                    'weight_capacity': float(row.get('weight_capacity', 0)), 
                    'material': str(row.get('material', '')).lower(), 
                    'shape': str(row.get('shape', '')).lower(), 
                    'strength': str(row.get('strength', 'medium')), 
                    'food_group': str(row.get('food_group', '')).lower(), 
                    'recycling': str(row.get('recycling', 'recyclable')), 
                    'number_of_units': int(row.get('number_of_units', 1)), 
                    'recyclability_percent': float(row.get('recyclability_percent', 70)), 
                } 
                 
                # Get predictions using ml_manager 
                cost_pred, co2_pred, _ = ml_manager.predict(input_data) 
 
                # Handle prediction failures 
                if cost_pred is None or co2_pred is None: 
                    results.append({ 
                        'row': idx + 1, 
                        'material': input_data['material'], 
                        'shape': input_data.get('shape', ''), 
                        'strength': input_data.get('strength', ''), 
                        'food_group': input_data.get('food_group', ''), 
                        'recycling': input_data.get('recycling', ''), 
                        'product_quantity': input_data.get('product_quantity', ''), 
                        'weight_measured': input_data.get('weight_measured', ''), 
                        'weight_capacity': input_data.get('weight_capacity', ''), 
                        'number_of_units': input_data.get('number_of_units', ''), 
                        'recyclability_percent': input_data.get('recyclability_percent', ''), 
                        'cost': '-', 
                        'co2': '-', 
                        'status': 'error', 
                        'error': 'Prediction failed' 
                    }) 
                    continue 
 
                # ✅ Return BOTH input data AND predictions 
                results.append({ 
                    'row': idx + 1, 
                    # Input fields (echo back) 
                    'material': input_data['material'], 
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
                    'material': str(row.get('material', '')), 
                    'status': 'error', 
                    'error': str(e) 
                }) 
          
        conn = get_db_connection()  
        if not conn: 
            # If DB fails, we still return the processed results but warn about save failure 
            # Or we can fail the whole request.  
            # Given bulk upload processes but saves history, maybe we should fail? 
            # Or just return results without saving to DB. 
            # Let's return error for now to be consistent. 
            return jsonify({'error': 'Database connection failed, results not saved', 'results': results}), 503 
 
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
