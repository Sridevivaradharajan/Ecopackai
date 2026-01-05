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
    def __init__(self): 
        self.cost_model = None 
        self.co2_model = None 
        self.label_encoders = None 
        self.cost_features = None 
        self.co2_features = None 
         
        # Exact mappings from notebook 
        self.strength_map = { 
            'Low': 1.0, 'Medium': 2.2, 'High': 3.8, 'Very High': 5.5 
        } 
         
        self.material_cost_map = { 
            'Plastic': 1.1, 'Paper': 0.75, 'Glass': 2.3, 'Metal': 2.8, 
            'Cardboard': 0.65, 'Wood': 1.4, 'Composite': 1.9, 'Aluminum': 3.2, 
            'PE': 1.05, 'PP': 1.1, 'PET': 1.25, 'HDPE': 1.15, 'LDPE': 1.0, 
            # Lowercase variants 
            'plastic': 1.1, 'paper': 0.75, 'glass': 2.3, 'metal': 2.8, 
            'cardboard': 0.65, 'aluminium': 3.2, 'aluminum': 3.2, 
            'steel': 2.8, 'plastic 7': 1.1, 'unknown': 1.3 
        } 
         
        self.shape_complexity = { 
            'Box': 1.0, 'Bag': 0.75, 'Bottle': 1.2, 'Can': 1.15, 
            'Jar': 1.3, 'Pouch': 0.85, 'Tray': 1.05, 'Tube': 1.25, 
            'Container': 1.1, 'Wrapper': 0.8, 
            # Lowercase variants 
            'box': 1.0, 'bag': 0.75, 'bottle': 1.2, 'can': 1.15, 
            'jar': 1.3, 'pouch': 0.85, 'tray': 1.05, 'tube': 1.25, 
            'container': 1.1, 'wrapper': 0.8, 'unknown': 1.0 
        } 
         
    def load_models(self): 
        """Load all required model files""" 
        try: 
            print("Loading ML models...") 
             
            required_files = { 
                'cost_model': 'models/final_cost_model.pkl', 
                'co2_model': 'models/final_co2_model.pkl', 
                'encoders': 'models/label_encoders.pkl', 
                'cost_features': 'models/cost_features.txt', 
                'co2_features': 'models/co2_features.txt' 
            } 
             
            missing_files = [f for f in required_files.values() if not os.path.exists(f)] 
            if missing_files: 
                print(f"⚠️ Missing model files: {missing_files}") 
                print("⚠️ Running in DEMO mode") 
                return False 
             
            self.cost_model = joblib.load(required_files['cost_model']) 
            self.co2_model = joblib.load(required_files['co2_model']) 
            self.label_encoders = joblib.load(required_files['encoders']) 
             
            # Load feature lists - MAINTAINS ORDER 
            with open(required_files['cost_features'], 'r') as f: 
                self.cost_features = [line.strip() for line in f if line.strip()] 
             
            with open(required_files['co2_features'], 'r') as f: 
                self.co2_features = [line.strip() for line in f if line.strip()] 
             
            print(f"✓ Models loaded successfully") 
            print(f"  Cost features: {len(self.cost_features)}") 
            print(f"  CO2 features: {len(self.co2_features)}") 
             
            # VALIDATION: Test feature generation 
            return self._validate_feature_pipeline() 
             
        except Exception as e: 
            print(f"✗ Error loading models: {e}") 
            print(f"✗ Traceback: {traceback.format_exc()}") 
            return False 
     
    def _validate_feature_pipeline(self): 
        """Validate that all features can be generated""" 
        try: 
            print("\n[Validating Feature Pipeline]") 
             
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
                'food_group': 'fruit-juices', 
                'categories_tags': 'fruit-juices', 
                'countries_tags': 'france' 
            } 
             
            features = self.engineer_features(test_input) 
             
            # Check cost features 
            missing_cost = [f for f in self.cost_features if f not in features] 
            if missing_cost: 
                print(f"❌ Missing cost features ({len(missing_cost)}):") 
                for feat in missing_cost[:10]:  # Show first 10 
                    print(f"   - {feat}") 
                return False 
             
            # Check for zero/null values in critical features 
            zero_features = [f for f in self.cost_features if features.get(f, 0) == 0 and 'encoded' in f] 
            if len(zero_features) > 3: 
                print(f"⚠️ WARNING: {len(zero_features)} encoded features are zero (possible encoding issues)") 
             
            # Check CO2 features 
            missing_co2 = [f for f in self.co2_features if f not in features] 
            if missing_co2: 
                print(f"❌ Missing CO2 features ({len(missing_co2)}):") 
                for feat in missing_co2[:10]: 
                    print(f"   - {feat}") 
                return False 
             
            print(f"   Validation passed:") 
            print(f"   Cost: {len(self.cost_features)} features available") 
            print(f"   CO2: {len(self.co2_features)} features available") 
             
            return True 
             
        except Exception as e: 
            print(f"❌ Validation failed: {e}") 
            return False 
     
    def engineer_features(self, product_dict): 
        """ 
        COMPLETE FEATURE ENGINEERING - Matches notebook exactly 
         
        Generates ALL features needed by both Cost and CO2 models 
        Returns dict with ALL 16+ unique features 
        """ 
        features = {} 
         
        # ================================================================ 
        # STEP 1: RAW NUMERIC FEATURES (Base inputs) 
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
         
        # Material cost factor 
        material = product_dict.get('material', 'plastic') 
        features['material_cost_factor'] = self.material_cost_map.get( 
            material,  
            self.material_cost_map.get(material.lower(), 1.3) 
        ) 
         
        # Shape complexity 
        shape = product_dict.get('shape', 'bottle') 
        features['shape_complexity'] = self.shape_complexity.get( 
            shape, 
            self.shape_complexity.get(shape.lower(), 1.0) 
        ) 
         
        # ================================================================ 
        # STEP 3: CATEGORICAL ENCODINGS (Only what models actually use) 
        # ================================================================ 
        # ✅ ONLY encode features that are ACTUALLY in cost_features.txt / co2_features.txt 
        # ❌ REMOVED: categories_tags, countries_tags (not used by models) 
         
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
        features['weight_sqrt'] = np.sqrt(weight)  # ✅ ADDED - Missing for CO2 
         
        # ================================================================ 
        # STEP 5: INTERACTION FEATURES 
        # ================================================================ 
         
        capacity = features['weight_capacity'] 
        material_enc = features['material_encoded'] 
        parent_mat_enc = features['parent_material_encoded'] 
        shape_enc = features['shape_encoded'] 
         
        # Weight × Capacity 
        features['capacity_weight_ratio'] = capacity / (weight + 0.01) 
        features['capacity_weight_prod'] = capacity * weight 
         
        # Material × Weight 
        features['material_weight'] = material_enc * weight 
        features['material_weight_sq'] = material_enc * (weight ** 2)  # ✅ ADDED - Missing for CO2 
         
        # Parent Material × Weight 
        features['parent_mat_weight'] = parent_mat_enc * weight  # ✅ ADDED - Missing for CO2 
         
        # Shape × Weight 
        features['shape_weight'] = shape_enc * weight  # ✅ ADDED - Missing for CO2 
         
        # ================================================================ 
        # STEP 6: COST-SPECIFIC DERIVED FEATURES 
        # ================================================================ 
         
        product_qty = features['product_quantity'] 
        features['packaging_ratio'] = weight / (product_qty + 1) 
         
        recyc_pct = features['recyclability_percent'] 
        features['recyclability_score'] = recyc_pct / 100 
        features['non_recyclable_penalty'] = 100 - recyc_pct 
         
        # ================================================================ 
        # VALIDATION: Ensure all critical features exist 
        # ================================================================ 
        required_features = [ 
            'product_quantity', 'weight_measured', 'weight_capacity', 'recyclability_percent', 
            'number_of_units', 'strength_num', 'material_cost_factor', 'shape_complexity', 
            'material_encoded', 'parent_material_encoded', 'shape_encoded', 'strength_encoded', 
            'recycling_encoded', 'food_group_encoded', 'weight_squared', 'weight_log',  
            'weight_sqrt', 'capacity_weight_ratio', 'capacity_weight_prod', 'material_weight', 
            'material_weight_sq', 'parent_mat_weight', 'shape_weight', 'packaging_ratio', 
            'recyclability_score', 'non_recyclable_penalty' 
        ] 
         
        missing = [f for f in required_features if f not in features] 
        if missing: 
            print(f"⚠️ WARNING: Missing features: {missing}") 
         
        return features 
     
    def _infer_parent_material(self, material): 
        """Infer parent material from material name (matches notebook logic)""" 
        material_lower = str(material).lower() 
         
        if material_lower in ['aluminium', 'aluminum', 'metal', 'steel']: 
            return 'metal' 
        elif material_lower in ['cardboard', 'paper']: 
            return 'paper-or-cardboard' 
        elif material_lower == 'glass': 
            return 'glass' 
        elif 'plastic' in material_lower: 
            return 'plastic' 
        else: 
            return 'plastic'  # Default 
     
    def _encode_categorical(self, value, column_name): 
        """ 
        Encode categorical values using trained LabelEncoders 
        CRITICAL: Handles missing values and provides fallbacks 
        """ 
        if column_name not in self.label_encoders: 
            print(f"⚠️ No encoder for {column_name}, returning 0") 
            return 0 
         
        encoder = self.label_encoders[column_name] 
        available_classes = list(encoder.classes_) 
         
        # Try exact match 
        if str(value) in available_classes: 
            return int(encoder.transform([str(value)])[0]) 
         
        # Try lowercase match 
        value_lower = str(value).lower().strip() 
         
        # Column-specific mappings (from notebook standardization) 
        mappings = { 
            'material': { 
                'cardboard': 'cardboard', 'plastic': 'plastic', 'glass': 'glass', 
                'metal': 'metal', 'aluminium': 'aluminium', 'aluminum': 'aluminium', 
                'paper': 'paper', 'steel': 'steel', 'plastic 7': 'plastic 7', 
                'pe': 'plastic', 'pp': 'plastic', 'pet': 'plastic', 
                'hdpe': 'plastic', 'ldpe': 'plastic', 'unknown': 'unknown' 
            }, 
            'parent_material': { 
                'glass': 'glass', 'metal': 'metal', 'plastic': 'plastic', 
                'paper': 'paper', 'cardboard': 'paper-or-cardboard', 
                'aluminium': 'metal', 'aluminum': 'metal', 'steel': 'metal' 
            }, 
            'shape': { 
                'bottle': 'bottle', 'box': 'box', 'bag': 'bag', 'can': 'can', 
                'jar': 'jar', 'pouch': 'pouch', 'tray': 'tray', 'tube': 'tube', 
                'container': 'container', 'wrapper': 'wrapper', 'lid': 'lid', 
                'cap': 'cap', 'film': 'film', 'seal': 'seal', 'label': 'label', 
                'sleeve': 'sleeve', 'pot': 'pot', 'wrap': 'wrap' 
            }, 
            'strength': { 
                'low': 'Low', 'medium': 'Medium', 'high': 'High', 'very high': 'Very High' 
            }, 
            'recycling': { 
                'recyclable': 'Recyclable', 
                'recycle': 'Recyclable', 
                'not recyclable': 'Not Recyclable', 
                'compost': 'Compost', 
                'reusable': 'Reusable', 
                'deposit return': 'Deposit Return', 
                'return to store': 'Return to Store' 
            }, 
            'food_group': { 
                'fruit-juices': 'fruit-juices', 
                'biscuits-and-cakes': 'biscuits-and-cakes', 
                'dairy-desserts': 'dairy-desserts', 
                'bread': 'bread', 
                'cheese': 'cheese' 
            } 
        } 
         
        # Apply mapping 
        if column_name in mappings: 
            mapped = mappings[column_name].get(value_lower) 
            if mapped and mapped in available_classes: 
                return int(encoder.transform([mapped])[0]) 
         
        # Try case-insensitive match 
        for cls in available_classes: 
            if str(cls).lower() == value_lower: 
                return int(encoder.transform([str(cls)])[0]) 
         
        # Fallback to most common value 
        fallback_map = { 
            'material': 'plastic', 
            'parent_material': 'plastic', 
            'shape': 'bottle', 
            'strength': 'Medium', 
            'recycling': 'Recyclable', 
            'food_group': 'fruit-juices', 
            'categories_tags': available_classes[0] if available_classes else 'unknown',  # Use first encoded class 
            'countries_tags': 'france' 
        } 
         
        fallback = fallback_map.get(column_name, available_classes[0] if available_classes else 'Unknown') 
        if fallback in available_classes: 
            print(f"⚠️ Using fallback '{fallback}' for {column_name}='{value}'") 
            return int(encoder.transform([fallback])[0]) 
         
        print(f"⚠️ Could not encode {column_name}='{value}', returning 0") 
        return 0 
     
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
                 
                print("⚠️ Using fallback prediction (models not loaded)") 
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
                    # Use intelligent defaults instead of 0 
                    if 'encoded' in feat: 
                        X_cost.append(0)  # Encoded features default to 0 
                    elif 'ratio' in feat or 'percent' in feat: 
                        X_cost.append(1.0)  # Ratios default to 1 
                    else: 
                        X_cost.append(0)  # Numeric features default to 0 
             
            if missing_cost_features: 
                print(f"⚠️ CRITICAL: Missing {len(missing_cost_features)} cost features:") 
                for feat in missing_cost_features[:5]: 
                    print(f"   - {feat}") 
                print(f"\n❌ This will cause prediction errors!") 
             
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
                print(f"⚠️ CRITICAL: Missing {len(missing_co2_features)} CO2 features:") 
                for feat in missing_co2_features[:5]: 
                    print(f"   - {feat}") 
                print(f"\n❌ This will cause prediction errors!") 
             
            X_co2 = np.array(X_co2).reshape(1, -1) 
             
            # CRITICAL: Model was trained on log-transformed CO2 values 
            co2_pred_log = self.co2_model.predict(X_co2)[0] 
            co2_pred = float(np.expm1(co2_pred_log))  # Inverse of log1p 
             
            # ================================================================ 
            # VALIDATION: Check if predictions are reasonable 
            # ================================================================

            # Calculate expected ranges based on weight (adjust multipliers from your domain knowledge)
            weight = features['weight_measured']
            product_qty = features['product_quantity']

            # Expected cost: roughly ₹0.5-5 per gram depending on material
            # For 1000 units of 50g packaging = 50,000g total = ₹25,000-250,000 potential range
            expected_max_cost = (weight * product_qty * 5)  # ₹5 per gram is very expensive material

            # Expected CO2: roughly 0.01-0.5 per gram
            expected_max_co2 = (weight * product_qty * 0.5)

            # Validate cost
            if cost_pred < 0:
                print(f"ERROR: Negative cost: ₹{cost_pred:.2f}")
            elif cost_pred > expected_max_cost:
                print(f"ERROR: Cost exceeds physical limits: ₹{cost_pred:.2f} (max expected: ₹{expected_max_cost:.2f})")
                print(f"Input: {weight}g × {product_qty} units = {weight * product_qty}g total")
            elif cost_pred > expected_max_cost * 0.5:
                print(f"High cost: ₹{cost_pred:.2f} (50%+ of theoretical maximum)")

            # Validate CO2
            if co2_pred < 0:
                print(f"ERROR: Negative CO2: {co2_pred:.2f}")
            elif co2_pred > expected_max_co2:
                print(f"ERROR: CO2 exceeds physical limits: {co2_pred:.2f} (max expected: {expected_max_co2:.2f})")
            elif co2_pred > expected_max_co2 * 0.5:
                print(f"High CO2: {co2_pred:.2f}")

            print(f"✓ Prediction: ₹{cost_pred:.2f} | CO2: {co2_pred:.2f}")
             
            print(f"✓ Prediction SUCCESS:") 
            print(f"  Cost: ₹{cost_pred:.2f}") 
            print(f"  CO2: {co2_pred:.2f}") 
            print(f"  Features generated: {len(features)}") 
            print(f"  Missing cost features: {len(missing_cost_features)}") 
            print(f"  Missing CO2 features: {len(missing_co2_features)}") 
             
            return cost_pred, co2_pred, features 
             
        except Exception as e: 
            print(f"❌ Prediction error: {e}") 
            print(f"Traceback: {traceback.format_exc()}") 
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
                    'material': alt['material'],  
                    'shape': alt['shape'],  
                    'strength': alt['strength'],  
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
            SELECT id, cost_savings, co2_reduction, created_at  
            FROM recommendations_history  
            WHERE user_id = %s  
            ORDER BY created_at DESC  
            LIMIT 10  
        ''', (current_user_id,))  
        history = cur.fetchall()  
          
        cur.close()  
        conn.close()  
          
        return jsonify({  
            'analytics': dict(analytics) if analytics else {},  
            'recent': [dict(h) for h in history]  
        }), 200  
    except Exception as e:  
        return jsonify({'error': str(e)}), 500  
  
# ============================================================================  
# CORRECTED: /api/materials - Returns ONLY fields models actually use 
# ============================================================================  
  
@app.route('/api/materials', methods=['GET'])  
def get_materials():  
    """  
    Return actual categories from label encoders  
    ✅ ONLY return the 6 fields that models actually use 
    ❌ REMOVED: categories_tags, countries_tags (not in feature files) 
    """  
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
