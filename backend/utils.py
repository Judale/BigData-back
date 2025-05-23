import jwt
from datetime import datetime, timedelta
from flask import request, jsonify
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash


# 🔐 Clé secrète à sécuriser (utiliser dotenv en prod)
SECRET_KEY = "super-secret-key"

# 🔐 Générer un token
def generate_token(user_id):
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

# 🔐 Middleware de protection
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            print("🚫 Aucun header Authorization trouvé")
            return jsonify({"error": "Token manquant"}), 401
        if not auth_header.startswith("Bearer "):
            print("🚫 Mauvais format d'Authorization")
            return jsonify({"error": "Format de token invalide"}), 401

        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            print("🚫 Token expiré")
            return jsonify({"error": "Token expiré"}), 401
        except jwt.InvalidTokenError:
            print("🚫 Token invalide")
            return jsonify({"error": "Token invalide"}), 401

        print(f"✅ Token valide pour user_id={payload['user_id']}")
        return f(user_id=payload["user_id"], *args, **kwargs)
    return decorated

def hash_password(password: str) -> str:
    return generate_password_hash(password)

def verify_password(hash_: str, password: str) -> bool:
    return check_password_hash(hash_, password)

# ─────────────────────────────
# Score helpers
# ─────────────────────────────
_DEC_PER_SEC = {"easy": 3, "medium": 6, "hard": 12}

def compute_score(elapsed_sec: float, difficulty: str) -> int:
    """Retourne un score entre 0 et 100."""
    dec = _DEC_PER_SEC[difficulty]
    return max(0, 100 - int(dec * elapsed_sec))