import jwt
from datetime import datetime, timedelta
from flask import request, jsonify
from functools import wraps

# 🔐 Clé secrète à sécuriser (utiliser dotenv en prod)
SECRET_KEY = "super-secret-key"

# 🔐 Générer un token
def generate_token(user_id):
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token if isinstance(token, str) else token.decode("utf-8")  # Compat PyJWT 1.x & 2.x

# 🔐 Middleware de protection
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Token manquant ou invalide"}), 401

        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expiré"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Token invalide"}), 401

        return f(user_id=payload["user_id"], *args, **kwargs)
    return decorated
