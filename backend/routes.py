from flask import Blueprint, request, jsonify, render_template_string
from backend.database import db
from backend.models import User, Game, Drawing
from backend.fake_model import predict
from backend.utils import generate_token, token_required
from werkzeug.security import generate_password_hash, check_password_hash
import random
from backend.model_inference import predict

api = Blueprint("api", __name__)

@api.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if User.query.filter_by(username=data["username"]).first():
        return jsonify({"error": "Username already exists"}), 400
    user = User(
        username=data["username"],
        password_hash=generate_password_hash(data["password"], method="pbkdf2:sha256")
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User created"}), 201

@api.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data["username"]).first()
    if user and check_password_hash(user.password_hash, data["password"]):
        token = generate_token(user.id)
        return jsonify({"token": token, "user_id": user.id})
    return jsonify({"error": "Invalid credentials"}), 401

@api.route("/start-game", methods=["POST"])
@token_required
def start_game(user_id):
    word_list = [
        ("airplane", "transport"),
        ("angel", "personnage"),
        ("apple", "fruit"),
        ("axe", "outil"),
        ("banana", "fruit"),
        ("bridge", "construction"),
        ("cup", "objet"),
        ("donut", "nourriture"),
        ("door", "objet"),
        ("mountain", "nature"),
    ]
    word, category = random.choice(word_list)
    game = Game(user_id=user_id, word=word, category=category)
    db.session.add(game)
    db.session.commit()
    return jsonify({
        "game_id": game.id,
        "word": word,
        "category": category
    })

@api.route("/submit-drawing", methods=["POST"])
@token_required
def submit_drawing(user_id):
    data = request.get_json()
    game = Game.query.get(data["game_id"])

    if not game or game.user_id != user_id:
        return jsonify({"error": "Partie introuvable ou non autorisée"}), 403

    ndjson = data["ndjson"]
    if isinstance(ndjson, dict) and "drawing" in ndjson:
        drawing = ndjson["drawing"]
    else:
        return jsonify({"error": "Format NDJSON invalide"}), 400

    is_final = data.get("is_final", False)
    db.session.add(Drawing(game_id=game.id, ndjson=ndjson, is_final=is_final))
    db.session.commit()

    # --- INFÉRENCE RÉELLE ---------------------------------------------------
    label, proba = predict(drawing)                # ← utilisation directe
    # -----------------------------------------------------------------------

    if label == game.word and proba >= 0.90:
        elapsed = data.get("elapsed_time", 30)
        if   elapsed <= 5:  score = 100
        elif elapsed <= 10: score = 80
        elif elapsed <= 15: score = 60
        elif elapsed <= 20: score = 40
        elif elapsed <= 25: score = 20
        else:               score = 10

        game.score = score
        db.session.commit()
        return jsonify({"status": "recognized",
                        "label": label,
                        "proba": round(proba, 2),
                        "score": score})

    return jsonify({"status": "pending",
                    "label": label,
                    "proba": round(proba, 2)})

@api.route("/drawing-view", methods=["POST"])
def drawing_view():
    data = request.get_json()
    strokes = []

    if "drawing_id" in data:
        drawing = Drawing.query.get(data["drawing_id"])
        if not drawing:
            return jsonify({"error": "Drawing not found"}), 404
        strokes = drawing.ndjson.get("drawing", [])

    elif "game_id" in data:
        drawings = Drawing.query.filter_by(game_id=data["game_id"]).all()
        for d in drawings:
            strokes.extend(d.ndjson.get("drawing", []))

    elif "drawing" in data:
        strokes = data["drawing"]

    else:
        return jsonify({"error": "Aucune donnée à afficher"}), 400

    svg_paths = ""
    for stroke in strokes:
        x, y = stroke
        path = "M " + " L ".join([f"{x[i]},{y[i]}" for i in range(len(x))])
        svg_paths += f'<path d="{path}" stroke="black" fill="none" stroke-width="2"/>' + "\n"

    html = f'''
    <!DOCTYPE html>
    <html>
    <head><title>Dessin NDJSON</title></head>
    <body>
        <h2>Dessin transmis</h2>
        <svg width="400" height="400" style="background:#fff;border:1px solid #ccc">
            {svg_paths}
        </svg>
    </body>
    </html>
    '''
    return render_template_string(html)

@api.route("/profile/me", methods=["GET"])
@token_required
def profile_me(user_id):
    games = Game.query.filter_by(user_id=user_id).all()
    total_games = len(games)
    total_score = sum(game.score or 0 for game in games)
    best_score = max((game.score or 0 for game in games), default=0)

    return jsonify({
        "total_games": total_games,
        "total_score": total_score,
        "best_score": best_score,
        "games": [
            {
                "game_id": game.id,
                "word": game.word,
                "category": game.category,
                "score": game.score
            }
            for game in games
        ]
    })

@api.route("/drawings/final", methods=["GET"])
@token_required
def get_final_drawings(user_id):
    # On récupère tous les dessins finaux liés aux parties du joueur
    games = Game.query.filter_by(user_id=user_id).all()
    game_ids = [g.id for g in games]

    final_drawings = Drawing.query.filter(
        Drawing.game_id.in_(game_ids),
        Drawing.is_final == True
    ).all()

    results = [
        {
            "drawing_id": d.id,
            "game_id": d.game_id,
            "ndjson": d.ndjson,
            "created_at": d.created_at.isoformat() if hasattr(d, "created_at") else None
        }
        for d in final_drawings
    ]

    return jsonify({"final_drawings": results})
