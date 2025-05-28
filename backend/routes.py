from collections import Counter
from datetime import datetime
import random

from flask import Blueprint, request, jsonify, render_template_string
from sqlalchemy.sql import func
from sqlalchemy.exc import IntegrityError   # ← pour capturer l’éventuelle violation d’unicité

from backend.database import db
from backend.models import (
    User,
    Game,
    Drawing,
    Mode,
    Word,
    Category,
    Difficulty,
    Round,
    Score,
)
from backend.model_inference import predict, CLASSES_ANIMAUX, CLASSES_OBJETS, CLASSES_NOURRITURE
from backend.utils import generate_token, token_required, compute_score
from werkzeug.security import generate_password_hash, check_password_hash

# ─────────────────────────────
# Blueprint
# ─────────────────────────────
api_blueprint = Blueprint("api", __name__)

CLASSES_10 = [
    "airplane", "angel", "apple", "axe", "banana",
    "bridge", "cup", "donut", "door", "mountain"
]

# ---------------------------------
# Auth routes
# ---------------------------------
@api_blueprint.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if User.query.filter_by(username=data["username"]).first():
        return jsonify({"error": "Username already exists"}), 400
    user = User(
        username=data["username"],
        password_hash=generate_password_hash(data["password"], method="pbkdf2:sha256"),
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User created"}), 201


@api_blueprint.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data["username"]).first()
    if user and check_password_hash(user.password_hash, data["password"]):
        token = generate_token(user.id)
        return jsonify({"token": token, "user_id": user.id})
    return jsonify({"error": "Invalid credentials"}), 401

# ---------------------------------
# Helpers
# ---------------------------------

def _select_words(mode: Mode, length: int, categories: list[str]) -> list[Word]:
    """Sélectionne les `Word` à deviner selon le mode."""
    if mode == Mode.SINGLE:
        if not categories:
            raise ValueError("categories required for single mode")
        cat = Category.query.filter_by(name=categories[0]).first_or_404()
        return random.sample(cat.words, min(length, len(cat.words)))

    if mode == Mode.MULTI:
        if not categories:
            raise ValueError("categories required for multi mode")
        words: list[Word] = []
        per_cat = max(1, length // len(categories))
        for cname in categories:
            cat = Category.query.filter_by(name=cname).first_or_404()
            words.extend(random.sample(cat.words, min(per_cat, len(cat.words))))
        pool = Word.query.join(Category).filter(Category.name.in_(categories)).all()
        while len(words) < length:
            words.append(random.choice(pool))
        return random.sample(words, length)

    if mode in (Mode.ALL, Mode.VERSUS):
        return random.sample(Word.query.all(), length)

    raise ValueError("unknown mode")

# ---------------------------------
# Game lifecycle
# ---------------------------------

@api_blueprint.route("/start-game", methods=["POST"])
@token_required
def start_game(user_id):  # Le token donne user_id
    data = request.get_json(force=True)
    length = int(data.get("length", 5))
    difficulty = Difficulty(data.get("difficulty", "easy"))
    mode = Mode(data.get("mode", "single"))
    categories = data.get("categories", [])
    opponent_id = data.get("opponent_id")

    game = Game(
        creator_id=user_id,
        opponent_id=opponent_id,
        length=length,
        difficulty=difficulty,
        mode=mode,
    )
    db.session.add(game)
    db.session.flush()

    try:
        words = _select_words(mode, length, categories)
    except ValueError as exc:
        db.session.rollback()
        return jsonify({"error": str(exc)}), 400

    for idx, w in enumerate(words):
        db.session.add(Round(game_id=game.id, word_id=w.id, order_idx=idx))

    db.session.commit()

    first_round = Round.query.filter_by(game_id=game.id, order_idx=0).first()
    return (
        jsonify(
            {
                "game_id": game.id,
                "round_id": first_round.id,
                "word": Word.query.get(first_round.word_id).text,
            }
        ),
        201,
    )


@api_blueprint.route("/next-word/<int:game_id>/<int:current_idx>")
def next_word(game_id: int, current_idx: int):
    rnd = Round.query.filter_by(game_id=game_id, order_idx=current_idx + 1).first()
    if not rnd:
        return jsonify({"word": None})
    return jsonify({
        "round_id": rnd.id,
        "word": Word.query.get(rnd.word_id).text,
    })


@api_blueprint.route("/submit-drawing", methods=["POST"])
def submit_drawing():
    data = request.get_json(force=True)
    round_id = data["round_id"]
    elapsed = float(data["elapsed_time"])
    drawing = data["ndjson"]

    rnd = Round.query.get_or_404(round_id)
    game = Game.query.get(rnd.game_id)
    target_word = Word.query.get(rnd.word_id).text

    # Normalisation du mot pour comparaison (underscore si besoin)
    normalized_word = target_word.lower().replace(" ", "_")

    # Détermine le modèle à utiliser
    if normalized_word in CLASSES_10:
        model_type = "default"
    elif normalized_word in CLASSES_ANIMAUX:
        model_type = "animaux"
    elif normalized_word in CLASSES_OBJETS:
        model_type = "objets"
    elif normalized_word in CLASSES_NOURRITURE:
        model_type = "nourriture"
    else:
        model_type = "extended"

    # Prédiction
    label, proba = predict(drawing["drawing"], model_type=model_type)

    # Reconnaissance correcte
    if label.replace("_", " ") == target_word.lower():
        sc = compute_score(elapsed, game.difficulty.value)
        rnd.time_taken = elapsed
        rnd.score = sc
        db.session.add(Drawing(round_id=rnd.id, ndjson=drawing, is_final=True))
        db.session.commit()
        return jsonify({"status": "recognized", "score": sc, "model": model_type})

    # Reconnaissance incorrecte
    db.session.add(Drawing(round_id=rnd.id, ndjson=drawing, is_final=False))
    db.session.commit()
    return jsonify({
        "status": "pending",
        "label": label.replace("_", " "),
        "proba": round(proba, 2),
        "model": model_type
    })

@api_blueprint.route("/finish-game/<int:game_id>", methods=["POST"])
@token_required
def finish_game(user_id, game_id: int):
    """Calcule le score total et l’enregistre (1 seule ligne par couple game/user)."""
    game = Game.query.get_or_404(game_id)

    total = (
        db.session.query(func.sum(Round.score))
        .filter_by(game_id=game.id)
        .scalar()
    ) or 0

    # user_id provient du token, on n’a pas besoin d’un header externe

    # Cherche un score déjà existant
    score = Score.query.filter_by(game_id=game.id, user_id=user_id).first()
    if score:
        score.total_points = total  # mise à jour
    else:
        db.session.add(Score(game_id=game.id, user_id=user_id, total_points=total))

    # Ne renseigne finished_at qu’une seule fois
    if game.finished_at is None:
        game.finished_at = datetime.utcnow()

    try:
        db.session.commit()
    except IntegrityError:  # au cas où deux requêtes concurrentes passent quand même
        db.session.rollback()
        return jsonify({"error": "Score already recorded"}), 409

    return jsonify({"total_points": total})

# ---------------------------------
# Debug : visualiser un dessin
# ---------------------------------
@api_blueprint.route("/drawing-view", methods=["POST"])
def drawing_view():
    data = request.get_json()
    strokes: list = []

    if "drawing_id" in data:
        drawing = Drawing.query.get(data["drawing_id"])
        if not drawing:
            return jsonify({"error": "Drawing not found"}), 404
        strokes = drawing.ndjson.get("drawing", [])

    elif "game_id" in data:
        rounds = Round.query.filter_by(game_id=data["game_id"]).all()
        for r in rounds:
            fin = Drawing.query.filter_by(round_id=r.id, is_final=True).first()
            if fin:
                strokes.extend(fin.ndjson.get("drawing", []))

    elif "drawing" in data:
        strokes = data["drawing"]

    else:
        return jsonify({"error": "Aucune donnée à afficher"}), 400

    svg_paths = ""
    for stroke in strokes:
        x, y = stroke
        path = "M " + " L ".join([f"{x[i]},{y[i]}" for i in range(len(x))])
        svg_paths += f'<path d="{path}" stroke="black" fill="none" stroke-width="2"/>' + "\n"

    html = f"""
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
    """
    return render_template_string(html)

# ---------------------------------
# Profil / scores
# ---------------------------------
@api_blueprint.route("/profile/me", methods=["GET"])
@token_required
def profile_me(user_id):
    scores = Score.query.filter_by(user_id=user_id).all()
    total_games = len(scores)
    total_score = sum(s.total_points for s in scores)
    best_score = max((s.total_points for s in scores), default=0)

    return jsonify(
        {
            "total_games": total_games,
            "total_score": total_score,
            "best_score": best_score,
            "games": [
                {
                    "game_id": s.game_id,
                    "total_points": s.total_points,
                }
                for s in scores
            ],
        }
    )


@api_blueprint.route("/drawings/final", methods=["GET"])
@token_required
def get_final_drawings(user_id):
    rounds = (
        db.session.query(Round)
        .join(Game, Round.game_id == Game.id)
        .join(Word, Round.word_id == Word.id)
        .filter((Game.creator_id == user_id) | (Game.opponent_id == user_id))
        .all()
    )

    # Mapping round_id → (game_id, word)
    round_map = {r.id: {"game_id": r.game_id, "word": r.word.text} for r in rounds}

    round_ids = list(round_map.keys())

    final_drawings = Drawing.query.filter(
        Drawing.round_id.in_(round_ids), Drawing.is_final.is_(True)
    ).all()

    return jsonify(
        {
            "final_drawings": [
                {
                    "drawing_id": d.id,
                    "round_id": d.round_id,
                    "game_id": round_map[d.round_id]["game_id"],
                    "word": round_map[d.round_id]["word"],
                    "ndjson": d.ndjson,
                }
                for d in final_drawings
            ]
        }
    )

@api_blueprint.route("/categories", methods=["GET"])
def get_categories():
    categories = Category.query.all()
    return jsonify([
        {"id": cat.id, "name": cat.name}
        for cat in categories
    ])

@api_blueprint.route("/profile/stats", methods=["GET"])
@token_required
def get_profile_stats(user_id):
    scores = Score.query.filter_by(user_id=user_id).all()
    game_ids = [s.game_id for s in scores]
    games = Game.query.filter(Game.id.in_(game_ids)).all()

    diff_stats = {}
    for g in games:
        diff = g.difficulty.value
        if diff not in diff_stats:
            diff_stats[diff] = {"count": 0, "total_score": 0}
        diff_stats[diff]["count"] += 1
        score = next((s.total_points for s in scores if s.game_id == g.id), 0)
        diff_stats[diff]["total_score"] += score

    category_counter = Counter()

    for g in games:
        rounds = Round.query.filter_by(game_id=g.id).all()
        categories_in_game = set()
        for r in rounds:
            word = Word.query.get(r.word_id)
            category = Category.query.get(word.category_id)
            categories_in_game.add(category.name)
        for cname in categories_in_game:
            category_counter[cname] += 1

    word_scores = {}
    for g in games:
        rounds = Round.query.filter_by(game_id=g.id).all()
        for r in rounds:
            word = Word.query.get(r.word_id)
            if word.text not in word_scores:
                word_scores[word.text] = {"total": 0, "count": 0, "has_score": False}
            word_scores[word.text]["total"] += r.score or 0
            word_scores[word.text]["count"] += 1
            if (r.score or 0) > 0:
                word_scores[word.text]["has_score"] = True

    top_words_avg_score = sorted(
        [
            (word, round(info["total"] / info["count"], 2))
            for word, info in word_scores.items()
            if info["has_score"]
        ],
        key=lambda x: x[1],
        reverse=True
    )[:5]

    return jsonify({
        "total_games": len(games),
        "avg_score": round(sum(s.total_points for s in scores) / len(scores), 2) if scores else 0,
        "difficulty_stats": {
            k: {
                "count": v["count"],
                "avg_score": round(v["total_score"] / v["count"], 2)
            } for k, v in diff_stats.items()
        },
        "category_stats": category_counter,
        "top_words": top_words_avg_score
    })
