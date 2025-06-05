from collections import Counter
from datetime import datetime, timedelta
import random

from flasgger import swag_from
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
    """
    Création d'un nouvel utilisateur
    ---
    tags:
      - Auth
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              username:
                type: string
                example: "alice"
              password:
                type: string
                example: "secret"
    responses:
      201:
        description: Utilisateur créé
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: "User created"
      400:
        description: Nom d'utilisateur déjà existant
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: "Username already exists"
    """
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
    """
    Authentification d'un utilisateur
    ---
    tags:
      - Auth
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              username:
                type: string
              password:
                type: string
    responses:
      200:
        description: Connexion réussie
        content:
          application/json:
            schema:
              type: object
              properties:
                token:
                  type: string
                user_id:
                  type: integer
      401:
        description: Identifiants invalides
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: "Invalid credentials"
    """
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
def start_game(user_id):
    """
    Démarre une nouvelle partie
    ---
    tags:
      - Game
    security:
      - Bearer: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              length:
                type: integer
                example: 5
              difficulty:
                type: string
                enum: [easy, medium, hard]
              mode:
                type: string
                enum: [single, multi, all, versus]
              categories:
                type: array
                items:
                  type: string
              opponent_id:
                type: integer
    responses:
      201:
        description: Partie créée
        content:
          application/json:
            schema:
              type: object
              properties:
                game_id:
                  type: integer
                round_id:
                  type: integer
                word:
                  type: string
      400:
        description: Paramètres invalides
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
    """
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
    """
    Récupère le mot suivant dans la partie
    ---
    tags:
      - Game
    parameters:
      - in: path
        name: game_id
        schema:
          type: integer
      - in: path
        name: current_idx
        schema:
          type: integer
    responses:
      200:
        description: Mot suivant
        content:
          application/json:
            schema:
              type: object
              properties:
                round_id:
                  type: integer
                word:
                  type: string
    """
    rnd = Round.query.filter_by(game_id=game_id, order_idx=current_idx + 1).first()
    if not rnd:
        return jsonify({"word": None})
    return jsonify({
        "round_id": rnd.id,
        "word": Word.query.get(rnd.word_id).text,
    })


@api_blueprint.route("/submit-drawing", methods=["POST"])
def submit_drawing():
    """
    Soumet un dessin pour reconnaissance
    ---
    tags:
      - Drawing
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              round_id:
                type: integer
              elapsed_time:
                type: number
              ndjson:
                type: object
    responses:
      200:
        description: Résultat de la reconnaissance
        content:
          application/json:
            schema:
              type: object
              properties:
                status:
                  type: string
                score:
                  type: integer
                model:
                  type: string
    """
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
    """
    Termine la partie et enregistre le score
    ---
    tags:
      - Game
    security:
      - Bearer: []
    parameters:
      - in: path
        name: game_id
        schema:
          type: integer
    responses:
      200:
        description: Score total
        content:
          application/json:
            schema:
              type: object
              properties:
                total_points:
                  type: integer
      409:
        description: Score déjà enregistré
    """
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
    """
    Affiche un dessin en SVG pour debug
    ---
    tags:
      - Debug
    requestBody:
      required: true
    responses:
      200:
        description: Vue SVG du dessin
      400:
        description: Pas de donnée à afficher
    """
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
    """
    Récupère le profil et scores de l'utilisateur
    ---
    tags:
      - Profile
    security:
      - Bearer: []
    responses:
      200:
        description: Statistiques de l'utilisateur
    """
    scores = Score.query.filter_by(user_id=user_id).all()
    total_games = len(scores)
    total_score = sum(s.total_points for s in scores)
    best_score = max((s.total_points for s in scores), default=0)

    user = User.query.get(user_id)
    username = user.username if user else None

    return jsonify(
        {
            "username": username,
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
    """
    Liste tous les dessins finaux de l'utilisateur
    ---
    tags:
      - Drawing
    security:
      - Bearer: []
    responses:
      200:
        description: Dessins finaux
    """
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
    """
    Récupère la liste des catégories de mots
    ---
    tags:
      - Meta
    responses:
      200:
        description: Liste des catégories
    """
    categories = Category.query.all()
    return jsonify([
        {"id": cat.id, "name": cat.name}
        for cat in categories
    ])

@api_blueprint.route("/profile/stats", methods=["GET"])
@token_required
def get_profile_stats(user_id):
    """
    Statistiques détaillées du profil utilisateur
    ---
    tags:
      - Profile
    security:
      - Bearer: []
    responses:
      200:
        description: Statistiques par difficulté, catégorie et mots
    """
    scores = Score.query.filter_by(user_id=user_id).all()
    game_ids = [s.game_id for s in scores]
    games = Game.query.filter(Game.id.in_(game_ids)).all()
    user = User.query.get(user_id)
    username = user.username if user else None

    # Rounds de l'utilisateur
    rounds = Round.query.join(Game).filter(Game.id.in_(game_ids)).all()

    # Nombre de parties jouées
    total_games = len(games)
    best_score = max((s.total_points for s in scores), default=0)
    avg_score = round(sum(s.total_points for s in scores) / total_games, 2) if total_games else 0

    # Temps moyen de détection IA global
    times = [r.time_taken for r in rounds if r.time_taken is not None]
    avg_time_taken = round(sum(times) / len(times), 2) if times else None

    # --- AVG TIME TAKEN PAR GAME ---
    avg_time_taken_per_game = {}
    for g in games:
        r_times = [r.time_taken for r in rounds if r.game_id == g.id and r.time_taken is not None]
        avg_time_taken_per_game[g.id] = round(sum(r_times) / len(r_times), 2) if r_times else None

    # --- PAR CATÉGORIE (par partie, pas par dessin) ---
    category_stats = {}
    for g in games:
        # On récupère tous les rounds de la partie
        rounds_in_game = [r for r in rounds if r.game_id == g.id]
        # On récupère toutes les catégories présentes dans cette partie
        categories_in_game = set()
        for r in rounds_in_game:
            word = Word.query.get(r.word_id)
            category = Category.query.get(word.category_id)
            categories_in_game.add(category.name)
        # On ajoute le score total de la partie à chaque catégorie concernée
        for cname in categories_in_game:
            if cname not in category_stats:
                category_stats[cname] = {"count": 0, "total_score": 0, "times": []}
            category_stats[cname]["count"] += 1
            # Score total de la partie
            score = next((s.total_points for s in scores if s.game_id == g.id), 0)
            category_stats[cname]["total_score"] += score
            # Temps moyen de détection IA pour cette partie
            r_times = [r.time_taken for r in rounds_in_game if r.time_taken is not None]
            if r_times:
                category_stats[cname]["times"].append(sum(r_times) / len(r_times))
    # Finalise les moyennes
    for cname, stat in category_stats.items():
        stat["avg_score"] = round(stat["total_score"] / stat["count"], 2) if stat["count"] else 0
        stat["avg_time_taken"] = round(sum(stat["times"]) / len(stat["times"]), 2) if stat["times"] else None
        del stat["total_score"]
        del stat["times"]

    # --- PAR DIFFICULTÉ ---
    diff_stats = {}
    for g in games:
        diff = g.difficulty.value
        if diff not in diff_stats:
            diff_stats[diff] = {"count": 0, "total_score": 0, "times": []}
        diff_stats[diff]["count"] += 1
        score = next((s.total_points for s in scores if s.game_id == g.id), 0)
        diff_stats[diff]["total_score"] += score
        r_times = [r.time_taken for r in rounds if r.game_id == g.id and r.time_taken is not None]
        diff_stats[diff]["times"].extend(r_times)
    for diff, stat in diff_stats.items():
        stat["avg_score"] = round(stat["total_score"] / stat["count"], 2) if stat["count"] else 0
        stat["avg_time_taken"] = round(sum(stat["times"]) / len(stat["times"]), 2) if stat["times"] else None
        del stat["total_score"]
        del stat["times"]

    # --- PAR DURÉE ---
    duration_stats = {
        "short": {"count": 0, "total_score": 0, "times": []},
        "medium": {"count": 0, "total_score": 0, "times": []},
        "long": {"count": 0, "total_score": 0, "times": []},
    }
    for g in games:
        if g.length == 5:
            key = "short"
            score = next((s.total_points for s in scores if s.game_id == g.id), 0)
            score_affiche = score
        elif g.length == 10:
            key = "medium"
            score = next((s.total_points for s in scores if s.game_id == g.id), 0)
            score_affiche = round(score * 500 / 1000, 2)
        elif g.length == 15:
            key = "long"
            score = next((s.total_points for s in scores if s.game_id == g.id), 0)
            score_affiche = round(score * 500 / 1500, 2)
        else:
            continue
        duration_stats[key]["count"] += 1
        duration_stats[key]["total_score"] += score_affiche
        r_times = [r.time_taken for r in rounds if r.game_id == g.id and r.time_taken is not None]
        if r_times:
            duration_stats[key]["times"].append(sum(r_times) / len(r_times))
    # Finalise les moyennes
    for key, stat in duration_stats.items():
        stat["avg_score"] = round(stat["total_score"] / stat["count"], 2) if stat["count"] else 0
        stat["avg_time_taken"] = round(sum(stat["times"]) / len(stat["times"]), 2) if stat["times"] else None
        del stat["total_score"]
        del stat["times"]

    # --- PAR PÉRIODE ---
    now = datetime.utcnow()
    periods = {
        "day": now - timedelta(days=1),
        "week": now - timedelta(weeks=1),
        "month": now - timedelta(days=30),
        "alltime": datetime.min,
    }
    period_stats = {}
    for pname, since in periods.items():
        games_in_period = [g for g in games if g.started_at and g.started_at >= since]
        scores_in_period = [s for s in scores if any(g.id == s.game_id for g in games_in_period)]
        rounds_in_period = [r for r in rounds if any(g.id == r.game_id for g in games_in_period)]
        total_games_p = len(games_in_period)
        avg_score_p = round(sum(s.total_points for s in scores_in_period) / total_games_p, 2) if total_games_p else 0
        best_score_p = max((s.total_points for s in scores_in_period), default=0)
        times_p = [r.time_taken for r in rounds_in_period if r.time_taken is not None]
        avg_time_taken_p = round(sum(times_p) / len(times_p), 2) if times_p else None
        period_stats[pname] = {
            "total_games": total_games_p,
            "avg_score": avg_score_p,
            "best_score": best_score_p,
            "avg_time_taken": avg_time_taken_p,
        }

    # --- Nombre de mots joués ---
    total_words = len(rounds)

    # --- Tableau games au format demandé ---
    games_array = []
    for s in scores:
        g = next((game for game in games if game.id == s.game_id), None)
        if not g:
            continue
        # Détermination de la durée
        if g.length == 5:
            duration = "short"
            total_points_affiche = s.total_points
        elif g.length == 10:
            duration = "medium"
            total_points_affiche = round(s.total_points * 500 / 1000, 2)
        elif g.length == 15:
            duration = "long"
            total_points_affiche = round(s.total_points * 500 / 1500, 2)
        else:
            duration = str(g.length)
            total_points_affiche = s.total_points
        # Récupère toutes les catégories jouées dans la partie
        rounds_in_game = [r for r in rounds if r.game_id == g.id]
        categories_in_game = set()
        for r in rounds_in_game:
            word = Word.query.get(r.word_id)
            category = Category.query.get(word.category_id)
            categories_in_game.add(category.name)
        games_array.append({
            "game_id": s.game_id,
            "total_points": total_points_affiche,
            "avg_time_taken": avg_time_taken_per_game.get(s.game_id),
            "duration": duration,
            "categories": list(categories_in_game),
            "difficulty": g.difficulty.value if hasattr(g, "difficulty") else None,
            "started_at": g.started_at.isoformat() if g.started_at else None,
            "finished_at": g.finished_at.isoformat() if g.finished_at else None
        })

    # --- TOP 5 & FLOP 10 MOTS ---
    # On regroupe les scores par mot
    word_scores = {}
    for r in rounds:
        word = Word.query.get(r.word_id)
        if word.text not in word_scores:
            word_scores[word.text] = []
        if r.score is not None:
            word_scores[word.text].append(r.score)
    # Calcule la moyenne par mot
    word_avg_scores = [
        (w, round(sum(scores) / len(scores), 2)) for w, scores in word_scores.items() if scores
    ]
    # Trie pour top et flop
    word_avg_scores_sorted = sorted(word_avg_scores, key=lambda x: x[1], reverse=True)
    top_5_words = word_avg_scores_sorted[:5]
    flop_5_words = sorted(word_avg_scores_sorted[-5:], key=lambda x: x[1])  # du pire au moins pire

    return jsonify({
        "username": username,
        "total_games": total_games,
        "best_score": best_score,
        "avg_score": avg_score,
        "avg_time_taken": avg_time_taken,
        "difficulty_stats": diff_stats,
        "category_stats": category_stats,
        "total_words": total_words,
        "period_stats": period_stats,
        "duration_stats": duration_stats, 
        "games": games_array,
        "top_words": [{"word": w, "avg_score_per_drawing": s} for w, s in top_5_words],
        "flop_words": [{"word": w, "avg_score_per_drawing": s} for w, s in flop_5_words],
    })

@api_blueprint.route("/general-stats", methods=["GET"])
def get_general_stats():
    """
    Renvoie les statistiques globales nécessaires au dashboard :
      - leaderboard (top 10 utilisateurs, somme des points ET meilleur score sur une partie)
      - avg_points_per_drawing (moyenne des points par dessin)
      - avg_points_per_category (moyenne des points par catégorie)
      - games_per_category (nombre de parties jouées par catégorie)
    """
    # ─── 1) Leaderboard top 10 (sum et max de Score.total_points) ───
    leaderboard = (
        db.session.query(
            User.username,
            func.sum(Score.total_points).label("sum_points"),
            func.max(Score.total_points).label("max_points")
        )
        .join(Score, Score.user_id == User.id)
        .group_by(User.id)
        .order_by(func.max(Score.total_points).desc())
        .limit(10)
        .all()
    )

    leaderboard_data = [
        {
            "username": row.username,
            "sum_points": int(row.sum_points or 0),
            "max_points": int(row.max_points or 0)
        }
        for row in leaderboard
    ]

    # ─── 2) Difficultés disponibles (non utilisé côté front ici) ───
    difficulties = [d.value for d in Difficulty]

    # ─── 3) Moyenne de points par dessin (Round.score != NULL) ───
    avg_points_per_drawing = (
        db.session.query(func.avg(Round.score))
        .filter(Round.score != None)
        .scalar()
    )
    avg_points_per_drawing = round(avg_points_per_drawing, 2) if avg_points_per_drawing else 0

    # ─── 4) Moyenne de points par catégorie ───
    category_scores = (
        db.session.query(
            Category.name,
            func.avg(Round.score)
        )
        .join(Word, Word.category_id == Category.id)
        .join(Round, Round.word_id == Word.id)
        .filter(Round.score != None)
        .group_by(Category.id)
        .all()
    )
    avg_points_per_category = {
        name: round(avg_score, 2) if avg_score else 0
        for name, avg_score in category_scores
    }

    # ─── 5) Nombre de parties jouées par catégorie ───
    # On considère qu'une partie « appartient » à une catégorie si au moins un round
    # de cette partie a été joué sur un mot de cette catégorie.
    rounds_with_scores = (
        db.session.query(Round.game_id, Category.name)
        .join(Word, Round.word_id == Word.id)
        .join(Category, Word.category_id == Category.id)
        .filter(Round.score != None)
        .all()
    )
    cat_to_game_ids: dict[str, set[int]] = {}
    for game_id, cat_name in rounds_with_scores:
        if cat_name not in cat_to_game_ids:
            cat_to_game_ids[cat_name] = set()
        cat_to_game_ids[cat_name].add(game_id)

    games_per_category = {cat: len(game_ids) for cat, game_ids in cat_to_game_ids.items()}

    return jsonify({
        "leaderboard": leaderboard_data,
        "difficulties": difficulties,
        "avg_points_per_drawing": avg_points_per_drawing,
        "avg_points_per_category": avg_points_per_category,
        "games_per_category": games_per_category
    })

@api_blueprint.route("/change-password", methods=["POST"])
@token_required
def change_password(user_id):
    data = request.get_json()
    old_password = data.get("old_password")
    new_password = data.get("new_password")

    if not old_password or not new_password:
        return jsonify({"error": "Both old and new passwords are required"}), 400

    user = User.query.get(user_id)
    if not user or not check_password_hash(user.password_hash, old_password):
        return jsonify({"error": "Old password is incorrect"}), 401

    user.password_hash = generate_password_hash(new_password, method="pbkdf2:sha256")
    db.session.commit()
    return jsonify({"message": "Password updated successfully"})
