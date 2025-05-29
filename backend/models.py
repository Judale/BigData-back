from datetime import datetime, timedelta
from enum import Enum
from backend.database import db

# ─────────────────────────────
# Enums
# ─────────────────────────────
class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"

class Mode(str, Enum):
    SINGLE  = "single"   # 1 catégorie
    MULTI   = "multi"    # plusieurs catégories
    ALL     = "all"      # tout le dico
    VERSUS  = "versus"   # 2 joueurs

# ─────────────────────────────
# Tables
# ─────────────────────────────
class User(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

class Category(db.Model):
    id   = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

class Word(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    text        = db.Column(db.String(50), unique=True, nullable=False)
    category_id = db.Column(
        db.Integer,
        db.ForeignKey("category.id", ondelete="CASCADE"),
        nullable=False
    )
    category    = db.relationship("Category", backref="words")

class Game(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    creator_id   = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    opponent_id  = db.Column(db.Integer, db.ForeignKey("user.id"))
    length       = db.Column(db.Integer, nullable=False)  # 5/10/20
    difficulty   = db.Column(db.Enum(Difficulty), nullable=False)
    mode         = db.Column(db.Enum(Mode), nullable=False)
    started_at   = db.Column(db.DateTime, default=datetime.utcnow)
    finished_at  = db.Column(db.DateTime)

class Round(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    game_id    = db.Column(db.Integer, db.ForeignKey("game.id"), nullable=False)
    word_id    = db.Column(db.Integer, db.ForeignKey("word.id"), nullable=False)
    order_idx  = db.Column(db.Integer, nullable=False)
    time_taken = db.Column(db.Float)  # secondes
    score      = db.Column(db.Integer)
    word = db.relationship("Word", backref="rounds")

class Drawing(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    round_id  = db.Column(db.Integer, db.ForeignKey("round.id"), nullable=False)
    ndjson    = db.Column(db.JSON, nullable=False)
    is_final  = db.Column(db.Boolean, default=False)

class Score(db.Model):
    __tablename__ = "scores"

    id           = db.Column(db.Integer, primary_key=True)
    game_id      = db.Column(db.Integer, db.ForeignKey("game.id"),  nullable=False)
    user_id      = db.Column(db.Integer, db.ForeignKey("user.id"),  nullable=False)
    total_points = db.Column(db.Integer, nullable=False)

    # ───── Contrainte d’unicité ─────
    __table_args__ = (
        db.UniqueConstraint("game_id", "user_id", name="uniq_game_user"),
    )