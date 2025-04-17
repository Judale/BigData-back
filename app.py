from flask import Flask
from backend.routes import api
from flask_cors import CORS
from backend.database import init_db

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///quickdraw.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

CORS(app)

init_db(app)
app.register_blueprint(api)

if __name__ == "__main__":
    app.run(debug=True)