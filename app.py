from flask import Flask
from flask_cors import CORS
from flasgger import Swagger
from backend.database import db, migrate
from backend.routes import api_blueprint

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SQLALCHEMY_DATABASE_URI="sqlite:///../instance/quickdraw.db",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SECRET_KEY="change-me-in-prod",
        SWAGGER = {
            'title': 'Gribouillon API',
            'uiversion': 3,
            'openapi': '3.0.2',
        }
    )

    # Extensions
    db.init_app(app)
    migrate.init_app(app, db)
    CORS(app, supports_credentials=True)

    Swagger(app)

    # Blueprints
    app.register_blueprint(api_blueprint, url_prefix="/api")

    return app

if __name__ == "__main__":
    create_app().run(debug=True)
