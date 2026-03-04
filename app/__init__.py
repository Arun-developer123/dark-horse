# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from config import DATABASE_URL, REDIS_URL, RATE_LIMIT_MAX

db = SQLAlchemy()

def create_app():
    app = Flask(__name__, template_folder="../templates")
    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    # Security
    Talisman(app, content_security_policy=None)

    # limiter
    if REDIS_URL:
        try:
            limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL, app=app)
        except Exception:
            limiter = Limiter(key_func=get_remote_address, default_limits=[f"{RATE_LIMIT_MAX}/hour"], app=app)
    else:
        limiter = Limiter(key_func=get_remote_address, default_limits=[f"{RATE_LIMIT_MAX}/hour"], app=app)

    # Register routes (blueprint style)
    from .routes import bp as routes_bp
    app.register_blueprint(routes_bp)

    # Create DB if needed
    with app.app_context():
        from . import models
        db.create_all()

    return app