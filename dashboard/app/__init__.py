from flask import Flask

from config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    from app.routes import bp as dashboard_bp

    app.register_blueprint(dashboard_bp)

    return app
