import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def CreateSqlAlchemyEngine() -> Engine:
    """Tworzy silnik SQLAlchemy na podstawie zmiennych z .env.
    Oczekiwane zmienne: MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD
    """
    load_dotenv()
    # Preferuj pojedynczy DATABASE_URL, jeśli dostępny
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        # Ujednolić driver do pymysql, jeśli użyto skrótu mysql://
        if db_url.startswith("mysql://") and not db_url.startswith("mysql+pymysql://"):
            db_url = db_url.replace("mysql://", "mysql+pymysql://", 1)
        # Dopnij charset, jeśli brak
        if "mysql+pymysql://" in db_url and "charset=" not in db_url:
            sep = "&" if "?" in db_url else "?"
            db_url = f"{db_url}{sep}charset=utf8mb4"
        engine = create_engine(db_url, pool_pre_ping=True)
        return engine
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    name = os.getenv("MYSQL_DATABASE")
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")

    if not all([name, user, password]):
        raise RuntimeError("Brak wymaganych zmiennych środowiskowych: MYSQL_DATABASE/MYSQL_USER/MYSQL_PASSWORD")

    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"
    engine = create_engine(url, pool_pre_ping=True)
    return engine
