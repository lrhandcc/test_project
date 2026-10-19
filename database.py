import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker, declarative_base


# 读取项目根目录中的 .env 配置
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")

# 检查必要的配置是否齐全
if not all([DB_USER, DB_PASSWORD, DB_NAME]):
    raise RuntimeError(
        "数据库配置不完整，请检查项目根目录下的 .env 文件"
    )

# 创建 SQLAlchemy 数据库连接地址
# 使用 URL.create，避免密码中的特殊字符破坏连接地址
DATABASE_URL = URL.create(
    drivername="mysql+pymysql",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME,
)

# 创建数据库引擎
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

# 创建数据库会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# 声明 ORM 模型的共同基类
Base = declarative_base()


# 为 FastAPI 请求提供数据库会话
def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
