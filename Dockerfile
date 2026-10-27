
# 使用 Python 3.14 的精简版基础镜像
FROM python:3.14-slim

# 容器内的项目工作目录
WORKDIR /app

# 先复制依赖清单，便于 Docker 缓存依赖安装层
COPY requirements.txt .

# 在容器自己的 Python 环境中安装项目依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码，不复制本地虚拟环境
COPY . .

# FastAPI 在容器内监听 8000 端口
EXPOSE 8000

# 启动应用；必须监听 0.0.0.0 才能从容器外访问
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
