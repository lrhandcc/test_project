# FastAPI AI 应用与 RAG / Agent 项目

一个基于 Python、FastAPI 和 MySQL 的后端应用项目，在用户数据管理和数据库持久化的基础上，集成大语言模型对话、检索增强生成（RAG）和 Agent 工具调用能力。

本项目用于实践 Python 后端开发、AI 应用集成、数据库访问、安全的工具调用设计和自动化测试，重点关注功能实现、工程结构、测试验证与技术取舍。

## 一、项目功能

### 1. 后端 API 与数据库

- 使用 FastAPI 构建 HTTP API。
- 使用 Pydantic 校验请求数据并定义响应结构。
- 使用 SQLAlchemy ORM 操作 MySQL 数据库。
- 实现用户数据的创建、查询、更新和删除。
- 使用依赖注入管理数据库 Session。
- 对不存在的用户、聊天会话等资源返回相应的 HTTP 错误。

### 2. 大语言模型对话

- 通过 OpenAI 兼容接口调用大语言模型，当前配置使用 OpenRouter。
- 支持普通对话与基于历史记录的多轮对话。
- 支持流式输出，采用 NDJSON 格式逐行返回数据。
- 支持聊天会话与消息记录持久化。
- 对模型连接失败、超时、请求受限等情况进行异常处理。
- 通过环境变量管理数据库和模型服务配置。

### 3. RAG 检索增强生成

- 使用 TF-IDF 字符 n-gram 特征检索相关文本。
- 结合检索分数与问题和文本块之间的特征重合度筛选候选证据。
- 支持文本分块、重叠配置、Top-K 和阈值等参数实验。
- 将检索到的文本作为大语言模型生成回答的参考依据。
- 在证据不足时直接拒答，避免在没有足够知识库依据时继续生成答案。
- 使用本地评测用例和参数实验观察检索行为。

当前 RAG 采用轻量级检索方案，重点在于理解检索、证据筛选、拒答和评测流程，并非完整的生产级知识检索平台。

### 4. Agent 工具调用

- 支持大语言模型发起工具调用。
- 提供计算器工具和受限的数据库查询工具。
- 使用工具白名单限制可执行操作。
- 校验工具名称、输入参数和数据库查询条件。
- 数据库工具通过预定义查询逻辑访问数据，不允许模型任意执行 SQL。
- 限制单次 Agent 执行中的工具调用轮数。
- 将工具执行结果回传模型，支持继续推理和生成回答。
- 对模型调用及工具执行异常进行受控处理。

## 二、技术栈

| 类别 | 技术 |
| --- | --- |
| 编程语言 | Python |
| Web 框架 | FastAPI、Uvicorn |
| 数据校验 | Pydantic |
| ORM | SQLAlchemy |
| 数据库 | MySQL |
| 数据库驱动 | PyMySQL |
| AI 接口 | OpenAI Python SDK、OpenRouter |
| RAG 检索 | scikit-learn、TF-IDF、字符 n-gram |
| 自动化测试 | pytest、Mock |
| 版本管理 | Git、GitHub |

## 三、项目结构

项目主要目录及其职责如下，具体文件以仓库当前内容为准。

```text
test_project/
├── main.py                  # FastAPI 应用入口与路由注册
├── database.py              # 数据库连接、Session 与基础配置
├── routers/                 # HTTP 路由与请求处理
├── schemas/                 # 请求与响应数据结构
├── models/                  # ORM 数据模型
├── services/                # 业务逻辑、AI 服务与 Agent 工具
├── rag/                     # RAG 检索、评测与实验
├── tests/                   # 自动化测试
├── requirements.txt         # Python 依赖清单
├── .env                     # 本地环境配置，不提交到 Git
└── README.md                # 项目说明文档
```

## 四、环境配置

### 1. 环境要求

- Python：当前开发环境为 Python 3.14；其他 Python 版本尚未在本项目中完整验证。
- MySQL：当前使用 MySQL 8.x。
- Git。
- 可访问的 OpenAI 兼容模型服务，以及有效的 API Key。

### 2. 克隆项目并创建虚拟环境

```bash
git clone <your-repository-url>
cd test_project

python3 -m venv .venv
source .venv/bin/activate
```

将 `<your-repository-url>` 替换为实际的 Git 仓库地址。

### 3. 安装依赖

在虚拟环境激活后执行：

```bash
python -m pip install -r requirements.txt
```

项目依赖版本记录在 `requirements.txt` 中。当前清单根据开发环境中已核对的核心依赖整理；如后续新增正式功能或测试依赖，应同步更新清单并验证安装结果。

### 4. 配置环境变量

在项目根目录创建本地 `.env` 文件。以下为配置示例，请根据自己的数据库和模型服务填写真实值：

```dotenv
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=python_backend

OPENROUTER_API_KEY=your_openrouter_api_key
```

说明：

- `DB_USER`：数据库用户名。
- `DB_PASSWORD`：数据库密码。
- `DB_HOST`：数据库地址；示例中的 `localhost` 表示当前运行环境。
- `DB_PORT`：数据库端口，示例使用 `3306`。
- `DB_NAME`：数据库名称。
- `OPENROUTER_API_KEY`：模型服务 API Key。

请确保数据库已经创建，并且配置的账号具有项目所需的数据库权限。远程服务器上的数据库地址应根据实际部署环境设置。

不要将 `.env`、真实数据库密码或 API Key 提交到 GitHub。

## 五、启动项目

完成依赖安装、数据库配置和环境变量设置后，在项目根目录执行：

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

常用访问地址：

- Swagger API 文档：`http://127.0.0.1:8000/docs`
- OpenAPI Schema：`http://127.0.0.1:8000/openapi.json`

如果应用运行在远程 Linux 服务器中，应将 `127.0.0.1` 替换为当前环境中可访问的服务器地址，并确认网络与端口策略允许访问。上述地址是本地访问示例，不代表服务已公开部署到互联网。

`--reload` 适用于开发调试，不建议直接作为生产环境的运行配置。

## 六、API 接口说明

以下路径依据当前路由定义整理。用户和 AI 路由直接注册到应用，Agent 路由使用 `/agent` 前缀。

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| POST | `/user` | 创建用户 |
| GET | `/users` | 查询用户列表 |
| GET | `/user/{user_id}` | 查询指定用户 |
| PUT | `/user/{user_id}` | 更新指定用户 |
| DELETE | `/user/{user_id}` | 删除指定用户 |
| POST | `/chat` | 普通 AI/RAG 聊天 |
| POST | `/chat/stream` | 流式聊天 |
| GET | `/chat/{session_id}/messages` | 查询指定会话的聊天历史 |
| POST | `/agent/chat` | Agent 对话与工具调用 |

可以打开 `/docs` 查看请求模型、参数约束、响应结构并进行接口调试。具体行为以当前代码和实际运行结果为准。

### 流式接口说明

`POST /chat/stream` 使用 `application/x-ndjson` 返回逐行 JSON 数据，数据行包含内容、完成状态或错误信息。客户端应逐行读取和解析，不能简单地把整个响应当成单个 JSON 对象解析。

### RAG 拒答说明

当检索结果没有满足当前证据筛选条件时，普通聊天接口会返回拒答结果，并使用 `rag-refusal` 标识该路径。该机制可以减少无关问题触发模型生成，但不能保证所有错误回答都被识别。

## 七、测试与评测

项目此前的测试记录包括：

- 全量自动化测试：22 passed，1 warning。
- Agent 工具与服务相关测试：11 项通过。
- RAG 本地评测集：12 个用例。
- RAG 参数实验：比较过不同文本分块大小、Top-K 和阈值设置。

以上是已有测试记录，并不代表这些结果已在所有环境中重复验证，也不意味着系统对所有真实问题都具有相同表现。

在项目根目录中，可以针对自动化测试目录执行：

```bash
pytest tests/ -q
```

如果测试结果与历史记录不同，应检查依赖版本、环境配置和具体失败用例，并以本次实际测试输出为准。

RAG 评测应区分检索命中、证据筛选和最终回答质量。有限的本地用例只能用于观察已测试场景下的表现，不能直接推导为通用准确率。

## 八、安全设计

- 使用环境变量管理数据库和模型服务配置。
- 将 `.env` 排除在版本控制之外。
- 使用 Pydantic 对请求参数进行校验。
- 通过工具白名单限制 Agent 可调用的功能。
- 对数据库查询采用预定义操作和参数校验，避免模型直接执行任意 SQL。
- 限制 Agent 工具调用轮数。
- 对外返回受控错误信息，避免直接泄露底层异常详情。

这些措施构成了项目的基础安全设计，并不等同于全面的生产环境安全审计。实际部署仍需根据风险补充访问认证、权限管理、日志脱敏、请求频率限制和网络安全配置等措施。

## 九、当前局限与后续改进

当前项目重点是打通 Python 后端、数据库、AI 服务、RAG 检索和 Agent 工具调用的主要流程。后续可以根据实际需求逐步完善：

- 验证全新环境中的依赖安装与项目启动流程。
- 完善结构化日志、异常追踪和配置管理。
- 增加 API 集成测试、边界测试和异常路径测试。
- 扩大 RAG 评测集，记录失败案例并分析检索与回答质量。
- 补充 Docker 部署和自动化测试流程。
- 完善接口认证、访问权限和生产环境配置。
- 根据实习岗位需求持续优化项目交付质量，而不是单纯堆叠技术组件。

## 十、项目总结

本项目围绕 Python 后端与 AI 应用开发，实践了从 HTTP 请求处理、数据库持久化，到 RAG 检索增强生成、Agent 工具调用和自动化测试的完整开发流程。

项目关注的不只是功能实现，也包括模块职责划分、数据库访问管理、工具权限控制、关键行为验证和系统局限分析。后续将继续通过可复现的测试、规范的工程文档和逐步完善的部署流程提升项目的可维护性与可交付性。

## Docker 部署（Linux）

### 环境要求

- Linux 系统
- Docker Engine
- Docker Compose 插件
- 已配置项目所需的 `.env` 文件
- MySQL 已启动，且数据库和用户配置正确

### 启动项目

在项目根目录执行：

```bash
# 检查 Compose 配置
sudo docker compose config -q

# 构建并启动服务
sudo docker compose up -d --build

# 查看容器状态
sudo docker compose ps

# 查看运行日志
sudo docker compose logs --tail 100 app
```

### 验证接口

```bash
# 健康检查对应的根路径
curl http://127.0.0.1:8000/

# 用户列表接口（需要 MySQL）
curl http://127.0.0.1:8000/users
```

浏览器访问：

- API 文档：`http://<服务器IP>:8000/docs`
- 根路径：`http://<服务器IP>:8000/`

### 常用操作

```bash
# 查看日志
sudo docker compose logs -f app

# 重启服务
sudo docker compose restart app

# 停止并移除 Compose 管理的容器
sudo docker compose down
```

### 配置说明

项目通过 `.env` 读取数据库连接参数和模型 API 密钥。不要将 `.env` 提交到 Git，也不要把密钥写入 Dockerfile。

当前部署使用 Linux 的 `network_mode: host`，以便应用访问宿主机本地 MySQL。该网络配置适用于当前 Linux 部署环境，不应直接照搬到 Windows 或 macOS。
