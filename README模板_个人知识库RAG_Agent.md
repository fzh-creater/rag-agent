# Personal Knowledge Base RAG Agent

## 项目简介

基于 RAG（检索增强生成）技术的个人知识库智能问答助手。支持上传 PDF/Markdown 文档，通过对话式交互精准检索并回答文档内容，附带引用来源溯源。

<br />

## 核心亮点

- **精准检索**：基于向量相似度匹配，解决大模型幻觉问题
- **引用溯源**：回答自动标注来源文档及页码，增强可信度
- **快速部署**：Streamlit 构建，一行命令启动

## 技术架构

| 层级            | 技术选型              |
| ------------- | ----------------- |
| LLM           | Qwen-Max / GLM-4  |
| RAG Framework | LangChain         |
| Vector DB     | FAISS             |
| Frontend      | Streamlit         |
| IDE           | Cursor / Windsurf |

## 快速开始

### 环境要求

- Python 3.9+
- pip 包管理器

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/fzh-creater/rag-agent.git
cd rag-agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
# 复制 .env.example 为 .env 并填入你的 API Key
cp .env.example .env

# 4. 运行应用
streamlit run app.py
```

### 环境变量配置 (.env)

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://your_api_endpoint
MODEL_NAME=qwen-max
```

## 功能演示

![Demo Screenshot](docs/screenshot.png)

> 演示视频：[点击观看](https://your-video-link-here)

## 项目结构

```
rag-agent/
├── app.py              # 主程序入口（Streamlit界面）
├── rag_chain.py        # RAG核心逻辑链
├── vector_store.py     # 向量数据库操作
├── utils.py            # 文档处理工具函数
├── requirements.txt    # 依赖包列表
├── .env.example        # 环境变量模板
├── docs/               # 文档与截图
│   └── screenshot.png
└── README.md           # 项目说明
```

## 核心代码说明

### RAG 检索流程

1. 用户上传文档（PDF/Markdown）
2. 文档被切分为固定长度的片段（Chunk）
3. 每个片段通过 Embedding 模型转为向量
4. 向量存入 FAISS 向量数据库
5. 用户提问时，问题也被转为向量
6. 在向量库中检索最相似的 Top-K 个片段
7. 将检索结果 + 用户问题一起传给 LLM 生成回答
8. 回答中自动标注来源片段引用

### 关键技术决策

- **文档切片策略**：采用递归字符文本分割器（RecursiveCharacterTextSplitter），chunk\_size=1000，chunk\_overlap=200，兼顾上下文完整性与检索精度
- **向量相似度度量**：使用余弦相似度（cosine similarity），在中文场景下表现稳定
- **Prompt 工程**：采用系统提示词 + 检索上下文 + 用户问题的三段式结构，有效抑制幻觉

## 未来规划

- [ ] 支持多轮对话记忆
- [ ] 增加图表/表格解析能力
- [ ] 支持多种文档格式（PPT/Excel）
- [ ] 部署到云端（Vercel/Render）
- [ ] 增加用户权限与知识库管理

## 关于作者

- **姓名**：范召豪
- **学校**：延边大学 · 信息管理与信息系统
- **Email**：<fzh-worker@163.com>
- **GitHub**：[https://github.com/fzh-creater](https://github.com/\[你的用户名)

## 开源协议

MIT License
