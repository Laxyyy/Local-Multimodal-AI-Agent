🧠 Local Multimodal AI Agent (本地多模态 AI 智能助手)

📖 项目简介 (Introduction)

本项目是基于 RAG (检索增强生成) 和 多模态深度学习 技术构建的本地智能助手。旨在解决传统的“基于文件名”搜索的局限性，通过向量化技术实现对本地论文库和图像库的语义理解、自动分类以及跨模态检索。

项目完全本地化运行（Privacy-First），无需依赖昂贵的云端 GPU，使用轻量级 SOTA 模型 (all-MiniLM-L6-v2 和 CLIP) 即可在普通 PC/Mac 上流畅运行。

✨ 核心功能 (Core Features)

本项目已完成作业要求的全部核心指标：

1. 📄 智能文献管理 (Document Intelligence)

基于语义的自动分类 (Zero-Shot Classification):

系统不依赖硬编码规则，而是计算论文内容与候选主题（如 EdgeAI, LLM, CV）在向量空间的余弦相似度，自动将论文归档至对应文件夹。

语义检索 (Semantic Search):

支持自然语言提问（例如："How does the agent fix errors?"）。即便文件名中不包含关键词，系统也能通过理解论文摘要的语义找到正确文档。

2. 🖼️ 智能图像管理 (Visual Intelligence)

以文搜图 (Text-to-Image Search):

利用 CLIP 模型对图像和文本进行联合编码。用户可以通过自然语言描述（如 "A cute cat"）直接检索本地未标记的图片。

🛠️ 技术栈 (Tech Stack)

编程语言: Python 3.10

向量数据库: ChromaDB (用于存储和检索高维向量)

文本嵌入模型: sentence-transformers/all-MiniLM-L6-v2 (384维稠密向量)

多模态模型: openai/clip-vit-base-patch32 (用于图文特征对齐)

PDF 处理: pypdf

图像处理: Pillow

⚡ 环境安装 (Installation)

建议使用 Conda 创建独立的虚拟环境以避免依赖冲突。

# 1. 创建环境 (推荐 Python 3.10)
conda create -n ai_agent python=3.10
conda activate ai_agent

# 2. 安装项目依赖
pip install -r requirements.txt


requirements.txt 内容:

chromadb
sentence-transformers
torch
Pillow
pypdf
scikit-learn


🚀 使用指南与实验展示 (Usage & Demo)

以下展示了基于真实论文（AirFL 和 VIGIL）的实验过程。

1. 初始化图片库索引

首先，让 AI "看"一遍本地的 images 文件夹。

python main.py index_images ./images


输出示例: 成功索引 3 张图片。

2. 论文智能分类 (Smart Classification)

我们要测试系统能否区分 联邦学习(Signal Processing) 和 大语言模型(LLM Agent) 两类截然不同的论文。

测试指令 1 (AirFL 论文):

python main.py add_paper ./papers/2512.03719v1.pdf --topics "AirFL,Edge_AI,Signal_Processing,CSIT_Aware"


实验结果: 系统正确识别该论文属于 Edge_AI 领域，并自动创建文件夹移动文件。

测试指令 2 (VIGIL 论文):

python main.py add_paper ./papers/2512.07094v2.pdf --topics "VIGIL,LLM_Agents,Self_Healing_Systems"


实验结果: 系统正确识别该论文属于 Self_Healing_Systems 领域。

[请在此处插入你运行 add_paper 时包含分类结果和文件夹移动的截图]

3. 语义文献检索 (RAG Search)

验证系统是否真正理解了论文内容，而不仅仅是匹配关键词。

场景 A: 询问通信效率机制

问题: "How does AirFL reduce latency and bandwidth consumption?"

指令:

python main.py search_paper "How does AirFL reduce latency and bandwidth consumption?"


结果分析:

Rank 1: 2512.03719v1.pdf (AirFL论文) - Score: 0.3557

Rank 2: 2512.07094v2.pdf (Agent论文) - Score: -0.1366

结论: 系统准确区分了相关与不相关文档（分差显著）。

场景 B: 询问系统内部组件

问题: "What is the function of EmoBank in the VIGIL runtime?" (EmoBank 是该论文特有的概念)

指令:

python main.py search_paper "What is the function of EmoBank in the VIGIL runtime?"


结果分析:

Rank 1: 2512.07094v2.pdf - Score: 0.3924

结论: 即使是非常具体的内部术语，系统也能精准定位。

[请在此处插入 search_paper 终端输出结果的截图]

4. 以文搜图 (Multimodal Search)

测试 CLIP 模型的跨模态能力。

场景: 寻找一张猫的图片。

指令:

python main.py search_image "A cat"


结果:

Rank 1: Picture3.jpeg (真实猫图) - Score: 0.2770

Rank 2: Picture2.jpg (干扰项) - Score: 0.1852

[请在此处插入 search_image 终端输出结果的截图]

📂 项目结构 (Project Structure)

MyAI_Agent/
├── main.py              # 核心逻辑入口 (单文件实现)
├── requirements.txt     # 依赖列表
├── db/                  # ChromaDB 数据库文件 (自动生成)
├── images/              # 图片素材库
└── papers/              # 论文库
    ├── Edge_AI/             # 自动归类生成的文件夹
    │   └── 2512.03719v1.pdf
    └── Self_Healing_Systems/# 自动归类生成的文件夹
        └── 2512.07094v2.pdf
