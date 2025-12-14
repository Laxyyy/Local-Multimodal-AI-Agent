import argparse
import os
import shutil
import warnings
from typing import List

# 忽略一些无关紧要的警告
warnings.filterwarnings("ignore")

try:
    import chromadb
    from chromadb.utils import embedding_functions
    from sentence_transformers import SentenceTransformer, util
    from PIL import Image
    from pypdf import PdfReader
except ImportError as e:
    print(f"缺少依赖库: {e}")
    print("请运行: pip install -r requirements.txt")
    exit(1)

# ===========================
# 1. 模型与数据库初始化
# ===========================

class AI_Agent:
    def __init__(self):
        print("正在初始化 AI Agent (加载模型与数据库)...")
        
        # --- 路径配置 ---
        self.db_path = "./db"
        self.image_folder = "./images" # 默认图片库搜索路径
        
        # --- 加载模型 (轻量级配置) ---
        # 1. 纯文本模型 (用于论文搜索与分类)
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. 多模态 CLIP 模型 (用于以文搜图)
        # clip-ViT-B-32 可以同时编码文本和图像到同一个向量空间
        self.clip_model = SentenceTransformer('clip-ViT-B-32')

        # --- 初始化 ChromaDB ---
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # 创建/获取集合 (Collection)
        # collection_papers: 存储论文的文本向量
        self.paper_collection = self.client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"} # 使用余弦相似度
        )
        
        # collection_images: 存储图片的视觉向量
        self.image_collection = self.client.get_or_create_collection(
            name="images",
            metadata={"hnsw:space": "cosine"}
        )
        print("初始化完成！\n")

    # ===========================
    # 2. 核心功能: 论文管理
    # ===========================

    def extract_text_from_pdf(self, pdf_path):
        """简单的 PDF 文本提取，仅提取前2页以节省时间"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            # 仅读取前 2 页的内容作为摘要索引
            for page in reader.pages[:2]: 
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"读取 PDF 失败: {e}")
            return None

    def add_paper(self, file_path, topics=None):
        """添加论文，生成嵌入，并根据 Topics 自动分类"""
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 {file_path}")
            return

        filename = os.path.basename(file_path)
        print(f"正在处理论文: {filename} ...")

        # 1. 提取文本
        text_content = self.extract_text_from_pdf(file_path)
        if not text_content:
            return

        # 2. 生成向量 (Embedding)
        embedding = self.text_model.encode(text_content).tolist()

        # 3. 存入数据库
        self.paper_collection.add(
            documents=[text_content[:500]], # 只存前500字符作为元数据预览
            embeddings=[embedding],
            metadatas=[{"filename": filename, "path": file_path}],
            ids=[filename] # 使用文件名作为ID (需确保唯一)
        )
        print(f"论文 [{filename}] 已建立索引。")

        # 4. 自动分类 (如果提供了 topics)
        if topics:
            topic_list = topics.split(',')
            # 计算论文内容与每个 topic 的相似度
            topic_embeddings = self.text_model.encode(topic_list)
            # 计算相似度 (Cosine Similarity)
            similarities = util.cos_sim(embedding, topic_embeddings)[0]
            
            # 找到最匹配的主题
            best_idx = similarities.argmax()
            best_topic = topic_list[best_idx].strip()
            score = similarities[best_idx]

            print(f"分类分析: 最匹配主题为 '{best_topic}' (置信度: {score:.2f})")

            # 移动文件
            target_dir = os.path.join(os.path.dirname(file_path), best_topic)
            os.makedirs(target_dir, exist_ok=True)
            new_path = os.path.join(target_dir, filename)
            
            # 移动文件操作
            try:
                shutil.move(file_path, new_path)
                print(f"已将文件移动至: {new_path}")
                # 更新数据库中的路径 (可选，为简单起见这里暂略，实际项目需更新 metadata)
            except Exception as e:
                print(f"移动文件失败: {e}")

    def search_paper(self, query):
        """语义搜索论文"""
        print(f"正在搜索: '{query}' ...")
        
        # 1. 将查询转换为向量
        query_embedding = self.text_model.encode(query).tolist()

        # 2. 在 ChromaDB 中搜索
        results = self.paper_collection.query(
            query_embeddings=[query_embedding],
            n_results=3 # 返回前3个结果
        )

        # 3. 展示结果
        print("\n--- 搜索结果 ---")
        if not results['ids'][0]:
            print("未找到相关论文。")
            return

        for i in range(len(results['ids'][0])):
            filename = results['metadatas'][0][i]['filename']
            # path = results['metadatas'][0][i]['path']
            distance = results['distances'][0][i]
            # distance 越小越相似 (Chroma 默认可能是距离，但我们前面设了 cosine，这里要注意)
            # 注：Chroma 的 cosine 距离是 1 - similarity。越小越好。
            print(f"[{i+1}] {filename} (相关性分数: {1-distance:.4f})")
        print("----------------\n")

    # ===========================
    # 3. 核心功能: 图像管理
    # ===========================

    def index_images(self, folder_path):
        """扫描文件夹中的图片并建立索引 (用于 setup 阶段)"""
        if not os.path.exists(folder_path):
            print("图片文件夹不存在。")
            return

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_extensions]
        
        print(f"发现 {len(image_files)} 张图片，开始建立索引...")
        
        ids = []
        embeddings = []
        metadatas = []

        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            try:
                image = Image.open(img_path)
                # 使用 CLIP 生成图片向量
                emb = self.clip_model.encode(image).tolist()
                
                ids.append(img_file)
                embeddings.append(emb)
                metadatas.append({"filename": img_file, "path": img_path})
            except Exception as e:
                print(f"跳过图片 {img_file}: {e}")

        if ids:
            self.image_collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            print(f"成功索引 {len(ids)} 张图片。")

    def search_image(self, query):
        """以文搜图"""
        print(f"正在搜图: '{query}' ...")
        
        # 1. 将文本描述转换为 CLIP 向量
        # CLIP 的神奇之处：文本和图片在同一个向量空间
        query_embedding = self.clip_model.encode(query).tolist()

        # 2. 搜索数据库
        results = self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        # 3. 展示结果
        print("\n--- 搜图结果 ---")
        if not results['ids'][0]:
            print("未找到相关图片 (请先确认是否已将图片放入 images 文件夹并运行索引命令)。")
            return

        for i in range(len(results['ids'][0])):
            filename = results['metadatas'][0][i]['filename']
            distance = results['distances'][0][i]
            print(f"[{i+1}] {filename} (匹配度: {1-distance:.4f})")
        print("----------------\n")

# ===========================
# 4. 命令行入口 (Main)
# ===========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本地多模态 AI 助手")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 命令: add_paper
    parser_add = subparsers.add_parser("add_paper", help="添加并分类论文")
    parser_add.add_argument("path", type=str, help="PDF文件路径")
    parser_add.add_argument("--topics", type=str, default="CV,NLP,RL", help="分类主题，用逗号分隔")

    # 命令: search_paper
    parser_search_p = subparsers.add_parser("search_paper", help="语义搜索论文")
    parser_search_p.add_argument("query", type=str, help="搜索问题")

    # 命令: index_images (辅助命令，用于预先加载图片库)
    parser_idx_img = subparsers.add_parser("index_images", help="索引文件夹下的所有图片")
    parser_idx_img.add_argument("folder", type=str, help="图片文件夹路径")

    # 命令: search_image
    parser_search_i = subparsers.add_parser("search_image", help="以文搜图")
    parser_search_i.add_argument("query", type=str, help="图片描述")

    args = parser.parse_args()
    
    # 实例化 Agent
    agent = AI_Agent()

    if args.command == "add_paper":
        agent.add_paper(args.path, args.topics)
    elif args.command == "search_paper":
        agent.search_paper(args.query)
    elif args.command == "index_images":
        agent.index_images(args.folder)
    elif args.command == "search_image":
        # 搜索前先尝试自动索引默认目录，防止库为空
        # agent.index_images("./images") 
        agent.search_image(args.query)
    else:
        parser.print_help()