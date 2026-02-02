"""
📖 eRisk T2 测试系统 - 代码阅读指南
===========================================

这个指南帮你理解我构建的完整测试系统。按顺序阅读，你会逐步理解整个架构。

📁 核心文件阅读顺序
===================

1️⃣ 【配置理解】config.py
   - 🔍 重点看: 行1-50 (服务器配置、团队token)
   - 🔍 重点看: 行70-90 (模型参数、特征列表)
   - 💡 作用: 了解系统的基本配置和参数

2️⃣ 【快速测试】quick_test.py (你当前的文件)
   - 🔍 重点看: test_feature_extraction()函数
   - 💡 作用: 验证所有特征提取器是否正常工作
   - 💡 运行: python quick_test.py

3️⃣ 【核心系统】test.py (主要文件，按部分阅读)
   
   📍 Part A: 特征提取系统 (行200-700)
   ==========================================
   
   🏗️ UserHistoryManager (行200-260)
   - 管理用户历史数据，用于时序特征计算
   - extract_liwc = 计算LIWC特征的地方
   
   🧠 PHQ9Extractor (行267-350) ⭐⭐⭐
   - 💎 最核心的特征！基于extremism模块
   - 使用 mixedbread-ai/mxbai-embed-large-v1 模型
   - 计算与9个抑郁症archetype的相似度
   - 输出31维特
   
   📊 CosineSimilarityExtractor (行354-395)
   - 计算连续文本的语义相似度
   - 使用 all-MiniLM-L6-v2 模型
   - 输出6维特征
   
   📝 LSMExtractor (行399-485) ⭐⭐
   - 语言风格匹配，基于LIWC分析
   - 从data-processing.ipynb提取的逻辑
   - 输出60维特征
   
   🔬 REExtractor (行489-570)
   - 相对熵特征，使用抑郁症检测模型
   - 使用train_Dep0_LM和train_Dep1_LM
   - 输出8维特征
   
   🔧 IntegratedFeatureExtractor (行574-650) ⭐⭐⭐
   - 🚀 将所有特征整合的核心类
   - extract_from_eRisk_data() - 处理API数据
   - extract_features() - 提取105维特征向量
   
   📍 Part B: 模型系统 (行100-200)
   =====================================
   
   🏛️ TransformerUserClassifier (行100-170)
   - Transformer模型架构定义
   - 处理变长序列的用户消息
   
   🔀 FeatureAdapterModel (行175-200)
   - 解决105维→99维的兼容性问题
   - 让新特征系统兼容旧模型
   
   📍 Part C: API通信 (行700-850)
   ===============================
   
   🌐 ERiskAPI (行700-780)
   - 与eRisk服务器的通信逻辑
   - get_discussions() - 获取用户数据
   - submit_decisions() - 提交预测结果
   
   👤 ERiskClient (行785-850)
   - 主要的测试客户端类
   - 整合模型、特征提取、API通信

🔍 关键理解点
=============

💡 特征维度分布:
   PHQ-9:    0-30   (31维) - 抑郁症相关特征
   CosSim:   31-36  (6维)  - 语义相似度
   LSM:      37-96  (60维) - 语言风格匹配  
   RE:       97-104 (8维)  - 相对熵特征
   总计:     105维

💡 数据流程:
   eRisk API → 原始JSON → extract_from_eRisk_data() → 
   105维特征 → FeatureAdapter → 99维 → 
   TransformerModel → 风险评分 → 决策提交

💡 时序特征:
   系统会跟踪用户历史，计算时序统计如:
   - max_so_far: 历史最大值
   - min_so_far: 历史最小值  
   - delta: 变化量
   - rolling_std: 滚动标准差

🎯 重要的外部依赖
=================

📁 extremism/item-scoring/
   - item_scoring_sim.py: PHQ-9相似度计算
   - PHQ-9_archetype_scale.txt: 9个抑郁症描述

📁 models/  
   - train_Dep0_LM/: 非抑郁症语言模型
   - train_Dep1_LM/: 抑郁症语言模型
   - best_model.pt: 训练好的分类器

📄 liwc.py
   - LIWC词汇特征提取
   - 66个心理语言学类别

🚀 如何运行系统
===============

1. 测试特征提取:
   python quick_test.py

2. 运行完整系统:  
   python test.py

3. 查看配置:
   编辑 config.py

💡 调试建议
===========

1. 先运行quick_test.py确保特征提取正常
2. 查看日志文件 erisk_test.log
3. 关注GPU/CPU使用情况
4. 检查网络连接到eRisk服务器

🎓 学习建议
===========

1. 从quick_test.py开始理解基本流程
2. 重点学习IntegratedFeatureExtractor类
3. 理解每个特征提取器的作用
4. 查看实际的特征输出数据
5. 分析时序特征的计算逻辑

"""