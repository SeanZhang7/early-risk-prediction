
import requests
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import sys
import time
from typing import Dict, List, Tuple
from datetime import datetime
import logging
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 添加相关模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'extremism', 'item-scoring'))

# 从config导入所有配置
from config import (
    TEAM_TOKEN, NUM_RUNS, DECISION_THRESHOLD, 
    MAX_RETRIES, RETRY_DELAY, REQUEST_TIMEOUT, ROUND_INTERVAL,
    ACTIVE_SERVER, MODEL_PATH, MODEL_CONFIG, FEATURE_COLS,
    RUN_STRATEGIES
)

# ============================================================================
# 配置日志
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('erisk_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入LIWC模块
try:
    from liwc import extract_liwc, liwc_keys
except ImportError:
    logger.warning("LIWC module not found, using fallback implementation")
    liwc_keys = ['ACHIEV', 'ADJ', 'ADVERB', 'AFFECT', 'AFFILIATION', 'ANGER', 'ANX', 'ARTICLE', 
                'AUXVERB', 'BIO', 'BODY', 'CAUSE', 'CERTAIN', 'COGMECH', 'COMPARE', 'CONJ', 
                'DEATH', 'DIFFER', 'DISCREP', 'EXCL', 'FAMILY', 'FEEL', 'FEMALE', 'FILLER', 
                'FRIEND', 'FUTURE', 'HEALTH', 'HEAR', 'HOME', 'HUMANS', 'I', 'INCL', 'INHIBIT', 
                'INSIGHT', 'JOB', 'LEISURE', 'MALE', 'MONEY', 'MOTION', 'NEGATE', 'NONFLU', 
                'NUMBER', 'OTHER', 'OVER', 'PAST', 'PERCEPT', 'POSEMO', 'POSFEEL', 'PRESENT', 
                'PREPS', 'PRONOUN', 'QUANT', 'RELATIV', 'RELIG', 'SAD', 'SCHOOL', 'SEE', 'SELF', 
                'SEXUAL', 'SHEHE', 'SLEEP', 'SOCIAL', 'SPACE', 'SWEAR', 'TENTAT', 'TIME', 
                'TV', 'UP', 'WE', 'WORK', 'YOU', 'NEGEMO']
    
    def extract_liwc(text):
        # Fallback implementation returns zeros
        return {key: 0.0 for key in liwc_keys}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('erisk_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# 使用config中的配置
# ============================================================================

BASE_URL = ACTIVE_SERVER


# ============================================================================
# Transformer模型定义（来自Final_model）
# ============================================================================

class TransformerUserClassifier(nn.Module):
    """
    用户级别的Transformer分类器
    处理变长的用户消息序列
    """
    def __init__(self, input_dim, hidden_dim=None, n_heads=None, n_layers=None, dropout=None):
        super().__init__()
        # 如果未指定，使用config中的默认值
        if hidden_dim is None:
            hidden_dim = MODEL_CONFIG['hidden_dim']
        if n_heads is None:
            n_heads = MODEL_CONFIG['n_heads']
        if n_layers is None:
            n_layers = MODEL_CONFIG['n_layers']
        if dropout is None:
            dropout = MODEL_CONFIG['dropout']
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            attention_mask: [batch_size, seq_len]
        """
        x = self.input_proj(x)
        
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer编码
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 平均池化
        if attention_mask is not None:
            out_masked = out * attention_mask.unsqueeze(-1)
            valid_len = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = out_masked.sum(dim=1) / valid_len
        else:
            pooled = out.mean(dim=1)
        
        # 分类
        logits = self.cls_head(pooled).squeeze(1)
        return logits


class FeatureAdapterModel(nn.Module):
    """
    特征适配器模型：将105维特征映射到99维，然后传入原始模型
    """
    
    def __init__(self, base_model: TransformerUserClassifier, input_dim: int = 105, target_dim: int = 99):
        super().__init__()
        self.base_model = base_model
        
        # 特征降维映射层
        # 简单的线性映射，保留最重要的特征
        self.feature_adapter = nn.Linear(input_dim, target_dim, bias=False)
        
        # 初始化适配器权重：前99维直接传递，后6维被丢弃
        with torch.no_grad():
            # 创建单位矩阵的前99列
            identity_mapping = torch.zeros(target_dim, input_dim)
            identity_mapping[:target_dim, :target_dim] = torch.eye(target_dim)
            self.feature_adapter.weight.data = identity_mapping
        
        logger.info(f"FeatureAdapter: {input_dim} -> {target_dim} dims")
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: [batch_size, seq_len, 105] or [batch_size, 105]
            attention_mask: [batch_size, seq_len] or [batch_size]
        """
        # 适配特征维度
        if x.dim() == 3:  # [batch_size, seq_len, 105]
            batch_size, seq_len, _ = x.shape
            x_adapted = self.feature_adapter(x.view(-1, x.shape[-1])).view(batch_size, seq_len, -1)
        else:  # [batch_size, 105]
            x_adapted = self.feature_adapter(x)
        
        # 传递给基础模型
        return self.base_model(x_adapted, attention_mask)


# ============================================================================
# 集成特征提取器
# ============================================================================

class UserHistoryManager:
    """管理用户历史数据用于时序特征计算"""
    
    def __init__(self):
        self.user_data = defaultdict(lambda: {
            'texts': [],
            'features': {
                'phq9_scores': [],
                'lsm_features': [],
                're_values': [],
                'cos_sim_values': [],
                'timestamps': []
            }
        })
    
    def add_text(self, user_id: str, text: str, timestamp: str = None):
        """添加新文本到用户历史"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        self.user_data[user_id]['texts'].append(text)
        self.user_data[user_id]['features']['timestamps'].append(timestamp)
    
    def get_user_texts(self, user_id: str) -> List[str]:
        """获取用户所有历史文本"""
        return self.user_data[user_id]['texts']
    
    def add_feature_history(self, user_id: str, feature_type: str, value):
        """添加特征历史"""
        if feature_type not in self.user_data[user_id]['features']:
            self.user_data[user_id]['features'][feature_type] = []
        self.user_data[user_id]['features'][feature_type].append(value)
    
    def get_feature_history(self, user_id: str, feature_type: str) -> List:
        """获取特征历史"""
        return self.user_data[user_id]['features'].get(feature_type, [])
    
    def compute_time_series_stats(self, user_id: str, values: List[float], window_size: int = 5) -> Dict[str, float]:
        """计算时序统计特征"""
        if not values:
            return {
                'max_so_far': 0.0,
                'min_so_far': 0.0,
                'max_gap': 0.0,
                'delta': 0.0,
                'rolling_std': 0.0
            }
        
        values = np.array(values)
        max_so_far = float(np.max(values))
        min_so_far = float(np.min(values))
        max_gap = max_so_far - min_so_far
        delta = float(values[-1] - values[-2]) if len(values) > 1 else 0.0
        rolling_std = float(np.std(values[-window_size:])) if len(values) > 0 else 0.0
        
        return {
            'max_so_far': max_so_far,
            'min_so_far': min_so_far,
            'max_gap': max_gap,
            'delta': delta,
            'rolling_std': rolling_std
        }

class PHQ9Extractor:
    """PHQ-9 特征提取器，使用极化分析方法"""
    
    def __init__(self, device='cpu', model_name='mixedbread-ai/mxbai-embed-large-v1'):
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.device = device
            
            # PHQ-9 archetype scale items (简化版)
            self.scale_items = [
                "Little interest or pleasure in doing things",
                "Feeling down, depressed, or hopeless",
                "Trouble falling or staying asleep",
                "Feeling tired or having little energy",
                "Poor appetite or overeating",
                "Feeling bad about yourself",
                "Trouble concentrating on things",
                "Moving or speaking slowly or being fidgety",
                "Thoughts that you would be better off dead"
            ]
            
            # 预计算archetype embeddings
            self.scale_embeddings = self.model.encode(self.scale_items)
            logger.info(f"PHQ-9 extractor initialized with {len(self.scale_items)} scale items")
            
        except Exception as e:
            logger.warning(f"Failed to initialize PHQ-9 extractor: {e}")
            self.model = None
            self.scale_embeddings = None
    
    def extract(self, text: str, user_id: str, history_manager: UserHistoryManager) -> np.ndarray:
        """提取PHQ-9特征 (31维)"""
        features = np.zeros(31, dtype=np.float32)
        
        if self.model is None or self.scale_embeddings is None:
            return features
        
        try:
            # 计算文本embedding
            text_embedding = self.model.encode(text)
            
            # 计算与9个archetype的相似度 (0-8)
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([text_embedding], self.scale_embeddings)[0]
            features[0:9] = similarities.astype(np.float32)
            
            # 计算总体PHQ-9分数
            phq_score = float(np.mean(similarities))
            
            # 添加到历史
            history_manager.add_feature_history(user_id, 'phq9_scores', phq_score)
            phq_history = history_manager.get_feature_history(user_id, 'phq9_scores')
            
            # 时序统计特征 (9-15: 7维)
            if phq_history:
                stats = history_manager.compute_time_series_stats(user_id, phq_history)
                features[9] = phq_score
                features[10] = stats['max_so_far']
                features[11] = stats['min_so_far']
                features[12] = stats['max_gap']
                features[13] = stats['delta']
                features[14] = stats['rolling_std']
                features[15] = len(phq_history)  # post_index
            
            # 每个archetype的历史最大值 (16-24: 9维)
            for i in range(9):
                arch_history = [sim[i] for sim in [similarities] + [np.array(similarities)] * (len(phq_history) - 1)]
                if arch_history:
                    features[16 + i] = np.max(arch_history)
            
            # 其他统计特征 (25-30: 6维)
            if len(similarities) > 0:
                features[25] = np.mean(similarities)  # mean
                features[26] = np.max(similarities)   # max
                features[27] = np.min(similarities)   # min
                features[28] = np.std(similarities)   # std
                
            # 趋势和波动性
            if len(phq_history) > 1:
                features[29] = phq_history[-1] - phq_history[0]  # trend
                features[30] = np.std(phq_history)  # volatility
            
        except Exception as e:
            logger.warning(f"PHQ-9 extraction failed: {e}")
        
        return features

class CosineSimilarityExtractor:
    """余弦相似度特征提取器"""
    
    def __init__(self, device='cpu', model_name='all-MiniLM-L6-v2'):
        try:
            self.model = SentenceTransformer(model_name, device=device)
            logger.info("Cosine similarity extractor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize cosine similarity extractor: {e}")
            self.model = None
    
    def extract(self, text: str, previous_texts: List[str], user_id: str, history_manager: UserHistoryManager) -> np.ndarray:
        """提取余弦相似度特征 (6维)"""
        features = np.zeros(6, dtype=np.float32)
        
        if self.model is None or not previous_texts:
            return features
        
        try:
            # 计算与最近文本的相似度
            text_embedding = self.model.encode(text)
            prev_embedding = self.model.encode(previous_texts[-1])
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([text_embedding], [prev_embedding])[0][0]
            
            features[0] = similarity
            
            # 时序特征
            history_manager.add_feature_history(user_id, 'cos_sim_values', similarity)
            sim_history = history_manager.get_feature_history(user_id, 'cos_sim_values')
            
            if sim_history:
                stats = history_manager.compute_time_series_stats(user_id, sim_history)
                features[1] = stats['max_so_far']
                features[2] = stats['min_so_far']
                features[3] = stats['max_gap']
                features[4] = stats['delta']
                features[5] = stats['rolling_std']
            
        except Exception as e:
            logger.warning(f"Cosine similarity extraction failed: {e}")
        
        return features

class LSMExtractor:
    """Language Style Matching (LSM) 特征提取器"""
    
    def __init__(self):
        # LIWC categories for LSM computation
        self.liwc_cols = ['ARTICLE', 'AUXVERB', 'CONJ', 'ADVERB', 'PREPS', 
                         'PRONOUN', 'QUANT', 'NEGATE', 'ASSENT', 'NONFLU']
        
        # Grouped categories for LSM dimensions
        self.social_cols = ['SOCIAL', 'FRIEND', 'FAMILY', 'HUMANS', 'AFFECT',
                           'POSEMO', 'POSFEEL', 'OPTIM', 'NEGEMO', 'ANX', 'ANGER', 'SAD', 'INHIB']
        self.positive_emo = ['POSEMO', 'POSFEEL', 'OPTIM']
        self.negative_emo = ['NEGEMO', 'ANX', 'ANGER', 'SAD']
        self.person = ['I', 'WE', 'SELF', 'YOU', 'SHEHE', 'OTHER']
        self.cognitive = ['INSIGHT', 'CAUSE', 'DISCREP', 'TENTAT', 'CERTAIN', 'INHIB']
        self.perceptual = ['SEE', 'HEAR', 'FEEL']
        
        logger.info("LSM extractor initialized")
    
    def extract(self, text: str, parent_text: str, user_id: str, history_manager: UserHistoryManager) -> np.ndarray:
        """提取LSM特征 (60维)"""
        features = np.zeros(60, dtype=np.float32)
        
        if not parent_text:
            return features
        
        try:
            # 提取LIWC特征
            body_liwc = extract_liwc(text)
            parent_liwc = extract_liwc(parent_text)
            
            # 计算LSM for base categories
            EPS = 1e-6
            lsm_values = []
            
            for i, cat in enumerate(self.liwc_cols):
                body_val = body_liwc.get(cat, 0.0)
                parent_val = parent_liwc.get(cat, 0.0)
                lsm = 1 - abs(body_val - parent_val) / (body_val + parent_val + EPS)
                lsm_values.append(lsm)
                features[i] = lsm
            
            # LSM means for grouped categories
            def compute_group_lsm(categories):
                group_lsm = []
                for cat in categories:
                    if cat in liwc_keys:
                        body_val = body_liwc.get(cat, 0.0)
                        parent_val = parent_liwc.get(cat, 0.0)
                        lsm = 1 - abs(body_val - parent_val) / (body_val + parent_val + EPS)
                        group_lsm.append(lsm)
                return np.mean(group_lsm) if group_lsm else 0.0
            
            # Compute grouped LSM features
            features[10] = compute_group_lsm(self.social_cols)      # social_mean
            features[11] = compute_group_lsm(self.positive_emo)     # positive_emo_mean
            features[12] = compute_group_lsm(self.negative_emo)     # negative_emo_mean
            features[13] = compute_group_lsm(self.person)           # person_mean
            features[14] = compute_group_lsm(self.cognitive)        # cognitive_mean
            features[15] = compute_group_lsm(self.perceptual)       # perceptual_mean
            
            # Overall LSM statistics
            features[16] = np.mean(lsm_values) if lsm_values else 0.0  # LSM_mean
            features[17] = np.std(lsm_values) if lsm_values else 0.0   # LSM_std
            features[18] = np.max(lsm_values) if lsm_values else 0.0   # LSM_max
            features[19] = np.min(lsm_values) if lsm_values else 0.0   # LSM_min
            
            # Time series features (20-59: 40 dimensions)
            # Store LSM features in history
            current_lsm = features[0:20].copy()
            history_manager.add_feature_history(user_id, 'lsm_features', current_lsm)
            lsm_history = history_manager.get_feature_history(user_id, 'lsm_features')
            
            # Compute time-series stats for each LSM dimension
            if lsm_history and len(lsm_history) > 0:
                for dim in range(10):  # For first 10 LSM dimensions
                    dim_history = [feat[dim] for feat in lsm_history if len(feat) > dim]
                    if dim_history:
                        stats = history_manager.compute_time_series_stats(user_id, dim_history)
                        base_idx = 20 + dim * 4
                        features[base_idx] = stats['max_so_far']
                        features[base_idx + 1] = stats['min_so_far']
                        features[base_idx + 2] = stats['delta']
                        features[base_idx + 3] = stats['rolling_std']
            
        except Exception as e:
            logger.warning(f"LSM extraction failed: {e}")
        
        return features

class REExtractor:
    """Relative Entropy (RE) 特征提取器"""
    
    def __init__(self, device='cpu'):
        self.device = device
        try:
            # 尝试加载预训练的抑郁症检测模型
            model_dir = "/u50/zhanh279/4Z03/jupyter/models"
            
            self.dep0_model = None
            self.dep1_model = None
            self.tokenizer = None
            
            # 尝试加载模型
            dep0_path = os.path.join(model_dir, "train_Dep0_LM")
            dep1_path = os.path.join(model_dir, "train_Dep1_LM")
            
            if os.path.exists(dep0_path) and os.path.exists(dep1_path):
                self.tokenizer = GPT2Tokenizer.from_pretrained(dep0_path)
                self.dep0_model = GPT2LMHeadModel.from_pretrained(dep0_path).to(device)
                self.dep1_model = GPT2LMHeadModel.from_pretrained(dep1_path).to(device)
                
                self.dep0_model.eval()
                self.dep1_model.eval()
                
                logger.info("RE extractor initialized with depression models")
            else:
                logger.warning("Depression models not found, RE features will be zeros")
                
        except Exception as e:
            logger.warning(f"Failed to initialize RE extractor: {e}")
            self.dep0_model = None
            self.dep1_model = None
            self.tokenizer = None
    
    def calc_loss(self, model, text: str) -> float:
        """计算模型在文本上的损失"""
        if model is None or self.tokenizer is None:
            return 0.0
        
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss.item()
            
            return loss
        except Exception:
            return 0.0
    
    def extract(self, text: str, user_id: str, history_manager: UserHistoryManager) -> np.ndarray:
        """提取RE特征 (8维)"""
        features = np.zeros(8, dtype=np.float32)
        
        try:
            # 计算两个模型的损失
            loss_dep0 = self.calc_loss(self.dep0_model, text)
            loss_dep1 = self.calc_loss(self.dep1_model, text)
            
            features[0] = loss_dep0
            features[1] = loss_dep1
            
            # 计算相对熵
            re_value = loss_dep1 - loss_dep0
            features[2] = re_value
            
            # 时序特征
            history_manager.add_feature_history(user_id, 're_values', re_value)
            re_history = history_manager.get_feature_history(user_id, 're_values')
            
            if re_history:
                stats = history_manager.compute_time_series_stats(user_id, re_history)
                features[3] = stats['max_so_far']  # max_re_so_far
                features[4] = stats['min_so_far']  # min_re_so_far
                features[5] = stats['max_gap']     # max_gap_re
                features[6] = stats['delta']       # delta_re
                features[7] = stats['rolling_std'] # rolling_std_re
            
        except Exception as e:
            logger.warning(f"RE extraction failed: {e}")
        
        return features

class IntegratedFeatureExtractor:
    """集成特征提取器 - 提取所有105维特征"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # 初始化各个特征提取器
        self.history_manager = UserHistoryManager()
        self.phq9_extractor = PHQ9Extractor(device)
        self.cos_sim_extractor = CosineSimilarityExtractor(device)
        self.lsm_extractor = LSMExtractor()
        self.re_extractor = REExtractor(device)
        
        logger.info("Integrated feature extractor initialized")
        
        # 特征维度验证
        self.feature_dims = {
            'phq9': 31,
            'cos_sim': 6,
            'lsm': 60,
            're': 8
        }
        self.total_dims = sum(self.feature_dims.values())  # 105
        logger.info(f"Total feature dimensions: {self.total_dims}")
    
    def extract_from_eRisk_data(self, json_data: List[Dict], target_user_id: str) -> np.ndarray:
        """
        从eRisk API数据中提取特征
        
        Args:
            json_data: eRisk API返回的JSON数据
            target_user_id: 目标用户ID
        
        Returns:
            features: shape (n_texts, 105)
        """
        all_features = []
        user_texts = []
        
        for thread in json_data:
            if thread.get('targetSubject') != target_user_id:
                continue
            
            # 处理submission
            submission = thread.get('submission', {})
            if submission.get('author') == target_user_id:
                text = submission.get('body', '')
                if text.strip():
                    user_texts.append(text)
                    self.history_manager.add_text(target_user_id, text, submission.get('created_utc', ''))
            
            # 处理comments
            comments = thread.get('comments', [])
            for comment in comments:
                if comment.get('author') == target_user_id:
                    text = comment.get('body', '')
                    if text.strip():
                        # 找到parent text用于LSM
                        parent_text = ""
                        parent_id = comment.get('parent_id', '')
                        
                        # 在submission或其他comments中找parent
                        if parent_id:
                            if submission.get('id') == parent_id:
                                parent_text = submission.get('body', '')
                            else:
                                for c in comments:
                                    if c.get('id') == parent_id:
                                        parent_text = c.get('body', '')
                                        break
                        
                        user_texts.append(text)
                        self.history_manager.add_text(target_user_id, text, comment.get('created_utc', ''))
                        
                        # 如果找不到parent，使用最后一个文本作为parent
                        if not parent_text and len(user_texts) > 1:
                            parent_text = user_texts[-2]
                        
                        # 提取特征
                        features = self.extract_features(text, target_user_id, parent_text, user_texts[:-1])
                        all_features.append(features)
        
        if not all_features:
            # 如果没有找到用户文本，返回零特征
            return np.zeros((1, self.total_dims), dtype=np.float32)
        
        return np.array(all_features, dtype=np.float32)
    
    def extract_features(self, text: str, user_id: str, parent_text: str = "", previous_texts: List[str] = None) -> np.ndarray:
        """提取单个文本的所有特征"""
        if previous_texts is None:
            previous_texts = []
        
        features = np.zeros(self.total_dims, dtype=np.float32)
        
        try:
            # PHQ-9 features (0-30: 31 dims)
            phq9_features = self.phq9_extractor.extract(text, user_id, self.history_manager)
            features[0:31] = phq9_features
            
            # Cosine similarity features (31-36: 6 dims)
            cos_sim_features = self.cos_sim_extractor.extract(text, previous_texts, user_id, self.history_manager)
            features[31:37] = cos_sim_features
            
            # LSM features (37-96: 60 dims)
            lsm_features = self.lsm_extractor.extract(text, parent_text, user_id, self.history_manager)
            features[37:97] = lsm_features
            
            # RE features (97-104: 8 dims)
            re_features = self.re_extractor.extract(text, user_id, self.history_manager)
            features[97:105] = re_features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for user {user_id}: {e}")
        
        return features

# ============================================================================
# 替换原来的FeatureExtractor类
# ============================================================================

class FeatureExtractor:
    """
    集成的特征提取器，提供与原API兼容的接口
    """
    
    def __init__(self, device='cpu'):
        self.integrated_extractor = IntegratedFeatureExtractor(device)
        self.feature_dim = 105  # 总特征维度
        logger.info(f"FeatureExtractor initialized with {self.feature_dim} dimensions")
    
    def extract_from_texts(self, texts: List[str], user_id: str = "unknown_user") -> np.ndarray:
        """
        从文本列表提取特征 (兼容性方法)
        
        Args:
            texts: 文本列表
            user_id: 用户ID
        
        Returns:
            features: shape (num_messages, 105)
        """
        if not texts:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        
        all_features = []
        
        for i, text in enumerate(texts):
            previous_texts = texts[:i] if i > 0 else []
            parent_text = texts[i-1] if i > 0 else ""
            
            features = self.integrated_extractor.extract_features(text, user_id, parent_text, previous_texts)
            all_features.append(features)
        
        return np.array(all_features, dtype=np.float32)
    
    def extract_from_eRisk_data(self, json_data: List[Dict], target_user_id: str) -> np.ndarray:
        """直接处理eRisk API数据"""
        return self.integrated_extractor.extract_from_eRisk_data(json_data, target_user_id)


# ============================================================================
# API通信模块
# ============================================================================

class ERiskAPI:
    """
    与eRisk服务器通信
    """
    
    def __init__(self, team_token: str, base_url: str = BASE_URL):
        self.team_token = team_token
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_discussions(self, retry_count=0) -> List[Dict]:
        """
        GET请求获取讨论（用户消息）
        
        Returns:
            讨论列表，或None表示错误
        """
        url = f"{self.base_url}/getdiscussions/{self.team_token}"
        
        try:
            logger.info(f"[GET] 请求讨论: {url}")
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✓ 获得 {len(data)} 个讨论")
                return data
            else:
                logger.error(f"✗ GET失败: {response.status_code}")
                if retry_count < MAX_RETRIES:
                    time.sleep(RETRY_DELAY ** retry_count)
                    return self.get_discussions(retry_count + 1)
                return None
        
        except Exception as e:
            logger.error(f"✗ 网络错误: {str(e)}")
            if retry_count < MAX_RETRIES:
                time.sleep(RETRY_DELAY ** retry_count)
                return self.get_discussions(retry_count + 1)
            return None
    
    def submit_decisions(self, run_id: int, decisions: List[Dict], retry_count=0) -> bool:
        """
        POST请求提交决策
        
        Args:
            run_id: 运行编号 (0-4)
            decisions: 决策列表
        
        Returns:
            成功与否
        """
        url = f"{self.base_url}/submit/{self.team_token}/{run_id}"
        
        try:
            logger.info(f"[POST] 提交Run {run_id}: {len(decisions)} 个用户决策")
            response = self.session.post(
                url,
                json=decisions,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                logger.info(f"✓ Run {run_id} 提交成功")
                return True
            else:
                logger.error(f"✗ POST失败 (Run {run_id}): {response.status_code}")
                if retry_count < MAX_RETRIES:
                    time.sleep(RETRY_DELAY ** retry_count)
                    return self.submit_decisions(run_id, decisions, retry_count + 1)
                return False
        
        except Exception as e:
            logger.error(f"✗ 网络错误 (Run {run_id}): {str(e)}")
            if retry_count < MAX_RETRIES:
                time.sleep(RETRY_DELAY ** retry_count)
                return self.submit_decisions(run_id, decisions, retry_count + 1)
            return False


# ============================================================================
# 主客户端类
# ============================================================================

class ERiskClient:
    """
    eRisk T2 测试客户端
    """
    
    def __init__(self, model_path: str, team_token: str, num_runs: int = NUM_RUNS):
        """
        初始化客户端
        
        Args:
            model_path: 训练好的模型路径
            team_token: 团队token
            num_runs: 运行数量 (1-5)
        """
        self.team_token = team_token
        self.num_runs = num_runs
        self.model_path = model_path
        
        # 初始化API
        self.api = ERiskAPI(team_token)
        
        # 初始化设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 先初始化特征提取器（模型加载需要知道特征维度）
        self.feature_extractor = FeatureExtractor(device=self.device)
        
        # 然后初始化模型
        self.model = self._load_model()
        
        # 用户数据管理
        self.user_messages: Dict[str, List[str]] = {}  # 用户的所有消息
        self.user_alerts: Dict[str, int] = {}          # 已警报用户 + 警报轮数
        self.user_scores: Dict[str, List[float]] = {}  # 用户的历史评分
        
        # 统计
        self.total_rounds = 0
        self.total_users = 0
        self.alerts_fired = 0
        
        logger.info(f"客户端初始化完成 (Device: {self.device}, Runs: {num_runs})")
    
    def _load_model(self) -> TransformerUserClassifier:
        """加载Transformer模型"""
        try:
            logger.info(f"加载模型: {self.model_path}")
            
            if os.path.exists(self.model_path):
                # 先加载现有模型检查结构
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
                
                # 检查输入维度
                input_proj_weight = state_dict.get('input_proj.weight')
                if input_proj_weight is not None:
                    old_input_dim = input_proj_weight.shape[1]
                    logger.info(f"检测到旧模型输入维度: {old_input_dim}")
                    
                    if old_input_dim == 99 and self.feature_extractor.feature_dim == 105:
                        # 需要适配：从105维映射到99维
                        logger.info("创建105->99维特征映射适配器")
                        
                        # 创建99维模型（与旧模型兼容）
                        model = TransformerUserClassifier(
                            input_dim=99,
                            hidden_dim=MODEL_CONFIG['hidden_dim'],
                            n_heads=MODEL_CONFIG['n_heads'],
                            n_layers=MODEL_CONFIG['n_layers'],
                            dropout=MODEL_CONFIG['dropout']
                        )
                        
                        # 修改分类头结构以匹配旧模型
                        model.cls_head = nn.Sequential(
                            nn.Linear(MODEL_CONFIG['hidden_dim'], 64),
                            nn.ReLU(),
                            nn.Linear(64, 1)  # 直接输出，没有dropout层
                        )
                        
                        # 加载旧模型权重
                        model.load_state_dict(state_dict)
                        logger.info("✓ 旧模型权重加载成功")
                        
                        # 包装模型以处理105维输入
                        model = FeatureAdapterModel(model, input_dim=105, target_dim=99)
                        logger.info("✓ 特征适配器创建成功")
                        
                    else:
                        # 正常加载
                        model = TransformerUserClassifier(
                            input_dim=old_input_dim,
                            hidden_dim=MODEL_CONFIG['hidden_dim'],
                            n_heads=MODEL_CONFIG['n_heads'],
                            n_layers=MODEL_CONFIG['n_layers'],
                            dropout=MODEL_CONFIG['dropout']
                        )
                        model.load_state_dict(state_dict)
                        logger.info("✓ 模型加载成功")
                else:
                    # 创建新模型
                    logger.warning("⚠️ 无法检测模型输入维度，创建新模型")
                    model = TransformerUserClassifier(
                        input_dim=105,
                        hidden_dim=MODEL_CONFIG['hidden_dim'],
                        n_heads=MODEL_CONFIG['n_heads'],
                        n_layers=MODEL_CONFIG['n_layers'],
                        dropout=MODEL_CONFIG['dropout']
                    )
            else:
                logger.warning(f"⚠️ 模型文件不存在: {self.model_path}，使用未训练的模型")
                model = TransformerUserClassifier(
                    input_dim=105,
                    hidden_dim=MODEL_CONFIG['hidden_dim'],
                    n_heads=MODEL_CONFIG['n_heads'],
                    n_layers=MODEL_CONFIG['n_layers'],
                    dropout=MODEL_CONFIG['dropout']
                )
            
            model.to(self.device)
            model.eval()
            
            return model
        
        except Exception as e:
            logger.error(f"✗ 加载模型失败: {str(e)}")
            raise
    
    def _extract_user_texts(self, thread: Dict) -> List[str]:
        """
        从讨论中提取目标用户的所有消息
        """
        target_user = thread.get('targetSubject')
        texts = []
        
        # 检查提交中的内容
        submission = thread.get('submission', {})
        if submission.get('author') == target_user:
            body = submission.get('body', '')
            if body:
                texts.append(body)
        
        # 检查评论中的内容
        comments = thread.get('comments', [])
        for comment in comments:
            if comment.get('author') == target_user:
                body = comment.get('body', '')
                if body:
                    texts.append(body)
        
        return texts
    
    def _predict_user_risk(self, user_id: str, discussions: List[Dict] = None) -> Tuple[float, float]:
        """
        预测用户的风险评分
        
        Args:
            user_id: 用户ID
            discussions: eRisk API数据 (可选，如果提供则直接使用)
        
        Returns:
            (评分0-1, 决策0/1)
        """
        try:
            if discussions is not None:
                # 使用eRisk API数据直接提取特征
                features = self.feature_extractor.extract_from_eRisk_data(discussions, user_id)
                logger.debug(f"从API数据提取特征: 用户 {user_id}, shape: {features.shape}")
            else:
                # 使用传统方法从用户消息提取特征
                if user_id not in self.user_messages:
                    logger.warning(f"用户 {user_id} 无消息记录")
                    return 0.5, 0
                
                texts = self.user_messages[user_id]
                features = self.feature_extractor.extract_from_texts(texts, user_id)
                logger.debug(f"从文本提取特征: 用户 {user_id}, shape: {features.shape}")
            
            if features.shape[0] == 0:
                logger.warning(f"用户 {user_id} 无有效特征")
                return 0.5, 0
            
            # 使用最新的特征进行预测
            latest_features = features[-1:] if len(features.shape) > 1 else features.reshape(1, -1)
            
            # 转为张量
            features_tensor = torch.from_numpy(latest_features).float().to(self.device)
            
            # 为单个样本序列创建正确的attention mask
            seq_len = 1  # 我们只使用最新的一个特征向量
            mask = torch.ones(1, seq_len, dtype=torch.long).to(self.device)  # [batch_size, seq_len]
            
            # 特征需要添加序列维度: [batch_size, seq_len, feature_dim]
            if features_tensor.dim() == 2:  # [1, feature_dim]
                features_tensor = features_tensor.unsqueeze(1)  # [1, 1, feature_dim]
            
            # 推理
            with torch.no_grad():
                logits = self.model(features_tensor, mask)
                score = torch.sigmoid(logits).squeeze().item()
            
            # 保存历史评分
            if user_id not in self.user_scores:
                self.user_scores[user_id] = []
            self.user_scores[user_id].append(score)
            
            logger.debug(f"用户 {user_id} 评分: {score:.4f}")
            
            return score, float(score)
        
        except Exception as e:
            logger.error(f"✗ 预测失败 (用户 {user_id}): {str(e)}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return 0.5, 0
    
    def _make_decision(self, user_id: str, score: float) -> int:
        """
        基于评分做出决策
        
        Args:
            user_id: 用户ID
            score: 风险评分
        
        Returns:
            决策 (0或1)
        """
        # 如果已经警报过，继续返回0（最终决策已做）
        if user_id in self.user_alerts:
            return 0
        
        # 基于阈值做决策
        if score > DECISION_THRESHOLD:
            logger.info(f"⚠️ 警报！用户 {user_id} 风险评分: {score:.4f}")
            self.user_alerts[user_id] = self.total_rounds
            self.alerts_fired += 1
            return 1
        
        return 0
    
    def process_round(self) -> Tuple[List[Dict], bool]:
        """
        处理一轮讨论
        
        Returns:
            (决策列表, 是否继续)
        """
        # 获取讨论
        discussions = self.api.get_discussions()
        
        if discussions is None:
            logger.error("✗ 无法获取讨论，中止")
            return [], False
        
        if not discussions:
            logger.info("✓ 空列表，所有数据已处理")
            return [], False
        
        self.total_rounds += 1
        logger.info(f"\n{'='*70}")
        logger.info(f"第 {self.total_rounds} 轮 - 处理 {len(discussions)} 个讨论")
        logger.info(f"{'='*70}")
        
        # 处理每个讨论
        predictions = []
        
        for thread in discussions:
            target_user = thread.get('targetSubject')
            
            # 维护用户消息历史
            if target_user not in self.user_messages:
                self.user_messages[target_user] = []
                self.total_users += 1
            
            # 添加新消息
            new_texts = self._extract_user_texts(thread)
            self.user_messages[target_user].extend(new_texts)
            
            logger.info(f"用户 {target_user}: +{len(new_texts)} 条消息 (总计: {len(self.user_messages[target_user])})")
            
            # 预测 (直接使用API数据进行特征提取)
            score, _ = self._predict_user_risk(target_user, discussions=[thread])
            
            # 决策
            decision = self._make_decision(target_user, score)
            
            # 记录预测
            predictions.append({
                'nick': target_user,
                'decision': decision,
                'score': score
            })
        
        logger.info(f"本轮处理完成: {len(predictions)} 用户, {self.alerts_fired} 个警报")
        
        return predictions, True
    
    def run(self):
        """
        主循环 - 持续处理轮数直到完成
        """
        logger.info(f"\n🚀 启动eRisk T2客户端")
        logger.info(f"Token: {self.team_token}")
        logger.info(f"Runs: {self.num_runs}")
        logger.info(f"Model: {self.model_path}")
        
        start_time = datetime.now()
        
        try:
            while True:
                # 处理一轮
                predictions, continue_flag = self.process_round()
                
                if not continue_flag:
                    break
                
                # 提交所有runs的决策
                all_submitted = True
                for run_id in range(self.num_runs):
                    success = self.api.submit_decisions(run_id, predictions)
                    if not success:
                        all_submitted = False
                        logger.error(f"✗ Run {run_id} 提交失败")
                
                if not all_submitted:
                    logger.error("✗ 某些runs提交失败，中止")
                    break
                
                # 等待下一轮
                logger.info("等待下一轮...")
                time.sleep(2)
        
        except KeyboardInterrupt:
            logger.warning("用户中断")
        
        except Exception as e:
            logger.error(f"✗ 异常错误: {str(e)}")
        
        finally:
            # 统计信息
            elapsed_time = datetime.now() - start_time
            logger.info(f"\n{'='*70}")
            logger.info(f"测试完成")
            logger.info(f"{'='*70}")
            logger.info(f"总轮数: {self.total_rounds}")
            logger.info(f"总用户: {self.total_users}")
            logger.info(f"警报数: {self.alerts_fired}")
            logger.info(f"用时: {elapsed_time}")
            logger.info(f"平均轮处理时间: {elapsed_time.total_seconds() / max(self.total_rounds, 1):.2f}s")


# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """
    主程序
    """
    # 使用config.py中的配置
    
    # 检查token
    if TEAM_TOKEN == "YOUR_TEAM_TOKEN":
        logger.error("❌ 请先在config.py中设置TEAM_TOKEN")
        return
    
    # 创建客户端
    client = ERiskClient(
        model_path=MODEL_PATH,
        team_token=TEAM_TOKEN,
        num_runs=NUM_RUNS
    )
    
    # 运行
    client.run()


if __name__ == "__main__":
    main()
