#!/usr/bin/env python3
"""
快速测试集成特征提取器
"""
import sys
import os
sys.path.insert(0, '/u50/zhanh279/4Z03/jupyter/Test')

import numpy as np
import torch
import logging
from test import IntegratedFeatureExtractor, FeatureExtractor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_feature_extraction():
    """测试特征提取功能"""
    print("🧪 开始测试特征提取器...")
    
    # 创建特征提取器
    extractor = FeatureExtractor(device='cpu')
    print(f"✅ 特征提取器创建成功 - 输出维度: {extractor.feature_dim}")
    
    # 测试文本
    test_texts = [
        "I feel really depressed and hopeless lately.",
        "Having trouble sleeping and concentrating.",
        "Nothing seems to bring me joy anymore."
    ]
    
    # 提取特征
    print("🔄 提取特征...")
    features = extractor.extract_from_texts(test_texts, "test_user")
    print(f"✅ 特征提取完成 - Shape: {features.shape}")
    
    # 验证特征
    assert features.shape == (3, 99), f"期望形状(3, 99)，得到{features.shape}"
    assert not np.isnan(features).any(), "特征中包含NaN"
    assert np.isfinite(features).all(), "特征中包含无限值"
    
    print("✅ 特征验证通过!")
    
    # 测试eRisk API数据格式
    print("🔄 测试eRisk数据处理...")
    
    erisk_data = [{
        'targetSubject': 'test_user',
        'submission': {
            'author': 'test_user',
            'body': 'I have been feeling really down lately.',
            'id': 'sub123',
            'created_utc': '2026-01-01'
        },
        'comments': [
            {
                'author': 'test_user',
                'body': 'My sleep is terrible and I can\'t focus.',
                'id': 'com456',
                'parent_id': 'sub123',
                'created_utc': '2026-01-02'
            }
        ]
    }]
    
    features_erisk = extractor.extract_from_eRisk_data(erisk_data, 'test_user')
    print(f"✅ eRisk数据处理完成 - Shape: {features_erisk.shape}")
    
    return True

if __name__ == "__main__":
    try:
        test_feature_extraction()
        print("🎉 所有测试通过!")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)