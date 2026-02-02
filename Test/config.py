"""
eRisk T2 测试框架 - 配置文件
"""

# ============================================================================
# 服务器配置
# ============================================================================

# 非官方测试服务器（带3个虚拟用户）
UNOFFICIAL_SERVER = "https://erisk.irlab.org/challenge-t2"

# 官方服务器（真实测试）
OFFICIAL_SERVER = "https://erisk.irlab.org/challenge-t2-official"

# 选择使用的服务器
# 改为OFFICIAL_SERVER进行真实测试
ACTIVE_SERVER = UNOFFICIAL_SERVER

# ============================================================================
# 团队配置
# ============================================================================

# 你的团队token（联系 anxo.pvila@udc.es 获取）
TEAM_TOKEN = "NrN7nPNuaxv3nbyRUjYrJh2FvQhvgJEDGvGAt0yLUQM"

# 运行数量 (1-5)
NUM_RUNS = 5

# ============================================================================
# 模型配置
# ============================================================================

# 模型路径（best_model.pt已从Develop复制到此目录）
MODEL_PATH = "best_model.pt"

# Transformer模型参数（需与Final_model.ipynb保持一致）
MODEL_CONFIG = {
    'input_dim': 105,                 # ⭐ 更新: 现在包含余弦相似度特征 (PHQ9:31 + LSM:60 + RE:6 + CosSim:6 + Other:2)
    'hidden_dim': 256,                # 隐层维度
    'n_heads': 4,                     # 注意力头数
    'n_layers': 3,                    # 编码层数
    'dropout': 0.1                    # dropout率
}

# ============================================================================
# 推理配置
# ============================================================================

# 警报阈值（0-1，越高越保守）
DECISION_THRESHOLD = 0.5

# 多轮运行的不同阈值策略
RUN_STRATEGIES = {
    0: {'threshold': 0.3, 'name': '激进（早期识别）'},
    1: {'threshold': 0.5, 'name': '中等（平衡）'},
    2: {'threshold': 0.7, 'name': '保守（低误报）'},
    3: {'threshold': 0.5, 'name': '中等+评分趋势'},
    4: {'threshold': 0.6, 'name': '中等偏激进'}
}

# ============================================================================
# 网络配置
# ============================================================================

# 重试参数
MAX_RETRIES = 5
RETRY_DELAY = 2  # 秒，使用指数退避

# 请求超时
REQUEST_TIMEOUT = 30

# 轮次间隔
ROUND_INTERVAL = 2  # 秒

# ============================================================================
# 特征配置
# ============================================================================

# 所有使用的特征（来自final_training_dataset.pickle，共105维）
# 特征架构:
# - PHQ-9: 31维 (9个archetype相似度 + 7个统计 + 9个per-archetype累计最大值 + 6个其他)
# - LSM: 60维 (10个基本 + 50个时间序列)
# - RE: 6维 (loss_dep0, loss_dep1, re + 时间序列统计)
# - CosineSimilarity: 6维 ⭐ 新增 (sim + 时间序列统计)
# - Other: 2维 (预留)
FEATURE_COLS = [
    'PHQ-9_archetype_scale.0.sim', 'PHQ-9_archetype_scale.1.sim', 'PHQ-9_archetype_scale.2.sim', 'PHQ-9_archetype_scale.3.sim', 'PHQ-9_archetype_scale.4.sim', 'PHQ-9_archetype_scale.5.sim', 'PHQ-9_archetype_scale.6.sim', 'PHQ-9_archetype_scale.7.sim', 'PHQ-9_archetype_scale.8.sim', 'phq_score',
    'max_phq_so_far', 'min_phq_so_far', 'max_gap', 'delta_phq', 'rolling_std_phq', 'post_index', 'max_0_so_far', 'max_1_so_far', 'max_2_so_far', 'max_3_so_far', 'max_4_so_far', 'max_5_so_far', 'max_6_so_far', 'max_7_so_far', 'max_8_so_far', 
    'sim', 'max_sim_so_far', 'min_sim_so_far', 'max_gap_sim_so_far', 'delta_sim', 'rolling_std_sim', 
    'LSM_mean', 'social_mean', 'positive_mean', 'negative_mean', 'first_person_singular', 'second_person', 'third_person_singular', 'third_person_plural', 'cognitive_process', 'perceptual_process', 
    'LSM_mean_max_so_far', 'LSM_mean_min_so_far', 'LSM_mean_max_gap_so_far', 'LSM_mean_delta', 'LSM_mean_rolling_std', 'social_mean_max_so_far', 'social_mean_min_so_far', 'social_mean_max_gap_so_far', 'social_mean_delta', 
    'social_mean_rolling_std', 'positive_mean_max_so_far', 'positive_mean_min_so_far', 'positive_mean_max_gap_so_far', 'positive_mean_delta', 'positive_mean_rolling_std', 'negative_mean_max_so_far', 'negative_mean_min_so_far', 
    'negative_mean_max_gap_so_far', 'negative_mean_delta', 'negative_mean_rolling_std', 'first_person_singular_max_so_far', 'first_person_singular_min_so_far', 'first_person_singular_max_gap_so_far', 'first_person_singular_delta', 
    'first_person_singular_rolling_std', 'second_person_max_so_far', 'second_person_min_so_far', 'second_person_max_gap_so_far', 'second_person_delta', 'second_person_rolling_std', 'third_person_singular_max_so_far', 
    'third_person_singular_min_so_far', 'third_person_singular_max_gap_so_far', 'third_person_singular_delta', 'third_person_singular_rolling_std', 'third_person_plural_max_so_far', 'third_person_plural_min_so_far', 
    'third_person_plural_max_gap_so_far', 'third_person_plural_delta', 'third_person_plural_rolling_std', 'cognitive_process_max_so_far', 'cognitive_process_min_so_far', 'cognitive_process_max_gap_so_far', 'cognitive_process_delta', 
    'cognitive_process_rolling_std', 'perceptual_process_max_so_far', 'perceptual_process_min_so_far', 'perceptual_process_max_gap_so_far', 'perceptual_process_delta', 'perceptual_process_rolling_std',
    'loss_dep0', 'loss_dep1', 're', 'max_re_so_far', 'min_re_so_far', 'max_gap_re_so_far', 'delta_re', 'rolling_std_re'
]

# ============================================================================
# 日志配置
# ============================================================================

LOG_FILE = 'erisk_test.log'
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR

# ============================================================================
# 输出配置
# ============================================================================

# 保存输出
SAVE_DISCUSSIONS = True
DISCUSSIONS_DIR = './output_discussions/'

SAVE_USERS = True
USERS_FILE = './dummy_users/target_users.txt'

SAVE_STATISTICS = True
STATS_FILE = './output_statistics.json'

# ============================================================================
# 调试配置
# ============================================================================

# 是否打印详细调试信息
DEBUG_MODE = False

# 是否在测试模式下（不真正发送请求）
TEST_MODE = False

# 最大轮数（用于测试，0表示无限制）
MAX_ROUNDS = 0
