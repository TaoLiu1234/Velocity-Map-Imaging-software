"""
VMI Algorithm Testing and Improvement Framework
================================================

综合测试框架，用于系统性评估和改进 VMI 重建算法。

功能：
1. 正交测试用例生成 - 高效覆盖参数空间
2. 前向模拟接口 - 使用 Abel_forward_simulation 生成带噪声的测试数据
3. 性能评估 - 计算与真值的百分比偏差
4. 改进算法 - 自动参数调整，减少人工干预
5. 测试报告 - 生成详细的性能报告

使用方法:
    from vmi_test_framework import run_comprehensive_tests
    
    results = run_comprehensive_tests(n_cases=27)
    results.save('test_results.json')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks, correlate
from scipy.ndimage import zoom, gaussian_filter1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
import json
import time
from pathlib import Path

# Import forward simulation
from Abel_forward_simulation import Config, run_simulation, ELECTRON_MASS_AMU, EV_TO_JOULE, AMU_TO_KG

# Import original reconstructor for comparison
from vmi_reconstruction import VMIReconstructor, PeakResult, abel_projection, P2


# =============================================================================
# Constants
# =============================================================================
DEFAULT_R_MAX = 20.0  # mm
DEFAULT_IMG_RES = 512
DEFAULT_PIXEL_SIZE = 0.05  # mm


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TestCase:
    """测试用例"""
    case_id: str                    # 用例 ID
    n_peaks: int                    # 峰数量
    event_count: int                # 事件数
    peak_separation: str            # 峰分离度: 'well', 'moderate', 'overlap'
    beta_range: str                 # β 范围: 'negative', 'zero', 'positive'
    amplitude_ratio: str            # 振幅比: 'equal', '10:1', '100:1'
    sigma_range: str                # σ 范围: 'narrow', 'medium', 'wide'
    r_position: str                 # 径向位置: 'inner', 'middle', 'outer'
    noise_level: str                # 噪声水平: 'clean', 'low', 'high'
    
    # 具体参数值 (由 TestCaseGenerator 填充)
    E_centers: List[float] = field(default_factory=list)
    r0_values: List[float] = field(default_factory=list)
    sigma_values: List[float] = field(default_factory=list)
    beta_values: List[float] = field(default_factory=list)
    branching_ratios: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'case_id': self.case_id,
            'n_peaks': self.n_peaks,
            'event_count': self.event_count,
            'peak_separation': self.peak_separation,
            'beta_range': self.beta_range,
            'amplitude_ratio': self.amplitude_ratio,
            'sigma_range': self.sigma_range,
            'r_position': self.r_position,
            'noise_level': self.noise_level,
            'E_centers': self.E_centers,
            'r0_values': self.r0_values,
            'sigma_values': self.sigma_values,
            'beta_values': self.beta_values,
            'branching_ratios': self.branching_ratios,
        }


@dataclass
class EvaluationResult:
    """评估结果"""
    case_id: str                    # 用例 ID
    
    # 峰匹配
    n_true_peaks: int               # 真实峰数
    n_detected_peaks: int           # 检测到的峰数
    n_matched_peaks: int            # 匹配的峰数
    n_missed_peaks: int             # 漏检的峰数
    n_false_positives: int          # 误检的峰数
    
    # 误差（每个匹配峰）
    r0_errors: List[float] = field(default_factory=list)    # r0 相对误差 (%)
    sigma_errors: List[float] = field(default_factory=list) # σ 相对误差 (%)
    beta_errors: List[float] = field(default_factory=list)  # β 绝对误差
    amp_errors: List[float] = field(default_factory=list)   # 振幅相对误差 (%)
    
    # 汇总
    mean_r0_error: float = 0.0
    mean_sigma_error: float = 0.0
    mean_beta_error: float = 0.0
    passed: bool = False            # 是否通过
    
    # 执行信息
    execution_time: float = 0.0     # 执行时间 (秒)
    error_message: str = ""         # 错误信息
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'case_id': self.case_id,
            'n_true_peaks': self.n_true_peaks,
            'n_detected_peaks': self.n_detected_peaks,
            'n_matched_peaks': self.n_matched_peaks,
            'n_missed_peaks': self.n_missed_peaks,
            'n_false_positives': self.n_false_positives,
            'r0_errors': self.r0_errors,
            'sigma_errors': self.sigma_errors,
            'beta_errors': self.beta_errors,
            'amp_errors': self.amp_errors,
            'mean_r0_error': self.mean_r0_error,
            'mean_sigma_error': self.mean_sigma_error,
            'mean_beta_error': self.mean_beta_error,
            'passed': self.passed,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
        }


@dataclass
class TestSummary:
    """测试汇总"""
    total_cases: int                # 总测试数
    passed_cases: int               # 通过数
    failed_cases: int               # 失败数
    
    # 按因子分组的通过率
    pass_rate_by_event_count: Dict[str, float] = field(default_factory=dict)
    pass_rate_by_separation: Dict[str, float] = field(default_factory=dict)
    pass_rate_by_beta: Dict[str, float] = field(default_factory=dict)
    pass_rate_by_r_position: Dict[str, float] = field(default_factory=dict)
    pass_rate_by_n_peaks: Dict[str, float] = field(default_factory=dict)
    
    # 误差统计
    r0_error_stats: Dict[str, float] = field(default_factory=dict)
    sigma_error_stats: Dict[str, float] = field(default_factory=dict)
    beta_error_stats: Dict[str, float] = field(default_factory=dict)
    
    # 性能极限
    min_event_count_for_5pct: int = 0
    min_separation_for_5pct: str = ""
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'total_cases': self.total_cases,
            'passed_cases': self.passed_cases,
            'failed_cases': self.failed_cases,
            'pass_rate_by_event_count': self.pass_rate_by_event_count,
            'pass_rate_by_separation': self.pass_rate_by_separation,
            'pass_rate_by_beta': self.pass_rate_by_beta,
            'pass_rate_by_r_position': self.pass_rate_by_r_position,
            'pass_rate_by_n_peaks': self.pass_rate_by_n_peaks,
            'r0_error_stats': self.r0_error_stats,
            'sigma_error_stats': self.sigma_error_stats,
            'beta_error_stats': self.beta_error_stats,
            'min_event_count_for_5pct': self.min_event_count_for_5pct,
            'min_separation_for_5pct': self.min_separation_for_5pct,
        }



# =============================================================================
# Orthogonal Test Designer
# =============================================================================

class OrthogonalTestDesigner:
    """正交测试设计器
    
    使用正交表设计测试用例，高效覆盖参数空间。
    支持 L27(3^13) 正交表，可覆盖 8 个 3-水平因子。
    """
    
    # 测试因子定义
    FACTORS = {
        'n_peaks': [1, 2, 3],
        'event_count': [int(1e4), int(1e6), int(1e8)],
        'peak_separation': ['well', 'moderate', 'overlap'],
        'beta_range': ['negative', 'zero', 'positive'],
        'amplitude_ratio': ['equal', '10:1', '100:1'],
        'sigma_range': ['narrow', 'medium', 'wide'],
        'r_position': ['inner', 'middle', 'outer'],
        'noise_level': ['clean', 'low', 'high'],
    }
    
    # L27(3^13) 正交表 - 前 8 列用于 8 个因子
    # 每行是一个测试用例，值为 0, 1, 2 表示因子的三个水平
    L27_ARRAY = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 2, 2, 2, 2],
        [0, 1, 1, 1, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 1, 2],
        [0, 1, 1, 1, 2, 2, 2, 0],
        [0, 2, 2, 2, 0, 0, 0, 2],
        [0, 2, 2, 2, 1, 1, 1, 0],
        [0, 2, 2, 2, 2, 2, 2, 1],
        [1, 0, 1, 2, 0, 1, 2, 0],
        [1, 0, 1, 2, 1, 2, 0, 1],
        [1, 0, 1, 2, 2, 0, 1, 2],
        [1, 1, 2, 0, 0, 1, 2, 1],
        [1, 1, 2, 0, 1, 2, 0, 2],
        [1, 1, 2, 0, 2, 0, 1, 0],
        [1, 2, 0, 1, 0, 1, 2, 2],
        [1, 2, 0, 1, 1, 2, 0, 0],
        [1, 2, 0, 1, 2, 0, 1, 1],
        [2, 0, 2, 1, 0, 2, 1, 0],
        [2, 0, 2, 1, 1, 0, 2, 1],
        [2, 0, 2, 1, 2, 1, 0, 2],
        [2, 1, 0, 2, 0, 2, 1, 1],
        [2, 1, 0, 2, 1, 0, 2, 2],
        [2, 1, 0, 2, 2, 1, 0, 0],
        [2, 2, 1, 0, 0, 2, 1, 2],
        [2, 2, 1, 0, 1, 0, 2, 0],
        [2, 2, 1, 0, 2, 1, 0, 1],
    ])
    
    def __init__(self, factors: dict = None):
        """
        Args:
            factors: 自定义因子定义，默认使用 FACTORS
        """
        self.factors = factors or self.FACTORS.copy()
        self.factor_names = list(self.factors.keys())
    
    def generate_orthogonal_array(self) -> np.ndarray:
        """生成正交表
        
        Returns:
            正交表矩阵，每行是一个测试用例的因子水平索引
        """
        return self.L27_ARRAY.copy()
    
    def generate_test_cases(self) -> List[TestCase]:
        """生成测试用例列表
        
        Returns:
            测试用例列表，每个用例包含所有参数的具体值
        """
        array = self.generate_orthogonal_array()
        test_cases = []
        
        for i, row in enumerate(array):
            case = TestCase(
                case_id=f"OA_{i+1:03d}",
                n_peaks=self.factors['n_peaks'][row[0]],
                event_count=self.factors['event_count'][row[1]],
                peak_separation=self.factors['peak_separation'][row[2]],
                beta_range=self.factors['beta_range'][row[3]],
                amplitude_ratio=self.factors['amplitude_ratio'][row[4]],
                sigma_range=self.factors['sigma_range'][row[5]],
                r_position=self.factors['r_position'][row[6]],
                noise_level=self.factors['noise_level'][row[7]],
            )
            test_cases.append(case)
        
        return test_cases
    
    def add_corner_cases(self, test_cases: List[TestCase]) -> List[TestCase]:
        """添加角落用例（极端条件）
        
        Returns:
            包含角落用例的完整测试列表
        """
        corner_cases = [
            # 极低事件数 + 单峰
            TestCase(
                case_id="CORNER_001",
                n_peaks=1,
                event_count=int(1e4),
                peak_separation='well',
                beta_range='zero',
                amplitude_ratio='equal',
                sigma_range='medium',
                r_position='middle',
                noise_level='high',
            ),
            # 极高事件数 + 多峰重叠
            TestCase(
                case_id="CORNER_002",
                n_peaks=3,
                event_count=int(1e8),
                peak_separation='overlap',
                beta_range='positive',
                amplitude_ratio='100:1',
                sigma_range='wide',
                r_position='outer',
                noise_level='clean',
            ),
            # 极端 β 值 (-1)
            TestCase(
                case_id="CORNER_003",
                n_peaks=1,
                event_count=int(1e6),
                peak_separation='well',
                beta_range='negative',  # β = -1
                amplitude_ratio='equal',
                sigma_range='medium',
                r_position='middle',
                noise_level='low',
            ),
            # 极端 β 值 (2)
            TestCase(
                case_id="CORNER_004",
                n_peaks=1,
                event_count=int(1e6),
                peak_separation='well',
                beta_range='positive',  # β = 2
                amplitude_ratio='equal',
                sigma_range='medium',
                r_position='middle',
                noise_level='low',
            ),
            # 极窄峰
            TestCase(
                case_id="CORNER_005",
                n_peaks=2,
                event_count=int(1e6),
                peak_separation='well',
                beta_range='zero',
                amplitude_ratio='equal',
                sigma_range='narrow',  # σ < 0.1
                r_position='middle',
                noise_level='low',
            ),
            # 极宽峰
            TestCase(
                case_id="CORNER_006",
                n_peaks=2,
                event_count=int(1e6),
                peak_separation='well',
                beta_range='zero',
                amplitude_ratio='equal',
                sigma_range='wide',  # σ > 2.0
                r_position='middle',
                noise_level='low',
            ),
            # 边缘位置 - 内侧
            TestCase(
                case_id="CORNER_007",
                n_peaks=1,
                event_count=int(1e6),
                peak_separation='well',
                beta_range='zero',
                amplitude_ratio='equal',
                sigma_range='medium',
                r_position='inner',  # r0 < 5mm
                noise_level='low',
            ),
            # 边缘位置 - 外侧
            TestCase(
                case_id="CORNER_008",
                n_peaks=1,
                event_count=int(1e6),
                peak_separation='well',
                beta_range='zero',
                amplitude_ratio='equal',
                sigma_range='medium',
                r_position='outer',  # r0 > 15mm
                noise_level='low',
            ),
            # 极端振幅比
            TestCase(
                case_id="CORNER_009",
                n_peaks=2,
                event_count=int(1e7),
                peak_separation='well',
                beta_range='zero',
                amplitude_ratio='100:1',
                sigma_range='medium',
                r_position='middle',
                noise_level='low',
            ),
            # 5 峰测试
            TestCase(
                case_id="CORNER_010",
                n_peaks=5,
                event_count=int(1e7),
                peak_separation='well',
                beta_range='zero',
                amplitude_ratio='equal',
                sigma_range='narrow',
                r_position='middle',
                noise_level='low',
            ),
        ]
        
        return test_cases + corner_cases
    
    def estimate_test_count(self, include_corners: bool = True) -> int:
        """估计总测试数
        
        Args:
            include_corners: 是否包含角落用例
            
        Returns:
            测试用例总数
        """
        base_count = len(self.L27_ARRAY)
        corner_count = 10 if include_corners else 0
        return base_count + corner_count



# =============================================================================
# Test Case Generator
# =============================================================================

class TestCaseGenerator:
    """测试用例生成器
    
    将正交设计转换为具体的模拟参数。
    """
    
    # 参数映射
    SIGMA_MAP = {'narrow': 0.08, 'medium': 0.4, 'wide': 1.5}  # mm (energy width in radius space)
    R_POSITION_MAP = {
        'inner': (2.0, 5.0),
        'middle': (7.0, 14.0),
        'outer': (15.0, 19.0)
    }
    BETA_MAP = {
        'negative': (-1.0, -0.3),
        'zero': (-0.2, 0.2),
        'positive': (0.5, 2.0)
    }
    NOISE_MAP = {
        'clean': {'psf_fwhm': 0.0, 'dld_resolution': 0.0},
        'low': {'psf_fwhm': 0.1, 'dld_resolution': 0.01},
        'high': {'psf_fwhm': 0.3, 'dld_resolution': 0.05}
    }
    
    def __init__(self, r_max: float = DEFAULT_R_MAX, vmi_k: float = None):
        """
        Args:
            r_max: 最大半径 (mm)
            vmi_k: VMI 校准系数，默认自动计算
        """
        self.r_max = r_max
        # 计算 vmi_k: 使 2 eV 对应 r_max
        E_max = 2.0  # eV
        if vmi_k is None:
            self.vmi_k = Config.calculate_vmi_k(E_max, r_max)
        else:
            self.vmi_k = vmi_k
    
    def _r_to_energy(self, r: float) -> float:
        """半径转能量"""
        mass_kg = ELECTRON_MASS_AMU * AMU_TO_KG
        v = r / self.vmi_k
        E = 0.5 * mass_kg * v**2 / EV_TO_JOULE
        return E
    
    def _energy_to_r(self, E: float) -> float:
        """能量转半径"""
        mass_kg = ELECTRON_MASS_AMU * AMU_TO_KG
        v = np.sqrt(2.0 * E * EV_TO_JOULE / mass_kg)
        return self.vmi_k * v
    
    def _generate_peak_positions(self, n_peaks: int, r_position: str, 
                                  separation: str, sigma: float) -> List[float]:
        """生成峰位置"""
        r_min, r_max = self.R_POSITION_MAP[r_position]
        
        if n_peaks == 1:
            return [(r_min + r_max) / 2]
        
        # 根据分离度确定间距
        if separation == 'well':
            min_sep = 6 * sigma  # > 5σ
        elif separation == 'moderate':
            min_sep = 3 * sigma  # ~3σ
        else:  # overlap
            min_sep = 1.5 * sigma  # < 2σ
        
        # 在范围内均匀分布
        total_range = r_max - r_min
        required_range = (n_peaks - 1) * min_sep
        
        if required_range > total_range:
            # 如果范围不够，压缩间距
            actual_sep = total_range / (n_peaks - 1) * 0.9
        else:
            actual_sep = min_sep
        
        # 居中分布
        center = (r_min + r_max) / 2
        start = center - (n_peaks - 1) * actual_sep / 2
        
        positions = [start + i * actual_sep for i in range(n_peaks)]
        
        # 确保在范围内
        positions = [max(r_min, min(r_max, p)) for p in positions]
        
        return positions
    
    def _generate_betas(self, n_peaks: int, beta_range: str) -> List[float]:
        """生成 β 值"""
        beta_min, beta_max = self.BETA_MAP[beta_range]
        
        if n_peaks == 1:
            return [(beta_min + beta_max) / 2]
        
        # 在范围内均匀分布
        return list(np.linspace(beta_min, beta_max, n_peaks))
    
    def _generate_amplitudes(self, n_peaks: int, ratio: str) -> List[float]:
        """生成振幅（分支比）"""
        if ratio == 'equal':
            return [1.0 / n_peaks] * n_peaks
        elif ratio == '10:1':
            if n_peaks == 1:
                return [1.0]
            # 第一个峰最强，其他峰弱 10 倍
            amps = [10.0] + [1.0] * (n_peaks - 1)
            total = sum(amps)
            return [a / total for a in amps]
        else:  # 100:1
            if n_peaks == 1:
                return [1.0]
            # 第一个峰最强，其他峰弱 100 倍
            amps = [100.0] + [1.0] * (n_peaks - 1)
            total = sum(amps)
            return [a / total for a in amps]
    
    def generate_config(self, test_case: TestCase) -> Config:
        """将测试用例转换为 Config 对象"""
        # 获取 sigma
        sigma = self.SIGMA_MAP[test_case.sigma_range]
        
        # 生成峰位置
        r0_values = self._generate_peak_positions(
            test_case.n_peaks, 
            test_case.r_position,
            test_case.peak_separation,
            sigma
        )
        
        # 转换为能量
        E_centers = [self._r_to_energy(r) for r in r0_values]
        
        # 生成 β 值
        beta_values = self._generate_betas(test_case.n_peaks, test_case.beta_range)
        
        # 生成分支比
        branching_ratios = self._generate_amplitudes(test_case.n_peaks, test_case.amplitude_ratio)
        
        # 获取噪声参数
        noise_params = self.NOISE_MAP[test_case.noise_level]
        
        # 更新 test_case 的具体参数
        test_case.E_centers = E_centers
        test_case.r0_values = r0_values
        test_case.sigma_values = [sigma] * test_case.n_peaks
        test_case.beta_values = beta_values
        test_case.branching_ratios = branching_ratios
        
        # 计算能量展宽 (sigma_laser)
        # 从半径展宽反推能量展宽: dE/E = 2 * dr/r
        avg_r = np.mean(r0_values)
        avg_E = np.mean(E_centers)
        sigma_laser = sigma / avg_r * avg_E if avg_r > 0 else 0.01
        
        # 创建 Config
        config = Config(
            E_centers=E_centers,
            Betas=beta_values,
            branching_ratios=branching_ratios,
            N_events=test_case.event_count,
            vmi_k=self.vmi_k,
            sigma_laser=sigma_laser,
            T_beam=0.0,  # 无 Doppler 展宽
            tau_lifetimes=0.0,  # 无寿命展宽
            photon_energy=0.0,
            target_mass=28.0,
            vol_sigma=(0.0, 0.0, 0.0),
            polarization_vec=[0, 1, 0],
            img_res=DEFAULT_IMG_RES,
            pixel_size=DEFAULT_PIXEL_SIZE,
            psf_fwhm=noise_params['psf_fwhm'],
            dld_resolution=noise_params['dld_resolution'],
            dark_rate=0.0,
            readout_sigma=0.0,
            readout_offset=0.0,
            bg_rate=0.0,
        )
        
        return config
    
    def fill_test_cases(self, test_cases: List[TestCase]) -> List[TestCase]:
        """填充所有测试用例的具体参数"""
        for tc in test_cases:
            self.generate_config(tc)
        return test_cases


# =============================================================================
# Simulation Runner
# =============================================================================

class SimulationRunner:
    """模拟运行器
    
    使用 Abel_forward_simulation 生成测试数据。
    默认使用 output_mode='xy_dld' 生成带噪声的 XY 数据。
    """
    
    def __init__(self, add_noise: bool = True):
        """
        Args:
            add_noise: 是否添加噪声（PSF + DLD 量化），默认 True
        """
        self.add_noise = add_noise
    
    def run(self, config: Config, test_case: TestCase = None) -> Tuple[np.ndarray, dict]:
        """运行模拟
        
        Args:
            config: 模拟配置
            test_case: 测试用例（用于获取真值）
            
        Returns:
            (xy_data, ground_truth)
        """
        # 运行模拟
        output_mode = 'xy_dld' if self.add_noise else 'xy_ideal'
        xy_data, metadata = run_simulation(
            config, 
            add_noise=False,  # 不添加相机噪声
            add_background=False,
            return_particles=False,
            output_mode=output_mode
        )
        
        # 构建真值
        ground_truth = {
            'E_centers': config.E_centers,
            'Betas': config.Betas,
            'branching_ratios': config.branching_ratios,
            'N_events': config.N_events,
            'vmi_k': config.vmi_k,
        }
        
        # 如果有 test_case，添加更多信息
        if test_case is not None:
            ground_truth['r0_values'] = test_case.r0_values
            ground_truth['sigma_values'] = test_case.sigma_values
            ground_truth['beta_values'] = test_case.beta_values
        else:
            # 从能量计算半径
            mass_kg = ELECTRON_MASS_AMU * AMU_TO_KG
            r0_values = []
            for E in config.E_centers:
                v = np.sqrt(2.0 * E * EV_TO_JOULE / mass_kg)
                r0_values.append(config.vmi_k * v)
            ground_truth['r0_values'] = r0_values
            # sigma 需要从 sigma_laser 估计
            avg_r = np.mean(r0_values)
            avg_E = np.mean(config.E_centers)
            sigma_r = config.sigma_laser / avg_E * avg_r if avg_E > 0 else 0.1
            ground_truth['sigma_values'] = [sigma_r] * len(config.E_centers)
            ground_truth['beta_values'] = config.Betas
        
        return xy_data, ground_truth
    
    def run_batch(self, configs: List[Tuple[Config, TestCase]], 
                  progress_callback: callable = None) -> List[Tuple[np.ndarray, dict]]:
        """批量运行模拟"""
        results = []
        total = len(configs)
        
        for i, (config, test_case) in enumerate(configs):
            xy_data, ground_truth = self.run(config, test_case)
            results.append((xy_data, ground_truth))
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results


# =============================================================================
# Performance Evaluator
# =============================================================================

class PerformanceEvaluator:
    """性能评估器
    
    计算重建结果与真值的偏差。
    
    注意：sigma 的比较比较困难，因为 Abel 投影会使峰变宽。
    重建得到的 sigma 是 Abel 投影后的宽度，而真值是内禀能量宽度。
    因此 sigma 不作为通过/失败的判断标准（设置很高的容差）。
    """
    
    def __init__(self, tolerance_r0: float = 0.05, 
                 tolerance_sigma: float = 10.0,  # 1000% - effectively ignore sigma
                 tolerance_beta: float = 0.2):
        """
        Args:
            tolerance_r0: r0 容差（相对误差）
            tolerance_sigma: σ 容差（相对误差）- 由于 Abel 投影，这个值设得很宽松
            tolerance_beta: β 容差（绝对误差）
        """
        self.tolerance_r0 = tolerance_r0
        self.tolerance_sigma = tolerance_sigma
        self.tolerance_beta = tolerance_beta
    
    def _match_peaks(self, estimated: List[PeakResult], 
                     true_r0s: List[float]) -> List[Tuple[int, int]]:
        """匹配估计峰与真实峰
        
        使用贪婪最近邻匹配。
        """
        if len(estimated) == 0 or len(true_r0s) == 0:
            return []
        
        est_r0s = [p.r0 for p in estimated]
        matches = []
        used_est = set()
        used_true = set()
        
        # 计算所有距离
        distances = []
        for i, er in enumerate(est_r0s):
            for j, tr in enumerate(true_r0s):
                distances.append((abs(er - tr), i, j))
        
        # 按距离排序，贪婪匹配
        distances.sort()
        for dist, i, j in distances:
            if i not in used_est and j not in used_true:
                matches.append((i, j))
                used_est.add(i)
                used_true.add(j)
        
        return matches
    
    def compute_errors(self, estimated: PeakResult, 
                       true_r0: float, true_sigma: float, 
                       true_beta: float, true_amp: float = None) -> dict:
        """计算单个峰的误差"""
        r0_error = abs(estimated.r0 - true_r0) / true_r0 * 100 if true_r0 > 0 else 0
        sigma_error = abs(estimated.sigma - true_sigma) / true_sigma * 100 if true_sigma > 0 else 0
        beta_error = abs(estimated.beta - true_beta)
        
        result = {
            'r0_error': r0_error,
            'sigma_error': sigma_error,
            'beta_error': beta_error,
        }
        
        if true_amp is not None and true_amp > 0:
            amp_error = abs(estimated.amp - true_amp) / true_amp * 100
            result['amp_error'] = amp_error
        
        return result
    
    def evaluate(self, estimated: List[PeakResult], 
                 ground_truth: dict, test_case: TestCase = None) -> EvaluationResult:
        """评估单个测试用例"""
        true_r0s = ground_truth['r0_values']
        true_sigmas = ground_truth['sigma_values']
        true_betas = ground_truth['beta_values']
        true_amps = ground_truth.get('branching_ratios', [1.0] * len(true_r0s))
        
        n_true = len(true_r0s)
        n_detected = len(estimated)
        
        # 匹配峰
        matches = self._match_peaks(estimated, true_r0s)
        n_matched = len(matches)
        n_missed = n_true - n_matched
        n_false_pos = n_detected - n_matched
        
        # 计算误差
        r0_errors = []
        sigma_errors = []
        beta_errors = []
        amp_errors = []
        
        for est_idx, true_idx in matches:
            errors = self.compute_errors(
                estimated[est_idx],
                true_r0s[true_idx],
                true_sigmas[true_idx],
                true_betas[true_idx],
                true_amps[true_idx] if true_idx < len(true_amps) else None
            )
            r0_errors.append(errors['r0_error'])
            sigma_errors.append(errors['sigma_error'])
            beta_errors.append(errors['beta_error'])
            if 'amp_error' in errors:
                amp_errors.append(errors['amp_error'])
        
        # 对漏检的峰，记录 100% 误差
        for _ in range(n_missed):
            r0_errors.append(100.0)
            sigma_errors.append(100.0)
            beta_errors.append(3.0)  # 最大可能误差
        
        # 计算均值
        mean_r0 = np.mean(r0_errors) if r0_errors else 100.0
        mean_sigma = np.mean(sigma_errors) if sigma_errors else 100.0
        mean_beta = np.mean(beta_errors) if beta_errors else 3.0
        
        # 判断是否通过
        passed = (
            mean_r0 <= self.tolerance_r0 * 100 and
            mean_sigma <= self.tolerance_sigma * 100 and
            mean_beta <= self.tolerance_beta and
            n_missed == 0 and
            n_false_pos == 0
        )
        
        return EvaluationResult(
            case_id=test_case.case_id if test_case else "unknown",
            n_true_peaks=n_true,
            n_detected_peaks=n_detected,
            n_matched_peaks=n_matched,
            n_missed_peaks=n_missed,
            n_false_positives=n_false_pos,
            r0_errors=r0_errors,
            sigma_errors=sigma_errors,
            beta_errors=beta_errors,
            amp_errors=amp_errors,
            mean_r0_error=mean_r0,
            mean_sigma_error=mean_sigma,
            mean_beta_error=mean_beta,
            passed=passed,
        )
    
    def aggregate_results(self, all_results: List[EvaluationResult], 
                          test_cases: List[TestCase]) -> TestSummary:
        """汇总所有测试结果"""
        total = len(all_results)
        passed = sum(1 for r in all_results if r.passed)
        failed = total - passed
        
        # 收集所有误差
        all_r0_errors = []
        all_sigma_errors = []
        all_beta_errors = []
        
        for r in all_results:
            all_r0_errors.extend(r.r0_errors)
            all_sigma_errors.extend(r.sigma_errors)
            all_beta_errors.extend(r.beta_errors)
        
        # 按因子分组统计
        def compute_pass_rate_by_factor(factor_name: str) -> Dict[str, float]:
            groups = {}
            for tc, result in zip(test_cases, all_results):
                key = str(getattr(tc, factor_name))
                if key not in groups:
                    groups[key] = {'passed': 0, 'total': 0}
                groups[key]['total'] += 1
                if result.passed:
                    groups[key]['passed'] += 1
            return {k: v['passed'] / v['total'] * 100 for k, v in groups.items()}
        
        summary = TestSummary(
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            pass_rate_by_event_count=compute_pass_rate_by_factor('event_count'),
            pass_rate_by_separation=compute_pass_rate_by_factor('peak_separation'),
            pass_rate_by_beta=compute_pass_rate_by_factor('beta_range'),
            pass_rate_by_r_position=compute_pass_rate_by_factor('r_position'),
            pass_rate_by_n_peaks=compute_pass_rate_by_factor('n_peaks'),
            r0_error_stats={
                'mean': np.mean(all_r0_errors) if all_r0_errors else 0,
                'median': np.median(all_r0_errors) if all_r0_errors else 0,
                'std': np.std(all_r0_errors) if all_r0_errors else 0,
                'max': np.max(all_r0_errors) if all_r0_errors else 0,
            },
            sigma_error_stats={
                'mean': np.mean(all_sigma_errors) if all_sigma_errors else 0,
                'median': np.median(all_sigma_errors) if all_sigma_errors else 0,
                'std': np.std(all_sigma_errors) if all_sigma_errors else 0,
                'max': np.max(all_sigma_errors) if all_sigma_errors else 0,
            },
            beta_error_stats={
                'mean': np.mean(all_beta_errors) if all_beta_errors else 0,
                'median': np.median(all_beta_errors) if all_beta_errors else 0,
                'std': np.std(all_beta_errors) if all_beta_errors else 0,
                'max': np.max(all_beta_errors) if all_beta_errors else 0,
            },
        )
        
        return summary



# =============================================================================
# Improved VMI Reconstructor - Physics-First Design v3.0
# =============================================================================

class ImprovedVMIReconstructor:
    """
    Physics-First VMI Reconstructor v3.0
    
    Physical Principles:
    ====================
    1. VMI Imaging: Photoelectrons fly in electric field, hit detector at r ∝ √E
    2. Angular Distribution: I(θ_3D) = σ/(4π) * [1 + β·P₂(cos θ_3D)]
       - β is constrained: -1 ≤ β ≤ 2 (from angular momentum conservation)
       - β = 2: parallel transition (Δl = +1)
       - β = -1: perpendicular transition (Δl = -1)
       - β = 0: isotropic
    3. Abel Projection: 3D → 2D causes peak broadening
    4. Polarization along Y-axis: 2D distribution is I(θ_XY) ∝ 1 + β·P₂(sin θ_XY)
    5. Radial Density: ρ(r) = H(r)/(2πr·dr), the Jacobian 2πr is crucial
    
    Key Improvements:
    =================
    1. Physics-constrained β: Correct P₂(sin θ) and -1 ≤ β ≤ 2
    2. Poisson Statistics: Proper handling, especially for inner peaks (small r, few counts)
    3. Abel Correction: Account for projection-induced peak broadening
    4. Adaptive Binning: Based on Poisson statistics for sufficient counts per bin
    5. Iterative Peak Detection: Start from strongest, subtract, find next
    """
    
    # Physical constants
    BETA_MIN = -1.0
    BETA_MAX = 2.0
    
    def __init__(self, xy_data: np.ndarray):
        """
        Args:
            xy_data: (N, 2) XY coordinates in mm
        """
        self.xy_data = np.asarray(xy_data, dtype=np.float64)
        self.n_events = len(xy_data)
        
        # Step 1: Find center using physical symmetry
        self.center = self._find_center_by_symmetry()
        
        # Step 2: Convert to polar coordinates
        dx = self.xy_data[:, 0] - self.center[0]
        dy = self.xy_data[:, 1] - self.center[1]
        self.r = np.sqrt(dx**2 + dy**2)
        self.theta = np.arctan2(dy, dx)  # -π to π
        
        # Step 3: Determine r_max (exclude outliers)
        self.r_max = np.percentile(self.r, 99.5)
        
        # Step 4: Auto-determine bin size based on Poisson statistics
        self.dr = self._compute_optimal_bin_size()
        
        # Results
        self.peaks: List[PeakResult] = []
    
    def _find_center_by_symmetry(self) -> Tuple[float, float]:
        """
        Find center using physical symmetry.
        
        Physics: VMI images should be symmetric about the center.
        For anisotropic distributions (β ≠ 0), we need to account for
        the fact that there are more particles along the polarization axis.
        
        Method: Use the median of X and Y separately, then refine using
        the symmetry of the radial distribution along perpendicular axes.
        For Y-polarization, the X distribution should be symmetric.
        """
        # Initial estimate: median (robust to outliers and anisotropy)
        cx = np.median(self.xy_data[:, 0])
        cy = np.median(self.xy_data[:, 1])
        
        # For anisotropic distributions, the median is more robust than mean
        # But we can refine using the fact that X should be symmetric
        
        # Method 1: Use X-axis symmetry (perpendicular to polarization)
        # The distribution along X should be symmetric about the center
        x_data = self.xy_data[:, 0]
        y_data = self.xy_data[:, 1]
        
        # Refine cx using X symmetry
        for _ in range(5):
            dx = x_data - cx
            # Compare positive and negative X
            pos_x = dx[dx > 0]
            neg_x = -dx[dx < 0]
            if len(pos_x) > 100 and len(neg_x) > 100:
                # The median of |x| should be the same on both sides
                med_pos = np.median(pos_x)
                med_neg = np.median(neg_x)
                cx += (med_pos - med_neg) * 0.3
        
        # Refine cy using radial symmetry at fixed |x|
        # For Y-polarization, at fixed |x|, the Y distribution should be symmetric
        for _ in range(5):
            dy = y_data - cy
            dx = x_data - cx
            r = np.sqrt(dx**2 + dy**2)
            
            # Use points at intermediate radii
            r_med = np.median(r)
            mask = (r > r_med * 0.3) & (r < r_med * 1.5)
            
            if np.sum(mask) < 100:
                break
            
            # For points with similar |x|, Y should be symmetric
            # Use the median of Y for points with |x| < r_med/2
            inner_mask = mask & (np.abs(dx) < r_med * 0.5)
            if np.sum(inner_mask) > 50:
                cy_inner = np.median(y_data[inner_mask])
                cy = 0.7 * cy + 0.3 * cy_inner
        
        # Final refinement using quadrant balance (but with smaller weight)
        for iteration in range(10):
            dx = self.xy_data[:, 0] - cx
            dy = self.xy_data[:, 1] - cy
            r = np.sqrt(dx**2 + dy**2)
            
            r_med = np.median(r)
            mask = (r > r_med * 0.25) & (r < r_med * 1.5)
            
            if np.sum(mask) < 50:
                break
            
            # Use median radius in each quadrant (more robust than mean)
            q1 = mask & (dx > 0) & (dy > 0)
            q2 = mask & (dx < 0) & (dy > 0)
            q3 = mask & (dx < 0) & (dy < 0)
            q4 = mask & (dx > 0) & (dy < 0)
            
            r_q1 = np.median(r[q1]) if np.sum(q1) > 10 else r_med
            r_q2 = np.median(r[q2]) if np.sum(q2) > 10 else r_med
            r_q3 = np.median(r[q3]) if np.sum(q3) > 10 else r_med
            r_q4 = np.median(r[q4]) if np.sum(q4) > 10 else r_med
            
            # Smaller damping for stability
            damping = 0.05 / (1 + iteration * 0.3)
            dcx = (r_q1 + r_q4 - r_q2 - r_q3) / 4 * damping
            dcy = (r_q1 + r_q2 - r_q3 - r_q4) / 4 * damping
            
            cx += dcx
            cy += dcy
            
            if abs(dcx) < 0.001 and abs(dcy) < 0.001:
                break
        
        return (cx, cy)
        
        return (cx, cy)
    
    def _compute_optimal_bin_size(self) -> float:
        """
        Compute optimal bin size based on Poisson statistics.
        
        Physics: Each bin should have enough counts for reliable statistics.
        For Poisson distribution, relative error ≈ 1/√N.
        We want ~30-50 counts per bin for ~15-20% statistical error.
        
        For low event counts, we need larger bins to get sufficient statistics.
        For high event counts, we can use smaller bins for better resolution.
        """
        min_counts_per_bin = 30  # Reduced from 50 for better resolution
        
        total_area = np.pi * self.r_max**2
        avg_density = self.n_events / total_area
        
        r_typical = self.r_max / 2
        dr_from_stats = min_counts_per_bin / (2 * np.pi * r_typical * avg_density + 1e-10)
        
        # Resolution-based: need to resolve peaks
        # Use smaller bins for high event counts
        if self.n_events > 1e7:
            dr_from_resolution = 0.01   # Very fine for very high counts
        elif self.n_events > 1e6:
            dr_from_resolution = 0.02   # Fine resolution
        elif self.n_events > 1e5:
            dr_from_resolution = 0.04   # Medium resolution
        elif self.n_events > 1e4:
            dr_from_resolution = 0.08   # Coarser for low counts
        else:
            dr_from_resolution = 0.15
        
        dr = max(dr_from_stats, dr_from_resolution)
        return np.clip(dr, 0.01, 0.5)
    
    def _compute_radial_distribution(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute radial histogram and density.
        
        Returns:
            r_centers: Bin centers (mm)
            counts: Raw counts per bin
            density: ρ(r) = counts / (2πr·dr) - the physical density
            
        Note: For inner peaks (small r), the density can be unreliable due to
        the 1/(2πr) factor. Use counts for peak detection in such cases.
        """
        # Use more bins for better resolution, especially for high event counts
        n_bins = max(50, min(500, int(self.r_max / self.dr)))
        counts, bin_edges = np.histogram(self.r, bins=n_bins, range=(0, self.r_max))
        r_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        dr = bin_edges[1] - bin_edges[0]
        
        # Physical density: ρ(r) = H(r) / (2πr·dr)
        # Use a minimum r to avoid division by zero
        r_min = max(0.5, dr)  # At least 0.5 mm or one bin width
        r_safe = np.maximum(r_centers, r_min)
        density = counts / (2 * np.pi * r_safe * dr)
        
        return r_centers, counts.astype(float), density
    
    def _detect_peaks_iterative(self, r_centers: np.ndarray, density: np.ndarray,
                                 max_peaks: int = 7) -> List[dict]:
        """
        Iterative peak detection with multi-scale approach.
        
        Physics: This approach handles overlapping peaks better by:
        1. Using multiple smoothing scales to detect peaks at different resolutions
        2. Using both counts and density for robust detection
        3. Merging nearby candidates
        4. Iteratively subtracting detected peaks
        5. Using lower thresholds for secondary peaks (unequal amplitude handling)
        
        For low event counts, we use more aggressive smoothing.
        For inner peaks (small r), we prefer counts-based detection.
        """
        dr = r_centers[1] - r_centers[0]
        
        # Also get counts for inner peak detection
        n_bins = len(r_centers)
        counts, _ = np.histogram(self.r, bins=n_bins, range=(0, self.r_max))
        counts = counts.astype(float)
        
        # Estimate background levels - exclude very inner region (r < 0.5 mm)
        valid_mask = r_centers > 0.5
        bg_density = np.percentile(density[valid_mask & (density > 0)], 10) if np.any(valid_mask & (density > 0)) else 0
        bg_counts = np.percentile(counts[valid_mask & (counts > 0)], 10) if np.any(valid_mask & (counts > 0)) else 0
        
        # Adaptive smoothing scales based on data quality and bin size
        n_events = self.n_events
        if n_events > 1e6:
            smooth_scales_mm = [0.02, 0.05, 0.1, 0.2]
            min_height_factor = 1.1
            min_prom_factor = 0.01
        elif n_events > 1e5:
            smooth_scales_mm = [0.05, 0.1, 0.2, 0.4]
            min_height_factor = 1.08
            min_prom_factor = 0.015
        else:
            smooth_scales_mm = [0.1, 0.2, 0.4, 0.6]
            min_height_factor = 1.05
            min_prom_factor = 0.02
        
        smooth_scales = [max(1, int(s / dr)) for s in smooth_scales_mm]
        
        all_candidates = []
        
        for smooth_sigma in smooth_scales:
            smoothed_density = gaussian_filter1d(density, sigma=smooth_sigma)
            max_val = np.max(smoothed_density[valid_mask])  # Exclude inner region
            if max_val > 0:
                min_height = max(bg_density * min_height_factor, max_val * 0.03)
                min_prominence = max_val * min_prom_factor
                min_distance = max(2, int(0.1 / dr))
                
                peaks_idx, props = find_peaks(
                    smoothed_density,
                    height=min_height,
                    prominence=min_prominence,
                    distance=min_distance
                )
                
                for idx, prom in zip(peaks_idx, props['prominences']):
                    # Skip peaks at very small r (likely artifacts)
                    if r_centers[idx] < 0.5:
                        continue
                    all_candidates.append({
                        'idx': idx,
                        'r': r_centers[idx],
                        'prominence': prom,
                        'amplitude': density[idx],
                        'scale': smooth_sigma * dr,
                        'source': 'density'
                    })
            
            smoothed_counts = gaussian_filter1d(counts, sigma=smooth_sigma)
            max_val_counts = np.max(smoothed_counts[valid_mask])
            if max_val_counts > 0:
                min_height_counts = max(bg_counts * min_height_factor, max_val_counts * 0.03)
                min_prominence_counts = max_val_counts * min_prom_factor
                
                peaks_idx_counts, props_counts = find_peaks(
                    smoothed_counts,
                    height=min_height_counts,
                    prominence=min_prominence_counts,
                    distance=min_distance
                )
                
                for idx, prom in zip(peaks_idx_counts, props_counts['prominences']):
                    r_val = r_centers[idx]
                    if r_val < 0.5:
                        continue
                    already_found = any(abs(c['r'] - r_val) < 0.15 for c in all_candidates)
                    if not already_found:
                        all_candidates.append({
                            'idx': idx,
                            'r': r_val,
                            'prominence': prom / (2 * np.pi * max(r_val, 0.5) * dr),
                            'amplitude': density[idx],
                            'scale': smooth_sigma * dr,
                            'source': 'counts'
                        })
        
        if len(all_candidates) == 0:
            if np.max(density[valid_mask]) > 0:
                max_idx = np.argmax(density * valid_mask)
                if density[max_idx] > bg_density * 1.02:
                    all_candidates.append({
                        'idx': max_idx,
                        'r': r_centers[max_idx],
                        'prominence': density[max_idx] - bg_density,
                        'amplitude': density[max_idx],
                        'scale': 0.1,
                        'source': 'fallback'
                    })
        
        # Merge nearby candidates - use smaller distance for fine resolution
        merge_distance = min(0.15, dr * 5)  # At most 0.15 mm or 5 bins
        merged = []
        used = set()
        all_candidates.sort(key=lambda x: x['prominence'], reverse=True)
        
        for c in all_candidates:
            if c['idx'] in used:
                continue
            nearby = [c]
            for other in all_candidates:
                if other['idx'] != c['idx'] and other['idx'] not in used:
                    if abs(other['r'] - c['r']) < merge_distance:
                        nearby.append(other)
                        used.add(other['idx'])
            used.add(c['idx'])
            
            best = max(nearby, key=lambda x: x['prominence'])
            finest = min(nearby, key=lambda x: x['scale'])
            merged.append({
                'r': finest['r'],
                'prominence': best['prominence'],
                'amplitude': density[finest['idx']]
            })
        
        # Iterative refinement
        residual = density.copy()
        detected = []
        
        for i, candidate in enumerate(merged[:max_peaks]):
            best_r = candidate['r']
            best_prominence = candidate['prominence']
            
            idx = np.argmin(np.abs(r_centers - best_r))
            threshold = bg_density * (1.02 if i == 0 else 1.01)
            if residual[idx] < threshold:
                continue
            
            sigma_est = self._estimate_sigma_fwhm(r_centers, residual, best_r)
            
            detected.append({
                'r': best_r,
                'sigma': sigma_est,
                'prominence': best_prominence,
                'amplitude': residual[idx]
            })
            
            peak_model = residual[idx] * np.exp(-(r_centers - best_r)**2 / (2 * sigma_est**2))
            residual = np.maximum(residual - peak_model * 0.9, 0)
            
            if np.max(residual[valid_mask]) < bg_density * 1.01:
                break
        
        return detected
    
    def _estimate_sigma_fwhm(self, r_centers: np.ndarray, density: np.ndarray,
                             r0: float, window: float = 2.0) -> float:
        """Estimate peak width using FWHM method.
        
        For overlapping peaks, this can overestimate sigma. Use a smaller
        window and be more conservative.
        """
        # Use a smaller window to avoid including neighboring peaks
        window = min(window, 0.5)  # At most 0.5 mm window
        mask = (r_centers > r0 - window) & (r_centers < r0 + window)
        r_local = r_centers[mask]
        rho_local = density[mask]
        
        if len(r_local) < 5:
            return 0.1  # Default to small sigma
        
        peak_val = np.max(rho_local)
        half_max = peak_val / 2
        
        above_half = r_local[rho_local > half_max]
        if len(above_half) >= 2:
            fwhm = above_half[-1] - above_half[0]
            sigma = fwhm / 2.355
            # Be conservative - don't let sigma get too large
            return np.clip(sigma, 0.05, 0.5)
        
        return 0.1  # Default to small sigma
    
    def _fit_single_peak_abel(self, r_centers: np.ndarray, counts: np.ndarray,
                               initial_r0: float, initial_sigma: float) -> dict:
        """
        Fit a single peak using Abel projection model.
        
        Physics: The counts distribution is H(r) = 2πr·dr·A·abel(r, r0, σ) + bg
        where abel(r, r0, σ) is the Abel projection of a Gaussian shell.
        
        This correctly accounts for the Abel projection shift, which is significant
        for inner peaks (small r0) where the shift can be ~8% of r0.
        """
        dr = r_centers[1] - r_centers[0]
        
        # Better initial sigma estimate from FWHM
        _, _, density = self._compute_radial_distribution()
        sigma_fwhm = self._estimate_sigma_fwhm(r_centers, density, initial_r0)
        sigma_init = max(sigma_fwhm, 0.1)  # At least 0.1 mm
        
        def model(params):
            r0, sigma, amp, bg = params
            abel = abel_projection(r_centers, r0, sigma)
            model_counts = 2 * np.pi * r_centers * dr * amp * abel + bg
            return np.maximum(model_counts, 0.1)
        
        def loss(params):
            pred = model(params)
            # Poisson-like weighting
            weights = 1.0 / np.sqrt(np.maximum(pred, 1) + 1)
            residual = (counts - pred) * weights
            
            # Regularization
            r0, sigma = params[0], params[1]
            reg = 0
            
            # Keep sigma reasonable - penalize very small sigma
            if sigma < 0.1:
                reg += 500 * (0.1 - sigma)**2
            if sigma > 2.0:
                reg += 100 * (sigma - 2.0)**2
            
            # Penalize r0 drifting too far from initial (but allow Abel correction)
            r0_drift = abs(r0 - initial_r0) / initial_r0
            if r0_drift > 0.1:  # More than 10% drift
                reg += 200 * (r0_drift - 0.1)**2
            
            return np.sum(residual**2) + reg
        
        # Initial guess
        amp_init = np.max(counts) / (2 * np.pi * initial_r0 * dr + 1e-10)
        bg_init = np.percentile(counts, 10)
        x0 = [initial_r0, sigma_init, amp_init, bg_init]
        
        # Bounds - tighter for outer peaks, wider for inner peaks
        # Inner peaks need more room to correct for Abel shift
        if initial_r0 < 5:  # Inner peak
            r0_low = max(0.5, initial_r0 * 0.85)
            r0_high = min(self.r_max, initial_r0 * 1.15)
        else:  # Middle/outer peak
            r0_low = max(0.5, initial_r0 * 0.92)
            r0_high = min(self.r_max, initial_r0 * 1.08)
        
        bounds = [
            (r0_low, r0_high),
            (0.05, 3.0),  # Minimum sigma 0.05 mm
            (0, None),
            (0, None)
        ]
        
        try:
            result = minimize(loss, x0, bounds=bounds, method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-8})
            fitted_r0, fitted_sigma, fitted_amp, _ = result.x
        except:
            fitted_r0, fitted_sigma, fitted_amp = initial_r0, sigma_init, amp_init
        
        return {
            'r0': fitted_r0,
            'sigma': fitted_sigma,
            'amp': fitted_amp
        }
    
    def _refine_peak_position(self, r0_init: float, sigma_est: float) -> float:
        """
        Refine peak position using weighted centroid of particle radii.
        
        This is more accurate than histogram-based detection because it uses
        the actual particle positions, not binned data.
        
        Physics: The true peak position is the centroid of particles within
        a window around the detected peak, weighted by proximity to peak.
        """
        # Use a window of ~3 sigma around the detected peak
        window = max(3 * sigma_est, 1.0)
        mask = (self.r >= r0_init - window) & (self.r < r0_init + window)
        r_local = self.r[mask]
        
        if len(r_local) < 100:
            return r0_init
        
        # Gaussian weights centered on initial estimate
        weights = np.exp(-(r_local - r0_init)**2 / (2 * sigma_est**2))
        
        # Weighted centroid
        r0_refined = np.sum(r_local * weights) / np.sum(weights)
        
        # Don't drift too far from initial estimate
        if abs(r0_refined - r0_init) > 0.5:
            return r0_init
        
        return r0_refined
    
    def _fit_peaks_physics(self, r_centers: np.ndarray, counts: np.ndarray,
                           peak_candidates: List[dict]) -> List[dict]:
        """
        Fit peaks using physics-based model with Abel projection.
        
        Key improvements v4.1:
        1. Use Abel projection model for fitting (accounts for projection shift)
        2. Fit counts directly with H(r) = 2πr·dr·A·abel(r, r0, σ) + bg
        3. This correctly handles inner peaks where Abel shift is significant
        4. For single peak, use dedicated single-peak fitting
        5. Improved handling of overlapping peaks with regularization
        """
        n_peaks = len(peak_candidates)
        if n_peaks == 0:
            return []
        
        dr = r_centers[1] - r_centers[0]
        peak_candidates = sorted(peak_candidates, key=lambda p: p['r'])
        
        # For single peak, use dedicated Abel-corrected fitting
        if n_peaks == 1:
            p = peak_candidates[0]
            fitted = self._fit_single_peak_abel(r_centers, counts, p['r'], p.get('sigma', 0.3))
            return [fitted]
        
        # For multiple peaks, use multi-peak Abel fitting
        init_r0s = [p['r'] for p in peak_candidates]
        
        # Sigma constraints based on peak separation
        separations = [peak_candidates[i+1]['r'] - peak_candidates[i]['r']
                      for i in range(n_peaks - 1)]
        min_sep = min(separations) if separations else 1.0
        sigma_max = max(min_sep / 2.0, 0.15)  # Increased minimum
        sigma_init = min(min_sep / 3.0, 0.3)
        
        def model_counts(params):
            model_density = np.zeros_like(r_centers, dtype=float)
            for i in range(n_peaks):
                r0 = params[3*i]
                sigma = params[3*i + 1]
                amp = params[3*i + 2]
                model_density += amp * abel_projection(r_centers, r0, sigma)
            bg = params[-1]
            model_density += bg
            model_counts = 2 * np.pi * np.maximum(r_centers, 0.05) * dr * model_density
            return np.maximum(model_counts, 0.1)
        
        def loss(params):
            pred = model_counts(params)
            weights = 1.0 / np.sqrt(pred + 10)
            residual = (counts - pred) * weights
            
            # Regularization
            reg = 0
            for i in range(n_peaks):
                r0 = params[3*i]
                sigma = params[3*i + 1]
                
                # Penalize sigma being too small or too large
                if sigma < 0.08:  # Minimum sigma
                    reg += 1000 * (0.08 - sigma)**2
                if sigma > sigma_max:
                    reg += 200 * (sigma - sigma_max)**2
                
                # Penalty for r0 drifting from detected position
                # Allow more drift for inner peaks (Abel correction)
                if init_r0s[i] < 5:  # Inner peak
                    max_drift = 0.15  # 15% allowed
                else:
                    max_drift = 0.08  # 8% allowed
                
                r0_drift = abs(r0 - init_r0s[i]) / init_r0s[i]
                if r0_drift > max_drift:
                    reg += 500 * (r0_drift - max_drift)**2
            
            # Penalize peaks getting too close (overlap regularization)
            for i in range(n_peaks - 1):
                r0_i = params[3*i]
                r0_j = params[3*(i+1)]
                sigma_i = params[3*i + 1]
                sigma_j = params[3*(i+1) + 1]
                min_allowed_sep = (sigma_i + sigma_j) * 1.5
                actual_sep = r0_j - r0_i
                if actual_sep < min_allowed_sep:
                    reg += 300 * (min_allowed_sep - actual_sep)**2
            
            return np.sum(residual**2) + reg
        
        init_params = []
        bounds = []
        
        for p in peak_candidates:
            r0 = p['r']
            sigma = np.clip(p.get('sigma', sigma_init), 0.08, sigma_max)
            amp = p['amplitude']
            init_params.extend([r0, sigma, amp])
            
            # Bounds depend on radial position - inner peaks need more room for Abel correction
            if r0 < 5:  # Inner peak
                r0_low = max(r0 * 0.85, 0.3)
                r0_high = min(r0 * 1.20, self.r_max - 0.2)
            else:  # Middle/outer peak
                r0_low = max(r0 * 0.92, 0.3)
                r0_high = min(r0 * 1.08, self.r_max - 0.2)
            bounds.extend([
                (r0_low, r0_high),
                (0.05, sigma_max),
                (0, None)
            ])
        
        bg_init = np.percentile(counts / (2 * np.pi * r_centers * dr + 1e-10), 10)
        init_params.append(max(bg_init, 0))
        bounds.append((0, None))
        
        try:
            result = minimize(loss, init_params, bounds=bounds, method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-7})
            best_params = result.x
        except:
            best_params = init_params
        
        fitted_peaks = []
        for i in range(n_peaks):
            fitted_peaks.append({
                'r0': best_params[3*i],
                'sigma': best_params[3*i + 1],
                'amp': best_params[3*i + 2]
            })
        
        return fitted_peaks
    
    def _estimate_beta_physics(self, r0: float, sigma: float) -> float:
        """
        Estimate β parameter using physics-correct angular analysis.
        
        Physics:
        - 3D angular distribution: I(θ_3D) ∝ 1 + β·P₂(cos θ_3D)
        - For polarization along Y-axis in XY detector plane:
          cos(θ_3D) = sin(θ_XY), so I(θ_XY) ∝ 1 + β·P₂(sin θ_XY)
        - β is constrained: -1 ≤ β ≤ 2
        """
        # Use a wider window for β estimation - at least 1.5 mm or 3σ
        window = max(3.0 * sigma, 1.5)
        mask = (self.r >= r0 - window) & (self.r < r0 + window)
        theta_peak = self.theta[mask]
        n_points = np.sum(mask)
        
        if n_points < 50:
            return 0.0
        
        # Method 1: FFT analysis
        beta_fft = self._beta_from_fft(theta_peak, n_points)
        
        # Method 2: Direct fitting (more accurate, especially for positive β)
        beta_fit = self._beta_from_fit(theta_peak, n_points)
        
        # Combine results - prefer fit method as it's more accurate
        if np.isnan(beta_fft) and np.isnan(beta_fit):
            return 0.0
        elif np.isnan(beta_fft):
            beta = beta_fit
        elif np.isnan(beta_fit):
            beta = beta_fft
        else:
            # The fit method is more accurate for all β values
            # FFT systematically underestimates positive β
            # Use fit as primary method
            beta = beta_fit
        
        # Enforce physical constraint: -1 ≤ β ≤ 2
        return np.clip(beta, self.BETA_MIN, self.BETA_MAX)
    
    def _beta_from_fft(self, theta: np.ndarray, n_points: int) -> float:
        """
        Estimate β using FFT - extract cos(2θ) component directly.
        
        Physics:
        I(θ) ∝ 1 + β·P₂(sin θ) = 1 + β·(3sin²θ - 1)/2
        
        Using sin²θ = (1 - cos(2θ))/2:
        I(θ) = (1 + β/4) - (3β/4)·cos(2θ)
        
        So: β = -4·(a₂/a₀) / 3
        where a₀ is DC component and a₂ is cos(2θ) amplitude.
        """
        n_bins = np.clip(int(np.sqrt(n_points) * 1.5), 24, 180)
        hist, _ = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi))
        
        # Light smoothing
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=max(1.0, n_bins / 80))
        
        # FFT
        fft = np.fft.fft(hist_smooth)
        
        # DC component (a₀) and cos(2θ) component (a₂)
        a0 = np.real(fft[0]) / n_bins
        a2 = np.real(fft[2]) * 2 / n_bins  # Factor 2 for proper normalization
        
        if a0 < 1e-10:
            return np.nan
        
        # Simple formula: β = -4·(a₂/a₀) / 3
        beta = -4 * (a2 / a0) / 3
        
        return np.clip(beta, -1.0, 2.0)
    
    def _beta_from_fit(self, theta: np.ndarray, n_points: int) -> float:
        """Estimate β by fitting I(θ) = A(1 + β·P₂(sin θ))."""
        n_bins = np.clip(int(np.sqrt(n_points)), 20, 90)
        hist, bin_edges = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi))
        theta_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        counts = hist.astype(float)
        
        if np.sum(counts) < 10:
            return np.nan
        
        def model(theta, A, beta):
            return A * (1 + beta * P2(np.sin(theta)))
        
        try:
            sigma = np.sqrt(np.maximum(counts, 1))
            popt, _ = curve_fit(
                model, theta_centers, counts,
                p0=[np.mean(counts), 0.0],
                sigma=sigma,
                bounds=([0, -1], [np.inf, 2]),
                maxfev=2000
            )
            return popt[1]
        except:
            return np.nan
    
    def reconstruct(self, n_peaks: int = None, verbose: bool = True) -> List[PeakResult]:
        """Main reconstruction method."""
        if verbose:
            print("=" * 60)
            print("Physics-First VMI Reconstruction v3.0")
            print("=" * 60)
            print(f"Events: {self.n_events:,}")
            print(f"Center: ({self.center[0]:.3f}, {self.center[1]:.3f}) mm")
            print(f"r_max: {self.r_max:.2f} mm")
            print(f"Bin size: {self.dr:.3f} mm")
        
        r_centers, counts, density = self._compute_radial_distribution()
        candidates = self._detect_peaks_iterative(r_centers, density, max_peaks=7)
        
        if verbose:
            print(f"Detected {len(candidates)} peak candidates")
        
        if n_peaks is None:
            n_peaks = len(candidates)
        else:
            n_peaks = min(n_peaks, len(candidates))
        
        if n_peaks == 0:
            if verbose:
                print("No peaks detected!")
            self.peaks = []
            return []
        
        candidates = sorted(candidates, key=lambda p: p['prominence'], reverse=True)
        selected = candidates[:n_peaks]
        selected = sorted(selected, key=lambda p: p['r'])
        
        if verbose:
            peak_positions_str = [f"{p['r']:.2f}" for p in selected]
            print(f"Selected {n_peaks} peaks at r = {peak_positions_str}")
        
        fitted = self._fit_peaks_physics(r_centers, counts, selected)
        
        self.peaks = []
        for i, p in enumerate(fitted):
            r0 = p['r0']
            sigma = p['sigma']
            amp = p['amp']
            beta = self._estimate_beta_physics(r0, sigma)
            
            self.peaks.append(PeakResult(
                r0=r0, sigma=sigma, amp=amp, beta=beta,
                r0_err=0, sigma_err=0, beta_err=0
            ))
            
            if verbose:
                print(f"Peak {i+1}: r0={r0:.3f} mm, σ={sigma:.3f} mm, β={beta:.2f}")
        
        if verbose:
            print("=" * 60)
        
        return self.peaks
    
    def summary(self):
        """Print results summary"""
        print("=" * 60)
        print(f"{'Peak':<6} {'r0 (mm)':<12} {'σ (mm)':<12} {'β':<12}")
        print("-" * 60)
        for i, p in enumerate(self.peaks):
            print(f"{i+1:<6} {p.r0:<12.3f} {p.sigma:<12.3f} {p.beta:<12.2f}")
        print("=" * 60)




# =============================================================================
# Test Reporter
# =============================================================================

class TestReporter:
    """测试报告生成器"""
    
    def __init__(self, results: List[EvaluationResult], 
                 test_cases: List[TestCase],
                 summary: TestSummary = None):
        self.results = results
        self.test_cases = test_cases
        self.summary = summary
    
    def generate_summary_table(self) -> pd.DataFrame:
        """生成汇总表格"""
        data = []
        for tc, result in zip(self.test_cases, self.results):
            data.append({
                'case_id': tc.case_id,
                'n_peaks': tc.n_peaks,
                'event_count': tc.event_count,
                'separation': tc.peak_separation,
                'beta_range': tc.beta_range,
                'amp_ratio': tc.amplitude_ratio,
                'sigma_range': tc.sigma_range,
                'r_position': tc.r_position,
                'noise': tc.noise_level,
                'detected': result.n_detected_peaks,
                'matched': result.n_matched_peaks,
                'r0_err_%': result.mean_r0_error,
                'sigma_err_%': result.mean_sigma_error,
                'beta_err': result.mean_beta_error,
                'passed': result.passed,
                'time_s': result.execution_time,
            })
        
        return pd.DataFrame(data)
    
    def generate_error_plots(self, save_dir: str = None):
        """生成误差图表"""
        df = self.generate_summary_table()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 误差 vs 事件数
        ax = axes[0, 0]
        for metric, label in [('r0_err_%', 'r0'), ('sigma_err_%', 'σ'), ('beta_err', 'β')]:
            grouped = df.groupby('event_count')[metric].mean()
            ax.plot(grouped.index, grouped.values, 'o-', label=label)
        ax.set_xscale('log')
        ax.set_xlabel('Event Count')
        ax.set_ylabel('Error')
        ax.set_title('Error vs Event Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 误差 vs 峰分离度
        ax = axes[0, 1]
        sep_order = ['well', 'moderate', 'overlap']
        for metric, label in [('r0_err_%', 'r0'), ('sigma_err_%', 'σ'), ('beta_err', 'β')]:
            grouped = df.groupby('separation')[metric].mean()
            values = [grouped.get(s, 0) for s in sep_order]
            ax.plot(sep_order, values, 'o-', label=label)
        ax.set_xlabel('Peak Separation')
        ax.set_ylabel('Error')
        ax.set_title('Error vs Peak Separation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 误差 vs β 范围
        ax = axes[0, 2]
        beta_order = ['negative', 'zero', 'positive']
        for metric, label in [('r0_err_%', 'r0'), ('sigma_err_%', 'σ'), ('beta_err', 'β')]:
            grouped = df.groupby('beta_range')[metric].mean()
            values = [grouped.get(b, 0) for b in beta_order]
            ax.plot(beta_order, values, 'o-', label=label)
        ax.set_xlabel('Beta Range')
        ax.set_ylabel('Error')
        ax.set_title('Error vs Beta Range')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 误差 vs 径向位置
        ax = axes[1, 0]
        pos_order = ['inner', 'middle', 'outer']
        for metric, label in [('r0_err_%', 'r0'), ('sigma_err_%', 'σ'), ('beta_err', 'β')]:
            grouped = df.groupby('r_position')[metric].mean()
            values = [grouped.get(p, 0) for p in pos_order]
            ax.plot(pos_order, values, 'o-', label=label)
        ax.set_xlabel('Radial Position')
        ax.set_ylabel('Error')
        ax.set_title('Error vs Radial Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 误差 vs 峰数量
        ax = axes[1, 1]
        for metric, label in [('r0_err_%', 'r0'), ('sigma_err_%', 'σ'), ('beta_err', 'β')]:
            grouped = df.groupby('n_peaks')[metric].mean()
            ax.plot(grouped.index, grouped.values, 'o-', label=label)
        ax.set_xlabel('Number of Peaks')
        ax.set_ylabel('Error')
        ax.set_title('Error vs Number of Peaks')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 通过率饼图
        ax = axes[1, 2]
        passed = df['passed'].sum()
        failed = len(df) - passed
        ax.pie([passed, failed], labels=['Passed', 'Failed'], 
               autopct='%1.1f%%', colors=['green', 'red'])
        ax.set_title(f'Pass Rate: {passed}/{len(df)}')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f'{save_dir}/error_plots.png', dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def identify_failure_conditions(self) -> List[dict]:
        """识别失败条件"""
        failures = []
        for tc, result in zip(self.test_cases, self.results):
            if not result.passed:
                failures.append({
                    'case_id': tc.case_id,
                    'conditions': tc.to_dict(),
                    'errors': {
                        'r0': result.mean_r0_error,
                        'sigma': result.mean_sigma_error,
                        'beta': result.mean_beta_error,
                    },
                    'peak_detection': {
                        'true': result.n_true_peaks,
                        'detected': result.n_detected_peaks,
                        'missed': result.n_missed_peaks,
                        'false_pos': result.n_false_positives,
                    }
                })
        return failures
    
    def compute_pass_rate(self) -> dict:
        """计算通过率"""
        total = len(self.results)
        
        r0_pass = sum(1 for r in self.results if r.mean_r0_error <= 5)
        sigma_pass = sum(1 for r in self.results if r.mean_sigma_error <= 15)
        beta_pass = sum(1 for r in self.results if r.mean_beta_error <= 0.2)
        overall_pass = sum(1 for r in self.results if r.passed)
        
        return {
            'r0_pass_rate': r0_pass / total * 100,
            'sigma_pass_rate': sigma_pass / total * 100,
            'beta_pass_rate': beta_pass / total * 100,
            'overall_pass_rate': overall_pass / total * 100,
        }
    
    def save_results(self, filepath: str):
        """保存结果到文件"""
        data = {
            'test_cases': [tc.to_dict() for tc in self.test_cases],
            'results': [r.to_dict() for r in self.results],
            'summary': self.summary.to_dict() if self.summary else None,
            'pass_rates': self.compute_pass_rate(),
        }
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif filepath.endswith('.csv'):
            df = self.generate_summary_table()
            df.to_csv(filepath, index=False)
        else:
            # 默认 JSON
            with open(filepath + '.json', 'w') as f:
                json.dump(data, f, indent=2, default=str)
    
    def print_summary(self):
        """打印汇总"""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        if self.summary:
            print(f"Total cases: {self.summary.total_cases}")
            print(f"Passed: {self.summary.passed_cases}")
            print(f"Failed: {self.summary.failed_cases}")
            print(f"Pass rate: {self.summary.passed_cases / self.summary.total_cases * 100:.1f}%")
            
            print("\n--- Error Statistics ---")
            print(f"r0 error:    mean={self.summary.r0_error_stats['mean']:.2f}%, "
                  f"median={self.summary.r0_error_stats['median']:.2f}%, "
                  f"max={self.summary.r0_error_stats['max']:.2f}%")
            print(f"sigma error: mean={self.summary.sigma_error_stats['mean']:.2f}%, "
                  f"median={self.summary.sigma_error_stats['median']:.2f}%, "
                  f"max={self.summary.sigma_error_stats['max']:.2f}%")
            print(f"beta error:  mean={self.summary.beta_error_stats['mean']:.3f}, "
                  f"median={self.summary.beta_error_stats['median']:.3f}, "
                  f"max={self.summary.beta_error_stats['max']:.3f}")
            
            print("\n--- Pass Rate by Factor ---")
            print("Event count:", self.summary.pass_rate_by_event_count)
            print("Separation:", self.summary.pass_rate_by_separation)
            print("Beta range:", self.summary.pass_rate_by_beta)
            print("R position:", self.summary.pass_rate_by_r_position)
            print("N peaks:", self.summary.pass_rate_by_n_peaks)
        
        print("=" * 70)


# =============================================================================
# Main Test Runner
# =============================================================================

def run_comprehensive_tests(n_cases: int = None, 
                            include_corners: bool = True,
                            use_improved: bool = True,
                            verbose: bool = True,
                            save_results: str = None) -> TestReporter:
    """运行综合测试
    
    Args:
        n_cases: 测试用例数，None 表示使用完整正交表
        include_corners: 是否包含角落用例
        use_improved: 是否使用改进的重建器
        verbose: 是否打印详细信息
        save_results: 保存结果的文件路径
        
    Returns:
        TestReporter 对象
    """
    print("=" * 70)
    print("VMI ALGORITHM COMPREHENSIVE TESTING")
    print("=" * 70)
    
    # 1. 生成测试用例
    print("\n[1/4] Generating test cases...")
    designer = OrthogonalTestDesigner()
    test_cases = designer.generate_test_cases()
    
    if include_corners:
        test_cases = designer.add_corner_cases(test_cases)
    
    if n_cases is not None and n_cases < len(test_cases):
        test_cases = test_cases[:n_cases]
    
    print(f"  Total test cases: {len(test_cases)}")
    
    # 2. 填充具体参数
    generator = TestCaseGenerator()
    test_cases = generator.fill_test_cases(test_cases)
    
    # 3. 运行测试
    print("\n[2/4] Running simulations and reconstructions...")
    runner = SimulationRunner(add_noise=True)
    evaluator = PerformanceEvaluator()
    
    results = []
    total = len(test_cases)
    
    for i, tc in enumerate(test_cases):
        if verbose:
            print(f"  [{i+1}/{total}] {tc.case_id}: n_peaks={tc.n_peaks}, "
                  f"events={tc.event_count:.0e}, sep={tc.peak_separation}")
        
        start_time = time.time()
        
        try:
            # 生成配置并运行模拟
            config = generator.generate_config(tc)
            xy_data, ground_truth = runner.run(config, tc)
            
            # 重建
            if use_improved:
                reconstructor = ImprovedVMIReconstructor(xy_data)
                peaks = reconstructor.reconstruct(n_peaks=tc.n_peaks, verbose=False)
            else:
                reconstructor = VMIReconstructor(xy_data)
                peaks = reconstructor.reconstruct(n_peaks=tc.n_peaks, verbose=False)
            
            # 评估
            result = evaluator.evaluate(peaks, ground_truth, tc)
            result.execution_time = time.time() - start_time
            
        except Exception as e:
            # 记录错误
            result = EvaluationResult(
                case_id=tc.case_id,
                n_true_peaks=tc.n_peaks,
                n_detected_peaks=0,
                n_matched_peaks=0,
                n_missed_peaks=tc.n_peaks,
                n_false_positives=0,
                mean_r0_error=100,
                mean_sigma_error=100,
                mean_beta_error=3.0,
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
        
        results.append(result)
        
        if verbose and result.passed:
            print(f"    ✓ PASSED (r0={result.mean_r0_error:.1f}%, β={result.mean_beta_error:.2f})")
        elif verbose:
            print(f"    ✗ FAILED (r0={result.mean_r0_error:.1f}%, β={result.mean_beta_error:.2f})")
    
    # 4. 汇总结果
    print("\n[3/4] Aggregating results...")
    summary = evaluator.aggregate_results(results, test_cases)
    
    # 5. 生成报告
    print("\n[4/4] Generating report...")
    reporter = TestReporter(results, test_cases, summary)
    reporter.print_summary()
    
    if save_results:
        reporter.save_results(save_results)
        print(f"\nResults saved to: {save_results}")
    
    return reporter


def quick_test(n_events: int = int(1e6), n_peaks: int = 2, 
               beta: float = 1.0, verbose: bool = True) -> dict:
    """快速单次测试
    
    Args:
        n_events: 事件数
        n_peaks: 峰数量
        beta: β 值 (所有峰使用相同的 β)
        verbose: 是否打印详细信息
        
    Returns:
        测试结果字典
    """
    # 创建简单配置 - 使用更大的能量间隔以便区分
    E_centers = [0.3 + i * 0.6 for i in range(n_peaks)]  # 更大间隔
    betas = [beta] * n_peaks
    branching_ratios = [1.0 / n_peaks] * n_peaks
    
    r_max = 15.0
    vmi_k = Config.calculate_vmi_k(max(E_centers), r_max)
    
    config = Config(
        E_centers=E_centers,
        Betas=betas,
        branching_ratios=branching_ratios,
        N_events=n_events,
        vmi_k=vmi_k,
        sigma_laser=0.02,
        psf_fwhm=0.1,
        dld_resolution=0.01,
    )
    
    # 计算真值
    mass_kg = ELECTRON_MASS_AMU * AMU_TO_KG
    true_r0s = [vmi_k * np.sqrt(2 * E * EV_TO_JOULE / mass_kg) for E in E_centers]
    true_sigmas = [config.sigma_laser / E * r for E, r in zip(E_centers, true_r0s)]
    
    if verbose:
        print(f"True r0 values: {[f'{r:.2f}' for r in true_r0s]}")
        print(f"True β values: {betas}")
    
    # 运行模拟
    xy_data, _ = run_simulation(config, output_mode='xy_dld')
    
    # 重建
    reconstructor = ImprovedVMIReconstructor(xy_data)
    peaks = reconstructor.reconstruct(n_peaks=n_peaks, verbose=verbose)
    
    # 匹配峰 - 使用最近邻匹配
    errors = []
    est_r0s = [p.r0 for p in peaks]
    
    # 贪婪匹配
    used_est = set()
    used_true = set()
    matches = []
    
    distances = []
    for i, er in enumerate(est_r0s):
        for j, tr in enumerate(true_r0s):
            distances.append((abs(er - tr), i, j))
    distances.sort()
    
    for dist, i, j in distances:
        if i not in used_est and j not in used_true:
            matches.append((i, j))
            used_est.add(i)
            used_true.add(j)
    
    # 计算匹配峰的误差
    for est_idx, true_idx in sorted(matches, key=lambda x: x[1]):
        peak = peaks[est_idx]
        true_r0 = true_r0s[true_idx]
        true_beta = betas[true_idx]
        true_sigma = true_sigmas[true_idx]
        true_amp = branching_ratios[true_idx]
        
        r0_err = abs(peak.r0 - true_r0) / true_r0 * 100
        beta_err = abs(peak.beta - true_beta)
        sigma_err = abs(peak.sigma - true_sigma) / true_sigma * 100 if true_sigma > 0 else 0
        
        errors.append({
            'peak': true_idx + 1,
            'true_r0': true_r0,
            'est_r0': peak.r0,
            'r0_error_%': r0_err,
            'true_sigma': true_sigma,
            'est_sigma': peak.sigma,
            'sigma_error_%': sigma_err,
            'true_beta': true_beta,
            'est_beta': peak.beta,
            'beta_error': beta_err,
            'true_amp': true_amp,
            'est_amp': peak.amp,
        })
        
        if verbose:
            print(f"Peak {true_idx+1}: r0 error = {r0_err:.2f}%, σ error = {sigma_err:.1f}%, β error = {beta_err:.3f}")
    
    # 报告漏检的峰
    missed = set(range(len(true_r0s))) - used_true
    for idx in missed:
        if verbose:
            print(f"Peak {idx+1}: MISSED (true r0 = {true_r0s[idx]:.2f})")
        errors.append({
            'peak': idx + 1,
            'true_r0': true_r0s[idx],
            'est_r0': None,
            'r0_error_%': 100.0,
            'true_beta': betas[idx],
            'est_beta': None,
            'beta_error': 3.0,
            'missed': True
        })
    
    # 报告误检的峰
    false_pos = set(range(len(est_r0s))) - used_est
    for idx in false_pos:
        if verbose:
            print(f"FALSE POSITIVE: detected r0 = {est_r0s[idx]:.2f}")
    
    return {
        'config': config,
        'peaks': peaks,
        'errors': errors,
        'n_matched': len(matches),
        'n_missed': len(missed),
        'n_false_pos': len(false_pos),
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        # 完整测试
        reporter = run_comprehensive_tests(
            include_corners=True,
            use_improved=True,
            verbose=True,
            save_results='vmi_test_results.json'
        )
        reporter.generate_error_plots(save_dir='.')
    else:
        # 快速测试
        print("Running quick test...")
        print("-" * 50)
        
        # 测试不同条件
        for n_events in [int(1e5), int(1e6), int(1e7)]:
            print(f"\n=== N_events = {n_events:.0e} ===")
            result = quick_test(n_events=n_events, n_peaks=2, beta=1.5, verbose=True)
        
        print("\n" + "=" * 50)
        print("To run full test suite, use: python vmi_test_framework.py full")

