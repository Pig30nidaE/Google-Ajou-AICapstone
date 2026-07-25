"""ML4 SHAP 분석 패키지

- 트리 기반 모델(sklearn 계열)과 시퀀스 모델(tf.keras)을 동일 인터페이스로 처리
- 모델 의존성 최소화: 학습 코드는 학습된 모델만 넘기면 됨

공개 API:
    ShapAnalyzer  -- 메인 진입점. 모델 타입 감지, SHAP 실행, 원본/집계/
                     DataFrame/플롯 출력 제공
    AGGREGATIONS  -- 시퀀스 모델 시간축 집계 모드 목록
"""

from .analyzer import ShapAnalyzer
from .aggregation import AGGREGATIONS

__all__ = ["ShapAnalyzer", "AGGREGATIONS"]
__version__ = "0.1.0"
