# 🔥 날씨 빅데이터를 활용한 열 수요 예측

## 📌 프로젝트 개요

### 🎯 분석 배경
기후 변화 및 에너지 수요의 변동성이 증가함에 따라, 기상 데이터를 활용한 수요 예측의 중요성이 커지고 있습니다. 본 프로젝트는 시간별·지점별 열수요와 기상 변수의 관계를 학습하고, 단기 예측 모델을 개발하여 효율적인 에너지 관리 정책 수립에 기여하는 데 목적을 둡니다.

---

## 🗂️ 데이터 및 전처리

### 🔹 1. 데이터 불러오기 및 기본 정제
- Pandas 라이브러리를 사용해 원본 데이터(`train_heat.csv`)를 불러옵니다. `read_csv` 과정에서 자동 생성될 수 있는 `Unnamed: 0` 컬럼을 제거하고, `replace`를 이용해 컬럼명에 포함된 공통 접두어(`train_heat.`)를 제거하여 변수명을 간결하게 만듭니다.

```python
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 1-1. 데이터 불러오기
df = pd.read_csv('train_heat.csv')

# 1-2. 불필요한 컬럼 제거
if 'Unnamed: 0' in df.columns:
    df.drop(columns='Unnamed: 0', inplace=True)

# 1-3. 컬럼명 정리
df.columns = [col.replace('train_heat.', '') for col in df.columns]
```

### 🔹2. 시간 정보 처리 및 정렬
- pd.to_datetime 함수를 사용하여 문자열 형태의 tm 컬럼을 datetime 객체로 변환합니다. 이는 시계열 분석의 기반이 되며, 시간 단위의 다양한 연산을 가능하게 합니다. 이후 sort_values를 통해 각 지점(branch_id)별로 시간순으로 데이터를 명확하게 정렬하여 시계열 연산(예: shift, rolling) 시 발생할 수 있는 데이터 순서 오류를 방지합니다.

```python
# 2-1. datetime 형식으로 변환
df['tm'] = pd.to_datetime(df['tm'], format='%Y%m%d%H')

# 2-2. 지점별, 시간순 정렬
df = df.sort_values(['branch_id', 'tm']).reset_index(drop=True)
```

### 🔹3. 이상값 및 결측 처리
- 기상 데이터에서 측정 실패를 의미하는 값인 -99.0을 NumPy의 NaN으로 일괄 변환하여 결측치로 통일합니다. 또한, 타겟 변수인 heat_demand에 결측치가 있는 행은 모델 학습 및 평가에 사용할 수 없으므로 dropna를 이용해 제거합니다. 결측 비율이 50%를 초과하여 정보성이 낮다고 판단되는 si(일사량) 컬럼은 분석에서 제외합니다.

```python
# 3-1. 이상값(–99.0)을 NaN으로 변환
df_nan = df.replace(-99.0, np.nan)

# 3-2. 정보성 낮은 'si' 컬럼 제거
df_select = df_nan.drop(columns='si')

# 3-3. 타겟 변수 결측치 제거
df_select.dropna(subset=['heat_demand'], inplace=True)
```

### 🔹4. 시간 기반 연속 변수 보간
- 기온, 풍속 등 시간에 따라 연속적으로 변하는 물리량의 결측치는 interpolate(method='time')을 사용하여 보간합니다. 이 방법은 시간 간격에 비례하여 값을 채우므로, 불규칙한 시간 간격을 가진 데이터에서도 시계열의 흐름을 왜곡하지 않고 자연스럽게 결측을 처리할 수 있습니다. groupby('branch_id')를 통해 지점별로 독립적으로 보간을 수행합니다.

```python
# 4-1. 시간(tm)을 인덱스로 설정
df_interp = df_select.set_index('tm')

# 4-2. 보간 대상 컬럼 정의
time_interp_cols = ['ta', 'ta_chi', 'ws', 'hm', 'rn_day']

# 4-3. 지점별 시간 보간 적용
df_interp[time_interp_cols] = df_interp.groupby('branch_id')[time_interp_cols].transform(
    lambda group: group.interpolate(method='time')
)
```

### 🔹5. 풍향(wd) 보간을 위한 극좌표 변환
- 풍향은 359°와 0°가 인접한 원형(Circular) 데이터로, 단순 선형 보간 시 경계에서 큰 왜곡이 발생합니다. (예: 350°와 10°의 평균이 180°로 계산되는 문제) 이를 해결하기 위해 각도(degree)를 라디안(radian)으로 변환한 후, sin과 cos 성분으로 분해하여 2차원 평면의 좌표로 변환합니다. 각 성분은 선형성을 가지므로, 이들을 개별적으로 시간 보간한 뒤 원본 wd 컬럼은 제거합니다.

```python
# 5-1. 각도를 라디안으로 변환 후 sin, cos 성분 생성
wd_rad = np.deg2rad(df_interp['wd'])
df_interp['wd_sin'] = np.sin(wd_rad)
df_interp['wd_cos'] = np.cos(wd_rad)

# 5-2. 각 성분을 지점별로 시간 보간
df_interp['wd_sin'] = df_interp.groupby('branch_id')['wd_sin'].transform(lambda g: g.interpolate(method='time'))
df_interp['wd_cos'] = df_interp.groupby('branch_id')['wd_cos'].transform(lambda g: g.interpolate(method='time'))

# 5-3. 원본 풍향(wd) 컬럼 제거
df_interp.drop(columns='wd', inplace=True)
```

### 🔹 6. 강수 이벤트 생성 및 습도 조건부 보정
- 설명: 강수량(rn_hr1)의 결측치는 대부분 강수가 없었음을 의미하므로 0으로 채웁니다. 누적 강수량(rn_day)의 시간별 차이를 계산하여 rain_flag를 생성, 강수 발생 여부를 명확히 합니다. 습도(hm)는 강수 여부에 따라 분포가 크게 달라지므로, rain_flag를 기준으로 지점별 평균(비 오는 날) 또는 중앙값(비 오지 않는 날)을 계산하여 결측치를 정교하게 보정합니다.

```python
# 6-1. 시간당 강수량 결측치를 0으로 처리
df_interp['rn_hr1'] = df_interp['rn_hr1'].fillna(0)

# 6-2. 강수 이벤트(rain_flag) 생성
df_interp['rn_day_diff'] = df_interp.groupby('branch_id')['rn_day'].diff().fillna(0)
df_interp['rain_flag'] = ((df_interp['rn_hr1'] > 0) | (df_interp['rn_day_diff'] > 0)).astype(int)

# 6-3. 강수 여부에 따른 지점별 습도 통계 계산
hm_rainy_mean = df_interp[df_interp['rain_flag'] == 1].groupby('branch_id')['hm'].mean()
hm_dry_median = df_interp[df_interp['rain_flag'] == 0].groupby('branch_id')['hm'].median()

# 6-4. 조건에 따라 습도 결측치 채우기
def fill_hm_branchwise(row):
    if pd.notnull(row['hm']):
        return row['hm']
    branch = row['branch_id']
    if row['rain_flag'] == 1:
        return hm_rainy_mean.get(branch, df_interp['hm'].mean()) # 해당 지점 통계 없으면 전체 평균
    else:
        return hm_dry_median.get(branch, df_interp['hm'].median()) # 해당 지점 통계 없으면 전체 중앙값

df_interp['hm'] = df_interp.apply(fill_hm_branchwise, axis=1)
```

## 🤖 모델링 및 예측

### 🔹1. 로그 변환
- 분포가 한쪽으로 치우친(skewed) 변수들은 모델의 예측 성능을 저하시킬 수 있습니다. np.log1p (log(1+x)) 변환을 적용하여 데이터 분포를 정규분포에 가깝게 만들어 모델의 안정성과 성능을 향상시킵니다. 타겟 변수인 heat_demand를 포함하여 ws, rn_hr1 등에 적용합니다.

```python
df_final = df_interp.reset_index()

# 1-1. 로그 변환 대상 컬럼
log_cols = ['heat_demand', 'ws', 'rn_hr1', 'rn_day']

# 1-2. 새로운 로그 변환 컬럼 생성
for col in log_cols:
    df_final[col + '_log'] = np.log1p(df_final[col])
```

### 🔹2. 시계열 파생 변수 생성
- 시계열 데이터의 핵심은 과거의 정보가 현재에 영향을 미친다는 점입니다. shift() 함수를 이용해 과거 시점의 열수요 값을 가져와 Lag 변수를 생성하고, rolling().mean()을 이용해 특정 기간 동안의 평균 추세를 나타내는 이동평균(Moving Average) 변수를 생성합니다. 두 변수 모두 groupby('branch_id')를 통해 지점별로 독립적으로 계산됩니다.

```python
# 2-1. Lag 변수 생성
target_log = 'heat_demand_log'
for lag in [1, 2, 3, 6, 12, 24]:
    df_final[f'lag_{lag}'] = df_final.groupby('branch_id')[target_log].shift(lag)

# 2-2. 이동평균(MA) 변수 생성
for ma in [3, 6, 12, 24]:
    df_final[f'ma_{ma}'] = df_final.groupby('branch_id')[target_log].transform(
        lambda x: x.rolling(window=ma, min_periods=1).mean()
    )
```

### 🔹3. 시간 및 주기 변수 생성
- 열 수요는 시간에 따라 뚜렷한 패턴(일별, 계절별, 주중/주말)을 보입니다. dt 접근자를 이용해 시간, 월, 요일, 주말 여부 등 시간 기반 파생 변수를 생성합니다. 또한, 23시와 0시의 연속성을 모델이 이해할 수 있도록 시간(hour) 변수를 sin/cos으로 변환하여 주기성을 인코딩합니다.

```python
# 3-1. 시간 기반 파생 변수 생성
df_final['hour'] = df_final['tm'].dt.hour
df_final['month'] = df_final['tm'].dt.month
df_final['weekday'] = df_final['tm'].dt.weekday
df_final['weekend'] = (df_final['weekday'] >= 5).astype(int)

# 3-2. 시간의 주기성 인코딩
df_final['hour_sin'] = np.sin(2 * np.pi * df_final['hour'] / 24)
df_final['hour_cos'] = np.cos(2 * np.pi * df_final['hour'] / 24)
```

### 🔹4. 도메인 기반 파생 변수 생성
- 도메인 지식을 활용하여 모델의 예측력을 높일 수 있는 변수를 추가합니다. 특정 기준 온도(예: 18℃) 이하로 기온이 떨어질 때의 차이를 계산하여 난방 필요도(Heating Degree Days, HDD) 지표를 생성하고, 기온과 시간대의 복합적인 영향을 반영하기 위해 상호작용 변수를 만듭니다.

```python
# 4-1. 난방 필요도(HDD) 지표 생성
df_final['hdd_18'] = np.maximum(0, 18 - df_final['ta'])

# 4-2. 기온과 시간의 상호작용 변수 생성
df_final['temp_hour_interaction'] = df_final['ta'] * df_final['hour']

# 4-3. 최종 결측치 처리 (시계열 변수 생성으로 인한 초기 NaN)
df_final.fillna(method='bfill', inplace=True)
df_final.fillna(method='ffill', inplace=True)
```

### 🔹5. 지점별 모델 학습 및 저장
- 지점마다 다른 수요 패턴을 가질 것을 가정하여, RandomForestRegressor 모델을 지점별로 개별 학습하는 전략을 사용합니다. 각 지점의 데이터로 모델을 학습시킨 후, joblib 라이브러리를 사용해 학습된 모델을 파일로 저장하여 예측 시 재사용할 수 있도록 관리합니다.

```python
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import os

# 5-1. 모델 저장 경로 생성
os.makedirs('models', exist_ok=True)

# 5-2. 피처 및 타겟 정의
features = [col for col in df_final.columns if col not in ['tm', 'branch_id', 'heat_demand', 'heat_demand_log']]
target = 'heat_demand_log'

# 5-3. 지점별 모델 학습 루프
for branch_id in df_final['branch_id'].unique():
    print(f"--- 지점 [{branch_id}] 모델 학습 시작 ---")

    # 현재 지점 데이터 필터링
    branch_df = df_final[df_final['branch_id'] == branch_id]
    X_train = branch_df[features]
    y_train = branch_df[target]

    # 모델 정의 및 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=5)
    model.fit(X_train, y_train)

    # 모델 저장
    dump(model, f'models/rf_model_{branch_id}.joblib')
```

## 📊 분석 결과 요약
- 주요 예측 성능: 최종 검증 결과, **RMSE(Root Mean Squared Error)**는 약 19.09로, 전체 열수요 분포 대비 비교적 낮은 오차 수준을 보여 실무 적용 가능성을 확인했습니다.
- 주요 기여 변수: 장기 평균 기온(예: 7일 이동평균), 시간-기온 상호작용, 주기성을 인코딩한 시간 변수(hour_cos) 등이 예측력 향상에 크게 기여했습니다.
- 모델링 전략의 유효성: 지점별 개별 모델링 전략이 각 지역의 고유한 특성을 효과적으로 학습하는 데 유효했음을 확인했습니다.

## 📂 사용 기술 스택
- 언어: Python
- 라이브러리: Pandas, NumPy, Scikit-learn, Joblib
- 주요 모델: RandomForestRegressor
- 핵심 기법: 시계열 보간, 로그 변환, 파생 변수 생성(Lag, MA), 주기성 인코딩
