# 🔥 날씨 빅데이터를 활용한 열 수요 예측

## 📌 프로젝트 개요

### 🎯 분석 배경  
기후 변화 및 에너지 수요의 변동성이 증가함에 따라, 기상 데이터를 활용한 수요 예측의 중요성이 커지고 있음.  
본 프로젝트는 시간별·지점별 열수요와 기상 변수의 관계를 학습하고, 단기 예측 모델을 개발하는 데 목적을 둠.

---

## 🗂️ 데이터 및 전처리

### 🔹 1. 데이터 불러오기 및 컬럼 정리
```python
df = pd.read_csv('train_heat.csv')
df.drop(columns='Unnamed: 0', inplace=True)
df.columns = [col.replace('train_heat.', '') for col in df.columns]
```
- 원본 데이터를 불러온 뒤 자동 생성된 인덱스 컬럼을 제거
- 컬럼명에 포함된 접두어를 제거해 깔끔한 형태로 정리

### 🔹 2. 시간 정보 처리 및 정렬
```python
df['tm'] = pd.to_datetime(df['tm'], format='%Y%m%d%H')
df = df.sort_values(['branch_id', 'tm'])
```
- 시간 데이터를 datetime 형식으로 변환해 시계열 처리를 가능하게 함
- 지점별로 시간 순으로 정렬해 안정적인 분석이 가능한 형태로 정리

### 🔹 3. 이상값 및 결측 처리
```python
df = df.replace(-99.0, np.nan)
df.dropna(subset=['heat_demand'], inplace=True)
```
- 센서 오류를 의미하는 -99.0을 NaN으로 치환해 이상값 제거
- 예측이 불가능한 타겟 결측 행은 제거 형태로 정리

### 🔹 4. 시간 기반 보간 처리
```python
df = df.set_index('tm')
time_interp_cols = ['ta', 'ta_chi', 'ws', 'hm', 'rn_day']
df[time_interp_cols] = df.groupby('branch_id')[time_interp_cols].transform(
    lambda g: g.interpolate(method='time')
)
```
- 기온, 풍속 등 연속 변수에 대해 지점별 시간 보간을 적용
- 시계열의 흐름을 유지하면서 결측을 자연스럽게 채우는 형태로 정리

### 🔹 5. 풍향 보간을 위한 극좌표 변환
```python
wd_rad = np.deg2rad(df['wd'])
df['wd_sin'] = np.sin(wd_rad)
df['wd_cos'] = np.cos(wd_rad)
df['wd_sin'] = df.groupby('branch_id')['wd_sin'].transform(lambda g: g.interpolate(method='time'))
df['wd_cos'] = df.groupby('branch_id')['wd_cos'].transform(lambda g: g.interpolate(method='time'))
df.drop(columns='wd', inplace=True)
```
- 풍향은 각도 특성상 선형 보간시 왜곡이 발생하므로 sin, cos 성분으로 변환
- 변환 후 시간 보간을 적용하고, 원본 컬럼은 제거 형태로 정리

### 🔹 6. 강수 이벤트 생성 및 습도 조건 보정
```python
df['rn_hr1'] = df['rn_hr1'].fillna(0)
df['rn_day_diff'] = df.groupby('branch_id')['rn_day'].diff().fillna(0)
df['rain_flag'] = ((df['rn_hr1'] > 0) | (df['rn_day_diff'] > 0)).astype(int)
```
- 강수량 결측은 비가 오지 않았다고 간주해 0으로 처리
- 강수량 변화 여부에 따라 rain_flag를 이진 변수 형태로 생성

```python
hm_rainy = df[df['rain_flag'] == 1].groupby('branch_id')['hm'].mean()
hm_dry = df[df['rain_flag'] == 0].groupby('branch_id')['hm'].median()
```
- 습도 결측 보정을 위해 강수 여부에 따라 조건부 평균 또는 중앙값을 계산
- 비 오는 경우 평균, 비 오지 않는 경우 중앙값을 사용하는 형태로 정리

---

## 🤖 모델링 및 예측

### 🔹 1. 로그 변환
```python
log_cols = ['heat_demand', 'ws', 'rn_hr1']
for col in log_cols:
    df[col + '_log'] = np.log1p(df[col])
```
- 왜도가 높은 열수요, 풍속, 강수량 변수에 로그 변환을 적용
- 정규분포에 가까운 형태로 변형해 학습 안정성을 높이는 형태로 정리

### 🔹 2. 시계열 파생 변수 생성
```python
for lag in [1, 2, 3, 6, 12, 24]:
    df[f'heat_lag_{lag}'] = df.groupby('branch_id')['heat_demand'].shift(lag)
```
- 과거 시점의 열수요 값을 참조하는 lag 변수를 지점별로 생성
- 시계열 흐름을 반영한 입력 변수 형태로 정리

```python
for ma in [3, 6, 12, 24]:
    df[f'heat_ma_{ma}'] = df.groupby('branch_id')['heat_demand'].rolling(ma).mean().reset_index(level=0, drop=True)
```
- 단기~중기 평균 열수요를 반영하기 위한 이동평균 변수를 생성

### 🔹 3. 시간 및 주기 변수 생성
```python
df['hour'] = df.index.hour
df['month'] = df.index.month
df['weekday'] = df.index.weekday
df['weekend'] = (df['weekday'] >= 5).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```
- 시간, 월, 요일, 주말 여부 등 시간 기반 파생 변수 생성
- 시간 주기성을 반영하기 위해 hour 컬럼에 sin, cos 변환 적용 형태로 정리

### 🔹 4. 난방 지표 및 상호작용 변수
```python
df['hdd_18'] = np.maximum(0, 18 - df['ta'])
df['temp_hour_interaction'] = df['ta'] * df['hour']
```
- 18도를 기준으로 기온이 낮을수록 높은 난방 수요를 의미하는 지표 생성
- 기온과 시간의 곱을 상호작용 변수로 만들어 복합 영향을 반영하는 형태로 정리

### 🔹 5. 지점별 모델 학습 및 저장
```python
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
dump(model, f'models/rf_{branch_id}.joblib')
```
- RandomForest를 사용해 지점별 회귀 모델을 개별 학습

---

## 📊 분석 결과 요약

- 최종 검증에서 RMSE(Root Mean Squared Error)는 19.09로 계산됨.  
- 이는 전체 열수요 분포에 비해 비교적 낮은 오차 수준으로, 실무 적용 가능성을 보여주는 수치로 평가됨.  
- 기온 기반 이동 평균, 시간-기온 상호작용, 주기 인코딩 등 주요 피처가 예측력 향상에 기여함.

## 📂 사용 기술 스택
- Python (Pandas, NumPy, Scikit-learn)
- 모델: RandomForestRegressor
- 보간: 시간 기반 선형 보간
