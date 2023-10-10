#%%
import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import joblib

sns.set(font_scale=1.0)
import missingno as msno # 결측치 시각화하기
#%%
# Target
# patients.csv에 있는 dod 날짜와 icustays.csv에 있는 퇴원날짜가 일치하는 것을 Target으로 지정해준다.

#target
df_patients = pd.read_csv('./hosp/patients.csv.gz')
df_icustays = pd.read_csv('./icu/icustays.csv.gz')
#%%
# subject_id를 기준으로 두 데이터 프레임 병합
df_target = pd.merge(df_patients, df_icustays, on='subject_id', how='inner')
# subject_id 별로 중복된 횟수를 확인
stay_counts = df_target['subject_id'].value_counts()
# 새로운 칼럼 'subject_count'를 df_target에 추가
df_target['stay_count'] = df_target['subject_id'].map(stay_counts)
df_target['target'] = (pd.to_datetime(df_target['dod']).dt.date == pd.to_datetime(df_target['outtime']).dt.date).astype(int)


#%%
# 필요한 열만 선택합니다.
df = df_target[['subject_id', 'gender', 'anchor_age', 'stay_id', 'los', 'target']]
# 'gender' 열의 값을 안전하게 변환합니다. 'M'은 1로, 'F'는 0으로 변환합니다.
df.loc[:, 'gender'] = df['gender'].map({'M': 1, 'F': 0})

# target 값에 따른 anchor_age의 평균을 계산합니다.
average_ages = df.groupby('target')['anchor_age'].mean()
# 결과를 출력합니다.
print('age :', average_ages)

# target 값에 따른 los의 평균을 계산합니다.
average_los = df.groupby('target')['los'].mean()
# 결과를 출력합니다.
print('los :', average_los)

# target 값에 따른 anchor_age의 평균과 gender의 개수를 계산합니다.
df_target_FN = df.groupby(['target', 'gender']).agg({'anchor_age': 'mean', 'gender': 'count'})
# 결과를 출력합니다.
print('gende, age : ', df_target_FN)


# target 값에 따른 anchor_age의 평균과 gender의 개수를 계산합니다.
df_target_FN = df.groupby(['target', 'gender']).agg({'anchor_age': 'mean', 'los': 'mean'})
# 결과를 출력합니다.
print('gender, age : ', df_target_FN)


#%%
# 1. 평균 나이와 평균 병원 체류 기간에 대한 바 차트
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

sns.barplot(x='target', y='anchor_age', data=df, ax=axes[0])
axes[0].set_title('Average Age per Target Value')
axes[0].set_ylabel('Average Age')
axes[0].set_xlabel('Target')

sns.barplot(x='target', y='los', data=df, ax=axes[1])
axes[1].set_title('Average Length of Stay per Target Value')
axes[1].set_ylabel('Average Length of Stay')
axes[1].set_xlabel('Target')

plt.tight_layout()
plt.show()

# 2. 성별과 target 값에 따른 평균 나이와 평균 병원 체류 기간
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

sns.barplot(x='target', y='anchor_age', hue='gender', data=df, ax=axes[0])
axes[0].set_title('Average Age per Target and Gender')
axes[0].set_ylabel('Average Age')
axes[0].set_xlabel('Target')
axes[0].legend(title='Gender', labels=['Female', 'Male'])

sns.barplot(x='target', y='los', hue='gender', data=df, ax=axes[1])
axes[1].set_title('Average Length of Stay per Target and Gender')
axes[1].set_ylabel('Average Length of Stay')
axes[1].set_xlabel('Target')
axes[1].legend(title='Gender', labels=['Female', 'Male'])

plt.tight_layout()
plt.show()


#%%
df_target_EN = df_target[['subject_id', 'stay_id', 'target']]

# target 값이 1인 환자의 수를 센다.
target_one_count = (df_target_EN['target'] == 1).sum()
subject_id_one_count = (df_target_EN['subject_id']).nunique()

print(f"사망한 환자의 수: {target_one_count}")
print(f"환자의 수: {subject_id_one_count}")
df_target_EN

#%%
# EDA
# 미생물검사
cols_to_use = list(range(0, 17)) + list(range(18, 25))
df_micro = pd.read_csv('./hosp/microbiologyevents.csv.gz', usecols=cols_to_use)
df_micro_use=df_micro[['subject_id','spec_type_desc']]
df_micro_EDA = pd.merge(df_target_EN, df_micro_use, on='subject_id', how='left')

#%%
# 피벗 테이블 생성
df_micro_pi = df_micro_EDA.pivot_table(
    index=['subject_id'], 
    columns='spec_type_desc', 
    aggfunc='size',  # 각 조합에 대한 빈도를 계산합니다.
    fill_value=0  # NaN 값을 0으로 채웁니다.
).reset_index()

# 이제 병합을 시도합니다.
final_micro = pd.merge(df_target_EN, df_micro_pi, on='subject_id', how='left')
final_micro = final_micro.fillna(0)
final_micro

#%%
# target과 spec_type_desc에 따른 빈도 계산
melted_df = pd.melt(final_micro, id_vars=['subject_id', 'stay_id', 'target'], var_name='spec_type_desc', value_name='count')
grouped_df = melted_df.groupby(['target', 'spec_type_desc'])['count'].sum().reset_index()

# target과 spec_type_desc에 따른 전체 빈도 계산
total_counts = grouped_df.groupby('target')['count'].sum().reset_index()
total_counts.columns = ['target', 'total_count']

# 비율 계산
merged_df = pd.merge(grouped_df, total_counts, on='target')
merged_df['ratio'] = merged_df['count'] / merged_df['total_count']

# 바 차트로 시각화
plt.figure(figsize=(12, 6))
sns.barplot(x='spec_type_desc', y='ratio', hue='target', data=merged_df)
plt.xticks(rotation=90)  # x축 레이블을 90도 회전하여 가독성 향상
plt.title('Ratio of Each spec_type_desc by Target')
plt.xlabel('Spec Type Description')
plt.ylabel('Ratio')
plt.show()

#%%
#진단
df_diag_EDA=pd.read_csv('./hosp/diagnoses_icd.csv.gz')


#%%
#ICD_cdde 카테고르별 분류
def get_chapter(code):
    # Check if code is not a string and if it's NaN
    if not isinstance(code, str) and pd.isna(code):
        return "Unknown Chapter"
    
    code = str(code)[:3]  # Convert to string and extract the first three characters
    
    if code.isdigit():
        code = int(code)
        if 1 <= code <= 139:
            return "Infectious"
        elif 140 <= code <= 239:
            return "Neoplasms"
        elif 240 <= code <= 279:
            return "Endocrine"
        elif 280 <= code <= 289:
            return "Diseases of the Blood"
        elif 290 <= code <= 319:
            return "Mental Disorders"
        elif 320 <= code <= 389:
            return "Diseases of the Nervous System"
        elif 390 <= code <= 459:
            return "Diseases of the Circulatory System"
        elif 460 <= code <= 519:
            return "Diseases of the Respiratory System"
        elif 520 <= code <= 579:
            return "Diseases of the Digestive System"
        elif 580 <= code <= 629:
            return "Diseases of the Genitourinary System"
        elif 630 <= code <= 679:
            return "Complications of Pregnancy"
        elif 680 <= code <= 709:
            return "Diseases of the Skin"
        elif 710 <= code <= 739:
            return "Diseases of the Musculoskeletal System"
        elif 740 <= code <= 759:
            return "Congenital Anomalies"
        elif 760 <= code <= 779:
            return "Certain Conditions originating"
        elif 780 <= code <= 799:
            return "Ill-defined Conditions"
        elif 800 <= code <= 999:
            return "Injury and Poisoning"
        else:
            return "Unknown Chapter"
    else:
        return "other"

# Apply the function to create a new column 'icd_chapter'
df_diag_EDA['icd_chapter'] = df_diag_EDA['icd_code'].apply(get_chapter)

# Print the resulting DataFrame
df_diag_EDA



#%%
# 원-핫 인코딩
one_hot = pd.get_dummies(df_diag_EDA['icd_chapter'], prefix='icd_chapter')

# 원-핫 인코딩된 데이터프레임과 원래 데이터프레임 병합
df_diag_EDA_one_hot = pd.concat([df_diag_EDA, one_hot], axis=1)

# subject_id 별로 집계
df_diag= df_diag_EDA_one_hot.groupby('subject_id').sum().reset_index()
df_diag.drop([ 'hadm_id', 'seq_num', 'icd_version', 'icd_chapter','icd_code'], axis=1, inplace=True)


#%%
#데이터 합치기
df_diag_FN = pd.merge(df_target_EN,df_diag, on='subject_id', how='left')
df_diag_FN = df_diag_FN.fillna(0)

df_diag_FN

#%%
# target 값에 따른 각 질병 카테고리의 평균 발생 횟수 계산
mean_counts_by_target = df_diag_FN.groupby('target').mean().drop(columns=['subject_id', 'stay_id'])

# 각 target 그룹의 전체 평균 값으로 나누어 비율 계산
ratios_by_target = mean_counts_by_target.div(mean_counts_by_target.sum(axis=1), axis=0)

# 바 차트로 표현
plt.figure(figsize=(10, 6))

# Transpose the DataFrame and reset index to make 'index' a column
plot_data = ratios_by_target.transpose().reset_index()
plot_data = pd.melt(plot_data, id_vars='index', value_vars=[0, 1], var_name='target', value_name='average_ratio')

sns.barplot(data=plot_data, x='average_ratio', y='index', hue='target')
plt.ylabel('ICD Chapter')
plt.xlabel('Average Ratio')
plt.title('Average Ratio of Each ICD Chapter by Target Value')
plt.show()

#%%
#OUTPUT

# 필요한 열만 선택합니다.
df = df_target[['subject_id', 'gender', 'anchor_age', 'stay_id', 'los', 'target']]
# 'gender' 열의 값을 안전하게 변환합니다. 'M'은 1로, 'F'는 0으로 변환합니다.
df.loc[:, 'gender'] = df['gender'].map({'M': 1, 'F': 0})

# target 값에 따른 anchor_age의 평균을 계산합니다.
average_ages = df.groupby('target')['anchor_age'].mean()
# 결과를 출력합니다.
print('age :', average_ages)

# target 값에 따른 los의 평균을 계산합니다.
average_los = df.groupby('target')['los'].mean()
# 결과를 출력합니다.
print('los :', average_los)

# target 값에 따른 anchor_age의 평균과 gender의 개수를 계산합니다.
grouped = df.groupby(['target', 'gender']).agg({'anchor_age': 'mean', 'gender': 'count'})
# 결과를 출력합니다.
print('gende, age : ', grouped)


# target 값에 따른 anchor_age의 평균과 gender의 개수를 계산합니다.
grouped = df.groupby(['target', 'gender']).agg({'anchor_age': 'mean', 'los': 'mean'})
# 결과를 출력합니다.
print('gender, age : ', grouped)

#%%
df_output = pd.read_csv('./icu/outputevents.csv.gz')
df_items = pd.read_csv('./hosp/d_labitems.csv.gz')

#%%
# 먼저, stay_id로 그룹화합니다.
grouped = df_output.groupby('stay_id')

# 각 그룹에 대해 'value'의 합계와 횟수를 계산합니다.
result = grouped['value'].agg(['sum', 'count','min','max'])

# 합계를 횟수로 나누어 평균을 계산합니다.
result['average'] = result['sum'] / result['count']

# merged_df에서 'stay_id', 'los', 'target' 열만 선택합니다.
selected_columns = df_target[['subject_id', 'stay_id', 'los', 'target']].drop_duplicates()

# selected_columns를 stay_id를 기준으로 result와 병합합니다.
pre_result = pd.merge(result, selected_columns, on='stay_id', how='left')

# output이 없는 stay_id를 찾습니다.
no_output_stay_ids = set(df_target['stay_id']) - set(df_output['stay_id'])

# no_output_stay_ids를 사용하여 새 DataFrame 생성
new_data = pd.DataFrame({
    'stay_id': list(no_output_stay_ids),
    'sum': 0,
    'count': 0,
    'average': 0,
    'min':0,
    'max':0
})

# 새 데이터를 result와 병합
result = pd.concat([result.reset_index(), new_data])

# selected_columns를 stay_id를 기준으로 result와 병합합니다.
df_output_fn = pd.merge(result, df, on='stay_id', how='left')

# 칼럼 이름 변경
df_output_fn = df_output_fn.rename(columns={
    'sum': 'value_sum',
    'count': 'value_count',
    'min': 'value_min',
    'max': 'value_max',
    'average': 'value_average'
})


# 최종 결과를 확인합니다.
df_output_fnn = df_output_fn[['stay_id', 'subject_id', 'value_count','value_min','value_max','value_average']]
df_output_fnn

#%%
# df_target과 df_output을 stay_id를 기준으로 외부 조인합니다.
merged_df = pd.merge(df_target, df_output, on='stay_id', how='outer')

# output이 NaN인 행을 찾습니다. 이 행들은 df_output에 없는 행들입니다.
no_output = merged_df[merged_df['value'].isna()]

# no_output 데이터프레임에서 target과 los 값을 확인합니다.
print(no_output[['target', 'los']])

# target 값이 1인 환자의 수를 센다.
target_one_count = no_output[no_output['target'] == 1].shape[0]

print(f"배설이 없는 환자 중 target 값이 1인 환자의 수: {target_one_count}")

#%%
#OMR
df_omr = pd.read_csv('./hosp/omr.csv.gz')

#%%
# 'result_name'을 열로 변환하고 'result_value'를 값으로 사용하여 피벗합니다.
df_pivot = df_omr.pivot_table(index=['subject_id', 'chartdate'], 
                              columns='result_name', 
                              values='result_value', 
                              aggfunc='first').reset_index()


# 혈압 값을 'Systolic'과 'Diastolic'으로 분리합니다.
df_pivot[['Blood Pressure Systolic', 'Blood Pressure Diastolic']] = df_pivot['Blood Pressure'].str.split('/', expand=True)


# 제거할 열의 리스트를 생성합니다.
columns_to_drop = [
    'BMI',
    'Blood Pressure',
    'Blood Pressure Lying', 
    'Blood Pressure Sitting', 
    'Blood Pressure Standing', 
    'Blood Pressure Standing (1 min)', 
    'Blood Pressure Standing (3 mins)', 
    'Height', 
    'Height (Inches)', 
    'Weight', 
    'Weight (Lbs)', 
    'eGFR'
]

# 불필요한 열을 제거합니다.
df_pivot = df_pivot.drop(columns=columns_to_drop, errors='ignore')  # errors='ignore'를 추가하여 없는 열을 제거하려고 할 때 오류를 무시합니다.

# 결과를 확인합니다.
df_pivot

#%%
# hosp 전체
# 'Blood Pressure Systolic'과 'Blood Pressure Diastolic'를 숫자로 변환합니다.
df_pivot['Blood Pressure Systolic'] = pd.to_numeric(df_pivot['Blood Pressure Systolic'], errors='coerce')
df_pivot['Blood Pressure Diastolic'] = pd.to_numeric(df_pivot['Blood Pressure Diastolic'], errors='coerce')
df_pivot['BMI (kg/m2)'] = pd.to_numeric(df_pivot['BMI (kg/m2)'], errors='coerce')


# subject_id 별로 min, max, mean을 계산합니다.
grouped = df_pivot.groupby('subject_id').agg({
    'BMI (kg/m2)': ['min', 'max', 'mean'],
    'Blood Pressure Systolic': ['min', 'max', 'mean'],
    'Blood Pressure Diastolic': ['min', 'max', 'mean']
}).reset_index()

# MultiIndex를 Flatten 하기
grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

# 결과를 출력합니다.
grouped

#%%
# 'subject_id_'를 'subject_id'로 이름을 변경합니다.
grouped.rename(columns={'subject_id_': 'subject_id'}, inplace=True)

# 이제 병합을 시도합니다.
df_OMR_FN = pd.merge(df_target, grouped, on='subject_id', how='left')

# NaN 값을 0으로 변경합니다.
df_OMR_FN.fillna(0, inplace=True)

# 결과를 확인합니다.
df_OMR_FN


#%%
df_OMR_FN.columns
df_OMR_FNN = df_OMR_FN[['subject_id', 'stay_id', 'BMI (kg/m2)_min', 'BMI (kg/m2)_max', 'BMI (kg/m2)_mean',
       'Blood Pressure Systolic_min', 'Blood Pressure Systolic_max',
       'Blood Pressure Systolic_mean', 'Blood Pressure Diastolic_min',
       'Blood Pressure Diastolic_max', 'Blood Pressure Diastolic_mean']]
df_OMR_FNN

#%%
#데이터합치기
merged_df = pd.merge(df, final_micro, on=['stay_id', 'subject_id'], how='outer', suffixes=('', '_final_micro'))
merged_df = pd.merge(merged_df, df_diag_FN, on=['stay_id', 'subject_id', 'target'], how='outer', suffixes=('', '_df_diag_FN'))
merged_df = pd.merge(merged_df, df_output_fnn, on=['stay_id', 'subject_id'], how='outer', suffixes=('', '_df_output_fn'))
merged_df = pd.merge(merged_df, df_OMR_FNN, on=['stay_id', 'subject_id'], how='outer', suffixes=('', '_df_OMR_FN'))

merged_df.drop('target_final_micro', axis=1, inplace=True)
merged_df


#%%
merged_df.drop(columns=['stay_id','subject_id'])

#%%
# import wandb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# # wandb 초기화
# wandb.init(project="random-forest-example")

# 피처와 타겟 데이터 분리
X = merged_df.drop(columns=['target'])
y = merged_df['target']

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤포레스트 모델 초기화
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 모델 학습
rf_classifier.fit(X_train, y_train)

# 예측 확률 계산
y_probas = rf_classifier.predict_proba(X_test)

# 예측
y_pred = rf_classifier.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)

# # wandb에 정확도 기록
# wandb.log({"accuracy": accuracy})

# # wandb에 모델 저장 및 시각화
# wandb.sklearn.plot_classifier(rf_classifier, X_train, X_test, y_train, y_test, y_pred, y_probas, labels=y_test)

# # wandb 종료
# wandb.finish()


#%%
print("accuracy", accuracy)

#%%
import matplotlib.pyplot as plt
import numpy as np

# Get feature importances
importances = rf_classifier.feature_importances_

# Get the feature names
features = X_train.columns

# Sort the features based on importance
indices = np.argsort(importances)[::-1]

# Number of features to display
num_features = 20

# Plot the feature importances of the top 50
plt.figure(figsize=(15, 8))
plt.title("Top 20 Feature importances")
plt.bar(range(num_features), importances[indices][:num_features], align="center")
plt.xticks(range(num_features), [features[i] for i in indices[:num_features]], rotation=45)
plt.xlim([-1, num_features])
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.tight_layout()
plt.show()


#%%

# Get the feature names
features = X_train.columns

# Sort the features based on importance
indices = np.argsort(importances)[::-1]

# Number of features to display
num_features = 20

# Selecting the top 20 features
top_features = [features[i] for i in indices[:num_features]]

# Using only top 20 features for modeling
X_top_features = X[top_features]
y = merged_df['target']

# Splitting the data into train and test sets
X_train_top, X_test_top, y_train, y_test = train_test_split(X_top_features, y, test_size=0.2, random_state=42)

# Initializing the Random Forest model
rf_classifier_top = RandomForestClassifier(n_estimators=100, random_state=42)

# Fitting the model
rf_classifier_top.fit(X_train_top, y_train)

# Making predictions
y_pred_top = rf_classifier_top.predict(X_test_top)

# Calculating the accuracy
accuracy_top = accuracy_score(y_test, y_pred_top)

# Displaying the accuracy
print(f"Accuracy with top 20 features: {accuracy_top * 100:.2f}%")



#%%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Selecting the top 20 features plus 'gender' and 'target'
top_features_with_gender_target = ['gender', 'target'] + top_features

# Creating a new DataFrame with only the top features, 'gender', and 'target'
df_top = merged_df[top_features_with_gender_target]

# Splitting the data by gender
df_male = df_top[df_top['gender'] == 1].drop(columns=['gender'])
df_female = df_top[df_top['gender'] == 0].drop(columns=['gender'])

# Defining a function to train and evaluate the model for each gender
def train_and_evaluate(df, gender_str):
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy with top 20 features for {gender_str}: {accuracy * 100:.2f}%")

# Training and evaluating the model for each gender
train_and_evaluate(df_male, 'male')
train_and_evaluate(df_female, 'female')



#%%

# 모모델 저장하기
joblib.dump(rf_classifier, 'model.joblib')
