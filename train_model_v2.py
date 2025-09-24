import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загрузка данных
df = pd.read_csv('features_dataset_v1.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Определение признаков и цели
# Based on the previous script and dataset inspection
features = [
    'open_interest', 'funding_rate', 'volume_derivatives', 'oi_growth_rate_24h',
    'oi_trend_confirmation_score', 'oi_to_volume_ratio_24h', 'funding_rate_ma_8h',
    'funding_rate_zscore_24h', 'funding_rate_zscore_abs', 'oi_weighted_funding_rate',
    'funding_spot_divergence'
]
target = 'y_target'

# Handle NaNs
df[features] = df[features].fillna(0)
df[target] = df[target].fillna(0)


# 2. Анализ и Визуализация Признаков
plt.style.use('ggplot')

# Признаки для визуализации
features_to_plot = ['funding_rate_zscore_abs', 'oi_growth_rate_24h']

for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df[df[target] == 0][feature], label='y_target = 0', color='blue', shade=True)
    sns.kdeplot(df[df[target] == 1][feature], label='y_target = 1', color='red', shade=True)
    plt.title(f'Распределение плотности для {feature}')
    plt.legend()
    # Adding a check to avoid errors if the feature is not in the dataframe, although it should be.
    if feature in df.columns:
        plt.savefig(f'{feature}_density_plot.png')
        print(f'График для {feature} сохранен в {feature}_density_plot.png')
    else:
        print(f"Warning: Feature '{feature}' not found for plotting.")


# 3. Улучшенное Обучение Модели
# Временной сплит
train_df = df[df.index < '2024-01-01']
test_df = df[df.index >= '2024-01-01']

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Расчет коэффициента для борьбы с дисбалансом классов
# Adding a check to prevent division by zero if there are no positive samples in the training set
if (y_train == 1).sum() > 0:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
else:
    scale_pos_weight = 1
print(f"scale_pos_weight: {scale_pos_weight}")

# Инициализация и обучение модели
lgbm = lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
lgbm.fit(X_train, y_train)

# 4. Оценка Новой Модели
y_pred = lgbm.predict(X_test)

print("\nОтчет о классификации (Улучшенная модель):")
print(classification_report(y_test, y_pred))

print("\nМатрица ошибок (Улучшенная модель):")
print(confusion_matrix(y_test, y_pred))
