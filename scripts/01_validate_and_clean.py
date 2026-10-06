import pandas as pd
import numpy as np
import logging
import os
os.chdir('..')
# Виправляємо створення папок (додано '=True')
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Налаштування логування (запис і в консоль, і у файл)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/data_cleaning.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def validate_and_clean_dataset(input_path='data/incident_results.csv', output_path='data/incident_results_cleaned.csv'):
    logging.info(f"Завантаження датасету з {input_path}...")
    
    if not os.path.exists(input_path):
        logging.error(f"Файл {input_path} не знайдено! Перевірте шлях до файлу.")
        return
    
    df = pd.read_csv(input_path)
    logging.info(f"Початковий розмір датасету: {df.shape[0]} рядків, {df.shape[1]} колонок.")
    
    # 1. Перевірка на дублікати
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        logging.warning(f"Знайдено повних дублікатів рядків: {duplicates_count}. Видаляємо...")
        df = df.drop_duplicates()
    else:
        logging.info("Дублікатів не знайдено.")

    # 2. Аналіз та заповнення пропущених значень (NaN)
    missing_values = df.isnull().sum()
    cols_with_missing = missing_values[missing_values > 0]
    
    if len(cols_with_missing) == 0:
        logging.info("Пропущених значень (NaN) у датасеті не виявлено! Дані ідеальні.")
    else:
        logging.warning(f"Виявлено колонки з пропущеними значеннями:\n{cols_with_missing}")
        
        for col in cols_with_missing.index:
            missing_rows = df[df[col].isnull()].index.tolist()
            
            # Стратегія заповнення залежно від типу колонки
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()
                if pd.isna(fill_value):
                    fill_value = 0.0
                
                df[col] = df[col].fillna(fill_value)
                logging.info(f"[Числовий стовпчик: '{col}'] У рядках {missing_rows} знайдено NaN. Заповнено медіаною/нулем: {fill_value}")
                
            else:
                fill_value = "Unknown"
                df[col] = df[col].fillna(fill_value)
                logging.info(f"[Категоріальний стовпчик: '{col}'] У рядках {missing_rows} знайдено NaN. Заповнено значенням: '{fill_value}'")

    # 3. Збереження очищеного датасету
    df.to_csv(output_path, index=False)
    logging.info(f"Очищений датасет успішно збережено у {output_path}.\n")

if __name__ == "__main__":
    validate_and_clean_dataset()