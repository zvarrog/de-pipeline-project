#!/usr/bin/env python3
"""
Скрипт для загрузки полного датасета Kindle Reviews с Kaggle.

Использование:
    python scripts/download_full_dataset.py

Требования:
    pip install kaggle
    export KAGGLE_USERNAME="ваш_username"
    export KAGGLE_KEY="ваш_api_key"
"""

import os
import zipfile
from pathlib import Path
import sys


def download_full_dataset():
    """Загружает полный датасет с Kaggle."""

    # Проверяем наличие kaggle
    try:
        import kaggle
    except ImportError:
        print("❌ Ошибка: Модуль 'kaggle' не установлен.")
        print("💡 Установите: pip install kaggle")
        return False

    # Проверяем учётные данные
    if not all([os.getenv('KAGGLE_USERNAME'), os.getenv('KAGGLE_KEY')]):
        print("❌ Ошибка: Не настроены учётные данные Kaggle.")
        print("💡 Настройте:")
        print("   export KAGGLE_USERNAME='ваш_username'")
        print("   export KAGGLE_KEY='ваш_api_key'")
        return False

    # Создаём необходимые папки
    data_dir = Path("data/original")
    data_dir.mkdir(parents=True, exist_ok=True)

    output_file = data_dir / "kindle_reviews.csv"

    # Проверяем, не загружен ли уже файл
    if output_file.exists():
        print(f"✅ Файл уже существует: {output_file}")
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"📊 Размер: {size_mb:.1f} MB")
        return True

    print("📥 Загружаем датасет с Kaggle...")
    print("   Датасет: bharadwaj6/kindle-reviews")
    print("   Это может занять несколько минут...")

    try:
        # Загружаем ZIP
        kaggle.api.dataset_download_files(
            'bharadwaj6/kindle-reviews',
            path='data/temp',
            unzip=False
        )

        # Распаковываем CSV
        zip_path = Path("data/temp/kindle-reviews.zip")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Ищем CSV файл
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if not csv_files:
                print("❌ CSV файл не найден в архиве")
                return False

            # Извлекаем CSV
            csv_filename = csv_files[0]
            print(f"📦 Извлекаем: {csv_filename}")

            zip_ref.extract(csv_filename, "data/temp")

            # Перемещаем в нужное место
            temp_csv = Path("data/temp") / csv_filename
            temp_csv.rename(output_file)

        # Очищаем временные файлы
        zip_path.unlink()
        Path("data/temp").rmdir()

        # Показываем статистику
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"✅ Датасет успешно загружен!")
        print(f"📁 Путь: {output_file}")
        print(f"📊 Размер: {size_mb:.1f} MB")
        print(
            f"💡 Теперь можете запустить DAG с параметром: {{'data_mode': 'full'}}")

        return True

    except Exception as e:
        print(f"❌ Ошибка при загрузке: {e}")
        return False


if __name__ == "__main__":
    success = download_full_dataset()
    sys.exit(0 if success else 1)
