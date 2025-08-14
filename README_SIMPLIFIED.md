# DE Pipeline Project - Kindle Reviews

Проект по инженерии данных для обработки отзывов на Kindle книги с использованием Apache Airflow, PySpark и Docker.

## 🚀 Автоматическая обработка данных

Система автоматически выбирает источник данных:

1. **Локальный файл** - если найден `data/original/kindle_reviews.csv`
2. **Kaggle API** - автоматическая загрузка датасета `bharadwaj6/kindle-reviews`
3. **Ошибка** - если ни один источник недоступен

## ⚙️ Быстрый старт

### 1. Запуск инфраструктуры
```bash
docker compose up -d
```

### 2. Запуск обработки данных
```bash
# Через Airflow Web UI (http://localhost:8080)
# Логин: airflow / Пароль: airflow

# Или через CLI
docker compose exec airflow-apiserver airflow dags trigger end_to_end_kindle_pipeline
```

### 3. Результат
Обработанные данные сохраняются в `output/reviews_processed.parquet/`

## 🔧 Настройка Kaggle API (опционально)

Если нет локального файла, система попытается загрузить данные с Kaggle.

1. Получите API ключ на [kaggle.com/settings](https://www.kaggle.com/settings)
2. Поместите `kaggle.json` в папку проекта
3. Установите переменные окружения:
```bash
export KAGGLE_CONFIG_DIR=/workspaces/de-pipeline-project
```

## 📁 Структура проекта

```
├── dags/                           # Airflow DAG файлы
├── data/original/                  # Локальные датасеты (опционально)
├── output/                         # Результаты обработки
├── process_data.py                 # Основная логика обработки PySpark
├── docker-compose.yaml             # Конфигурация сервисов
└── Dockerfile                      # Образ для обработки данных
```

## 🛠 Компоненты системы

- **Apache Airflow 3.0.1** - оркестрация pipeline
- **PySpark 3.5.0** - обработка больших данных
- **PostgreSQL** - метаданные Airflow
- **Redis** - брокер сообщений для Celery
- **Docker** - контейнеризация

## 📊 Обработка данных

Система выполняет:
- Загрузку и валидацию данных
- Очистку текстовых полей
- Извлечение временных признаков
- Агрегацию метрик по продуктам
- Сохранение в оптимизированном Parquet формате

## ℹ️ Мониторинг

- **Airflow Web UI**: http://localhost:8080 (airflow/airflow)
- **Логи**: `docker compose logs airflow-scheduler`
- **Статус**: `docker compose ps`

## 💡 Преимущества упрощенного подхода

- ✅ **Простота**: Больше никаких режимов - система сама решает
- ✅ **Предсказуемость**: Всегда работа с полными данными 
- ✅ **Надежность**: Автоматический fallback на Kaggle
- ✅ **Прозрачность**: Понятные ошибки если данные недоступны
