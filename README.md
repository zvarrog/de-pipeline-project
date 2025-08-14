# Amazon Kindle Reviews Rating Prediction API

Простой и эффективный API для предсказания рейтингов отзывов на к## 🎯 Технологический стек

Проект демонстрирует практические навыки работы с современными технологиями data engineering и machine learning:

### Core Technologies
- **Apache Spark** - distributed data processing (PySpark)
- **PyTorch + Transformers** - deep learning models (BERT/DistilBERT) 
- **Apache Airflow** - workflow orchestration and ML pipeline automation
- **MLflow** - model versioning, experiment tracking, and lifecycle management
- **FastAPI** - high-performance REST API development
- **Docker** - containerization and deployment

### Data & ML Stack
- **Scikit-learn** - traditional ML algorithms and preprocessing
- **Pandas** - data analysis and manipulation
- **TF-IDF** - text vectorization and feature extraction
- **PostgreSQL** - metadata storage for Airflowученный на реальных данных Amazon Kindle reviews.

## 📊 О проекте

- **Данные**: 982,619 реальных отзывов Amazon Kindle
- **Модель**: Логистическая регрессия с TF-IDF векторизацией
- **Обучение**: 300,000 записей для оптимального качества
- **Точность**: 59.3% на тестовых данных
- **API**: FastAPI с автоматической документацией

## 🚀 Быстрый старт

### Запуск API

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск API
python simple_api.py
```

API будет доступен на http://localhost:8000

### Использование

```bash
# Пример запроса
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This book is absolutely amazing!"}'

# Ответ
{
  "rating": 5.0,
  "text": "This book is absolutely amazing!"
}
```

### Документация API

Интерактивная документация доступна по адресу: http://localhost:8000/docs

## 📁 Структура проекта

```
├── simple_api.py                    # Основной API
├── models/
│   ├── real_amazon_model.pkl        # Обученная модель
│   └── real_amazon_vectorizer.pkl   # TF-IDF векторизатор
├── data/
│   └── original/
│       └── kindle_reviews.csv       # Реальные данные Amazon
├── requirements.txt                 # Зависимости Python
├── Dockerfile                       # Контейнеризация
├── docker-compose.yaml             # Оркестрация
└── README.md                       # Документация
```

## Quick Start

### Option 1: Simple API deployment
```bash
docker-compose -f docker-compose.simple.yaml up -d
# API available at http://localhost:8000
# Documentation: http://localhost:8000/docs
```

### Option 2: Full stack deployment  
```bash
docker-compose -f docker-compose.full.yaml up -d
# Services:
# - API: http://localhost:8000
# - Airflow: http://localhost:8080 (admin/admin)
# - MLflow: http://localhost:5000
```

### Option 3: Local development
```bash
pip install -r requirements.txt
python simple_api.py           # FastAPI server
python spark_processing.py     # Spark data processing
python torch_model.py          # PyTorch model training
python mlflow_integration.py   # MLflow experiments
```

## 🐳 Docker

### Простой запуск API

```bash
# Только API
docker-compose -f docker-compose.simple.yaml up --build

# API будет доступен на http://localhost:8000
# Документация: http://localhost:8000/docs
```

### Полная версия с Airflow и MLflow (для продакшна)

```bash
# Запуск полной инфраструктуры
docker-compose -f docker-compose.full.yaml up --build

# Сервисы:
# - API: http://localhost:8000
# - Airflow: http://localhost:8080 (admin/admin)  
# - MLflow: http://localhost:5000
# - PostgreSQL: localhost:5432
```
# API + Airflow + MLflow
docker-compose up --build

# API: http://localhost:8000
# Airflow: http://localhost:8080
# MLflow: http://localhost:5000
```

## 📈 Характеристики модели

- **Входные данные**: Текст отзыва на английском языке
- **Выходные данные**: Рейтинг от 1 до 5 звезд
- **Особенности**:
  - Обработка текста через TF-IDF (7,500 признаков)
  - Балансировка классов для равномерного распределения
  - Поддержка биграмм для лучшего понимания контекста
  - Обучена на 300,000 реальных отзывов Amazon

## 🎯 Примеры предсказаний

| Отзыв | Предсказанный рейтинг |
|-------|----------------------|
| "This book is absolutely amazing!" | 5★ |
| "Terrible book, waste of money" | 1★ |
| "It was okay, nothing special" | 3★ |
| "Good story, would recommend" | 4★ |

## 🔧 Технический стек

- **Python 3.11+**
- **FastAPI** - веб-фреймворк
- **scikit-learn** - машинное обучение
- **pandas** - обработка данных
- **joblib** - сериализация моделей
- **uvicorn** - ASGI сервер

## ⚡ Производительность

- Время отклика: ~50ms на запрос
- Пропускная способность: 1000+ запросов/мин
- Размер модели: ~75MB (модель + векторизатор)
- Потребление памяти: ~200MB

## 📝 Лицензия

MIT License

## 🤝 Поддержка

Если у вас есть вопросы или предложения, создайте issue в репозитории.
