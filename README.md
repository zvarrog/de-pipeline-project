# Kindle Reviews Data Pipeline

🚀 **End-to-end data processing pipeline** для анализа отзывов о книгах Kindle с использованием Apache Spark и Apache Airflow.

## 🏗️ Архитектура

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Data      │ -> │   Apache Spark   │ -> │  Processed Data │
│ (kindle_reviews │    │   Processing     │    │   (Parquet)     │
│    .csv)        │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                         ┌──────────────────┐
                         │  Apache Airflow  │
                         │  (Orchestration) │
                         └──────────────────┘
```

## 🔧 Технический стек

- **🐋 Docker & Docker Compose** - контейнеризация
- **⚡ Apache Spark (PySpark)** - обработка больших данных
- **🌊 Apache Airflow** - оркестрация пайплайнов
- **🐘 PostgreSQL** - метаданные Airflow
- **📊 Parquet** - оптимизированное хранение данных

## 🚀 Быстрый старт

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/zvarrog/de-pipeline-project.git
cd de-pipeline-project
```

### 2. Получите датасет

Скачайте датасет Kindle Reviews:

- **Источник:** [Amazon Product Data](http://jmcauley.ucsd.edu/data/amazon/)
- **Файл:** `kindle_reviews.csv` (~670 MB)
- **Поместите в корень проекта**

### 3. Запустите инфраструктуру

```bash
# Создайте .env файл
echo AIRFLOW_UID=50000 > .env

# Запустите все сервисы
docker-compose up -d

# Создайте пользователя Airflow
docker-compose exec airflow-apiserver airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname User \
  --role Admin --email admin@example.com
```

### 4. Откройте веб-интерфейс

- **Airflow UI:** http://localhost:8080 (admin/admin)

## 📊 Обработка данных

### Что делает Spark pipeline:

1. **🔍 Парсинг данных:**

   - Извлекает `helpful_votes` и `total_votes` из строки `[x, y]`
   - Вычисляет `helpful_ratio = helpful_votes / total_votes`

2. **🧹 Очистка текста:**

   - Нормализует `reviewText` и `summary`
   - Удаляет специальные символы
   - Рассчитывает длину текста

3. **📈 Агрегация:**

   - Средний рейтинг по товарам (ASIN)
   - Средний рейтинг по пользователям
   - Статистика полезности отзывов

4. **💾 Сохранение:**
   - Результат в формате Parquet
   - Оптимизировано для аналитики

### Результат:

```
output/
└── reviews_processed.parquet/
    ├── part-00000-*.snappy.parquet
    ├── part-00001-*.snappy.parquet
    └── ...
```

## 🔄 Airflow DAG

- **DAG ID:** `kindle_data_processing`
- **Описание:** Автоматизация обработки данных
- **Триггер:** Ручной запуск
- **Контейнеры:** Изолированные Docker-контейнеры для каждой задачи

## 📁 Структура проекта

```
├── dags/                          # Airflow DAGs
│   └── process_kindle_reviews_dag.py
├── output/                        # Обработанные данные
├── logs/                          # Логи Airflow
├── config/                        # Конфигурации
├── artifacts/                     # Артефакты сборки
├── process_data.py               # Основной Spark скрипт
├── Dockerfile                    # Spark контейнер
├── docker-compose.yaml          # Инфраструктура
├── requirements.txt              # Python зависимости
└── README.md                     # Документация
```

## 🛠️ Разработка

### Локальная обработка без Airflow:

```bash
# Собрать Spark контейнер
docker build -t kindle-reviews-processor .

# Запустить обработку
docker run --rm -v "./output:/app/output" kindle-reviews-processor
```

### Мониторинг:

```bash
# Логи Airflow
docker-compose logs airflow-scheduler

# Статус контейнеров
docker-compose ps
```

## 📈 Следующие шаги

- [ ] **NLP анализ** с PyTorch/Transformers
- [ ] **Sentiment analysis** отзывов
- [ ] **API для предсказаний**
- [ ] **ML модель для рекомендаций**

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch
3. Сделайте изменения
4. Создайте Pull Request

## 📄 Лицензия

MIT License - см. [LICENSE](LICENSE) файл.

---

⭐ **Star** этот репозиторий, если он был полезен!
