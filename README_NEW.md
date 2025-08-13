# DE Pipeline Project

🚀 **Современный data engineering проект** с Airflow, Spark, Docker для изучения полного MLOps стека.

## 🚀 Быстрый старт

### В GitHub Codespaces (рекомендуется)

1. Нажмите зеленую кнопку **Code** → **Codespaces** → **Create codespace**
2. Дождитесь загрузки (2-3 минуты)
3. В терминале выполните:

```bash
# Сборка Spark-образа
docker build -t kindle-reviews-processor:latest .

# Запуск Airflow стека
docker compose up -d

# Ожидание готовности (30-60 сек)
docker compose logs -f airflow-init
```

4. Откройте Airflow UI: http://localhost:8080 (логин/пароль: airflow/airflow)
5. Включите и запустите DAG `end_to_end_kindle_pipeline`

**🎯 Что произойдёт при первом запуске:**
- Проект автоматически скачает полный датасет Kindle Reviews с Kaggle (~5.2M записей, 2.9GB)
- Обработает данные с помощью PySpark
- Сохранит результат в формате Parquet в папку `output/`

## 📊 Режимы работы

### 🔄 **Auto** (по умолчанию)
Умный режим, который автоматически:
1. Ищет локальный полный файл в `data/original/kindle_reviews.csv`
2. Если не найден → автоматически скачивает с Kaggle
3. Если Kaggle недоступен → использует встроенный семпл (1K записей)

### 🔸 **Sample** (для быстрого тестирования)
```bash
# Принудительно использовать семпл
airflow dags trigger end_to_end_kindle_pipeline --conf '{"data_mode": "sample"}'
```

## 📂 Настройка Kaggle API (опционально)

Для автоматической загрузки с Kaggle нужен API токен:

1. Создайте аккаунт на https://kaggle.com
2. Перейдите в Profile → Account → API → Create New Token
3. Скачается файл `kaggle.json`
4. Поместите учётные данные в переменные окружения:

```bash
export KAGGLE_USERNAME="ваш_username"
export KAGGLE_KEY="ваш_api_key"
```

**💡 Если Kaggle API не настроен** - проект автоматически переключится на встроенный семпл.

## 📊 Производительность

| Режим | Записей | Размер | Время обработки | RAM |
|-------|---------|--------|-----------------|-----|
| Sample | 1,000 | 701KB | ~30 сек | 2GB |
| Auto (Kaggle) | 5.2M | 2.9GB | ~15-30 мин | 8GB+ |

## 🎯 Результат обработки

После успешного выполнения в папке `output/` появится:
- `reviews_processed.parquet/` - обработанные данные в формате Parquet
- Новые колонки: `helpful_ratio`, `review_timestamp`, `days_since_review`, `*_clean`, `*_len`
- Агрегированные метрики по `asin` и `reviewerID`

## 🔧 Локальная установка (Windows/Mac/Linux)

Требования: Docker Desktop

```bash
git clone https://github.com/zvarrog/de-pipeline-project.git
cd de-pipeline-project

# Сборка и запуск
docker build -t kindle-reviews-processor:latest .
docker compose up -d

# Откройте http://localhost:8080
```

## 🏗️ Архитектура проекта

```
├── dags/                       # Airflow DAGs
│   └── process_kindle_reviews_dag.py
├── data/
│   ├── original/              # Локальные полные данные (если есть)
│   └── sample/                # Встроенный семпл (1K записей)
├── output/                    # Результаты обработки (Parquet)
├── scripts/                   # Вспомогательные скрипты
├── process_data.py            # Основная логика обработки (PySpark)
├── docker-compose.yaml        # Airflow стек
├── Dockerfile                 # Образ для обработки данных
└── requirements.txt           # Python зависимости
```

## 📈 Что делает pipeline

1. **Загрузка данных**: Автоматическая или ручная загрузка датасета
2. **Очистка**: Обработка текста, удаление HTML, нормализация
3. **Обогащение**: Расчёт метрик helpful_ratio, временных меток
4. **Агрегация**: Статистики по товарам и пользователям
5. **Сохранение**: Экспорт в оптимизированный формат Parquet

## 🛠️ Технологический стек

- **Orchestration**: Apache Airflow
- **Processing**: Apache Spark (PySpark)
- **Containerization**: Docker & Docker Compose
- **Data Format**: Parquet (сжатие + схема)
- **Infrastructure**: GitHub Codespaces ready

## 📝 Примеры кода

### Обработка текста
```python
# Очистка HTML и спецсимволов
df = df.withColumn(
    "reviewText_clean",
    regexp_replace(lower(col("reviewText")), r"<[^>]+>|[^\w\s]", " ")
)
```

### Агрегация метрик
```python
# Средний рейтинг по товарам
asin_agg = df.groupBy("asin").agg(
    avg("overall").alias("asin_overall_avg"),
    count("*").alias("asin_total_votes_count")
)
```

## 🤝 Вклад в проект

1. Fork репозиторий
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📄 Лицензия

Этот проект распространяется под MIT лицензией. Подробности в файле [LICENSE](LICENSE).

---

⭐ **Если проект понравился, поставьте звёздочку!**
