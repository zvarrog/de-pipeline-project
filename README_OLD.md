# DE Pipeline Project

🚀 **Современный data engineering проект** с Airflow, Spark, Docker для изучения полного MLOps стека.

## � Режимы обработки данных

Проект поддерживает **3 режима** работы с данными:

### 1. 🔸 **Sample** (по умолчанию)
- **Источник**: Встроенный в Docker образ семпл (1000 записей)
- **Применение**: Быстрое тестирование, разработка
- **Время выполнения**: ~30 секунд

### 2. 🔸 **Full (локальный файл)** 
- **Источник**: Полный датасет (5M+ записей) в `data/original/kindle_reviews.csv`
- **Применение**: Продакшн обработка больших данных
- **Требует**: Ручную загрузку файла (см. инструкции ниже)

### 3. 🔸 **Kaggle (автоматическая загрузка)**
- **Источник**: Автоматическое скачивание с Kaggle
- **Применение**: Первоначальная загрузка полного датасета
- **Требует**: Настройку API Kaggle (см. инструкции ниже)

## �🚀 Быстрый старт

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

## 📂 Настройка источников данных

### Для режима Sample (готов из коробки)
Ничего дополнительно не требуется - семпл данных встроен в Docker образ.

### Для режима Full (локальный файл)

**Для настоящего полного датасета (~5.2M записей, 2.9GB):**

#### Вариант 1: Автоматическая загрузка (рекомендуется)
```bash
# Установите kaggle CLI
pip install kaggle

# Настройте учётные данные Kaggle
export KAGGLE_USERNAME="ваш_username"
export KAGGLE_KEY="ваш_api_key"

# Запустите скрипт загрузки
python scripts/download_full_dataset.py
```

#### Вариант 2: Ручная загрузка
1. Перейдите на https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews
2. Нажмите **Download** → скачается `archive.zip`
3. Распакуйте `kindle_reviews.csv` из архива
4. Поместите файл в `data/original/kindle_reviews.csv`

**После загрузки:**
```bash
# Запустите DAG с параметром full
docker compose exec airflow-apiserver airflow dags trigger end_to_end_kindle_pipeline --conf '{"data_mode": "full"}'
```

**⚠️ Требования**: Убедитесь, что у Docker достаточно памяти (рекомендуется 8GB+ RAM)

### Для режима Kaggle (автоматическая загрузка)

**⚠️ Экспериментальный режим**: Требует API токен Kaggle.

1. Получите API токен Kaggle:
   - Зайдите в Kaggle Account → API → Create New Token
   - Скачайте `kaggle.json`

2. Обновите DAG с вашими учётными данными:
   ```python
   # В dags/process_kindle_reviews_dag.py
   params={
       "kaggle_username": "ваш_username",
       "kaggle_key": "ваш_api_key"
   }
   ```

3. Запустите DAG с параметром:
   - JSON: `{"data_mode": "kaggle"}`

**Рекомендация**: Для продакшн используйте режим **Full** с предварительно загруженным файлом.

## 📊 Производительность и ресурсы

| Режим | Записей | Размер файла | Время обработки | Требования RAM |
|-------|---------|--------------|-----------------|----------------|
| Sample | 1,000 | 701KB | ~30 сек | 2GB |
| Full | 5.2M | 2.9GB | ~15-30 мин | 8GB+ |
| Kaggle | 5.2M | 2.9GB + скачивание | ~20-40 мин | 8GB+ |

## 🎯 Примеры использования

### Быстрое тестирование (режим Sample)
```bash
# В Airflow UI просто запустите DAG без параметров
# Или через CLI:
docker compose exec airflow-apiserver airflow dags trigger end_to_end_kindle_pipeline
```

### Обработка полного датасета (режим Full)
```bash
# 1. Скачайте и поместите kindle_reviews.csv в data/original/
# 2. Запустите с параметром:
docker compose exec airflow-apiserver airflow dags trigger end_to_end_kindle_pipeline --conf '{"data_mode": "full"}'
```

### Результат обработки
После успешного выполнения в папке `output/` появится:
- `reviews_processed.parquet/` - обработанные данные в формате Parquet
- Новые колонки: `helpful_ratio`, `review_timestamp`, `days_since_review`, `*_clean`, `*_len`, агрегаты по `asin` и `reviewerID`

### Локально (Windows/Mac/Linux)

Требования: Docker Desktop

```bash
git clone <ваш-репозиторий>
cd de-pipeline-project
docker build -t kindle-reviews-processor:latest .
docker compose up -d
```

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
