from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator

# Импортируем специальный класс Mount для описания томов
from docker.types import Mount

# ------------------------------------------------------------------------------------
# ВАЖНО: Замените на ваш АБСОЛЮТНЫЙ путь к корневой папке проекта!
# Пример для Windows: 'C:/Users/dasiqe/de-pipeline-project'
HOST_PROJECT_PATH = "C:/Users/dasiqe/de-pipeline-project"
# ------------------------------------------------------------------------------------

# Создаем объекты Mount для каждого пробрасываемого тома
# Это более явный и современный способ
data_mount = Mount(
    source=f"{HOST_PROJECT_PATH}/data",
    target="/app/data",
    type="bind",
    read_only=True,  # Хорошая практика для входных данных
)

output_mount = Mount(
    source=f"{HOST_PROJECT_PATH}/output", target="/app/output", type="bind"
)


with DAG(
    dag_id="end_to_end_kindle_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["portfolio", "spark", "pytorch"],
) as dag:

    # ЗАДАЧА 1: Обработка данных с помощью PySpark в Docker
    spark_processing_task = DockerOperator(
        task_id="spark_data_processing",
        image="kindle-reviews-processor:latest",
        auto_remove="success",
        # ИСПОЛЬЗУЕМ ПАРАМЕТР `mounts` ВМЕСТО `volumes`
        mounts=[data_mount, output_mount],
        # Указываем, как Airflow должен найти Docker. Это стандартная настройка при запуске из Docker Compose.
        docker_url="unix://var/run/docker.sock",
        # Это может помочь избежать проблем с сетью между контейнерами
        network_mode="bridge",
    )
