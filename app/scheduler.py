from apscheduler.schedulers.blocking import BlockingScheduler

from app.config import Settings
from app.logger import setup_logger
from app.services import ProjectUpdateService


logger = setup_logger(__name__)


def run_scheduler(input_path: str, output_path: str, every_hours: int = 12) -> None:
    """
    Периодический запуск обновления Excel.
    """
    settings = Settings()
    settings.ensure_dirs()

    service = ProjectUpdateService(settings)
    scheduler = BlockingScheduler()

    scheduler.add_job(
        service.update_excel,
        trigger="interval",
        hours=every_hours,
        args=[input_path, output_path],
        id="project_update_job",
        replace_existing=True,
    )

    logger.info(f"Планировщик запущен. Обновление каждые {every_hours} часов.")
    service.update_excel(input_path, output_path)
    scheduler.start()