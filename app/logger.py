from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """
    Конфигурация проекта
    """

    # Корневая директория проекта и пути к рабочим поддиректориям
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "cache")
    models_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "models")

    # Директории для сохранения обученных моделей классификации
    stage_model_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "models" / "stage_model")
    interest_model_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "models" / "interest_model")

    # Параметры поискового движка: URL и лимиты на количество запросов и результатов
    search_engine_url: str = "https://html.duckduckgo.com/html/"
    max_queries_per_project: int = 6
    max_results_per_query: int = 6
    max_articles_per_project: int = 8

    # Параметры HTTP-запросов
    request_timeout: int = 20
    request_retries: int = 3
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )

    # Резервный парсинг через Selenium — отключён по умолчанию
    enable_selenium_fallback: bool = False
    selenium_headless: bool = True

    # Классификатор на базе XLM-RoBERTa
    transformer_model_name: str = "xlm-roberta-base"
    generator_model_name: str | None = None
    max_text_length: int = 512

    # Ключевые слова для повышения доверия к источнику при ранжировании статей
    trusted_domains_keywords: tuple = (
        "gov", "government", "official", "admin", "minprom", "economy",
        "tass", "interfax", "kommersant", "vedomosti", "rbc",
        "ria", "rg.ru", "roscongress", "invest", "mining", "industry"
    )

    # Колонки, добавляемые или обновляемые в Excel-файле по результатам анализа
    output_columns: tuple = (
        "actual_stage",       # Текущая стадия проекта по классификатору
        "last_news_date",     # Дата последней найденной новости
        "generated_description",  # Сгенерированное резюме по статьям
        "interest_score",     # Числовая оценка инвестиционного интереса
        "interest_label",     # Метка интереса
        "source_urls",        # Ссылки на источники, использованные при анализе
        "evidence_titles",    # Заголовки статей-доказательств
        "updated_at",         # Дата и время последнего обновления записи
    )

    def ensure_dirs(self) -> None:
        """
        Создаёт все служебные директории проекта, если они ещё не существуют.
        Вызывается при инициализации приложения до начала работы компонентов.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.stage_model_dir.mkdir(parents=True, exist_ok=True)
        self.interest_model_dir.mkdir(parents=True, exist_ok=True)


import logging
import sys


def setup_logger(name: str = "investment_parser") -> logging.Logger:
    """
    Создаёт и настраивает единый логгер для проекта.
    Повторный вызов с тем же именем возвращает уже существующий экземпляр
    без дублирования обработчиков.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"  # Формат: дата и время | уровень | имя | сообщение
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger