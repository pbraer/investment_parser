from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """
    Главный класс конфигурации, где хранятся все настройки проекта.
    """

    # Корневые директории проекта
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "cache")
    models_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "models")
    articles_archive_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "articles_archive")

    # Директории отдельных моделей
    stage_model_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "models" / "stage_model")
    interest_model_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "models" / "interest_model")

    # Директории генеративных моделей
    description_model_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "models" / "description_model")
    price_assistant_model_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "models" / "price_assistant_model")

    # Параметры поиска: сколько запросов и статей собирать на один проект
    search_engine_url: str = "https://html.duckduckgo.com/html/"
    max_queries_per_project: int = 7
    max_results_per_query: int = 8
    max_articles_per_project: int = 10

    # Настройки HTTP-соединения
    request_timeout: int = 25
    request_retries: int = 3
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )

    # Резервная загрузка через браузер — включается, если обычный запрос не сработал
    enable_selenium_fallback: bool = True
    selenium_headless: bool = True
    selenium_page_load_timeout: int = 35

    # Базовая трансформерная модель для классификации
    transformer_model_name: str = "xlm-roberta-base"
    generator_model_name: str | None = None
    max_text_length: int = 512

    # Флаги включения генеративных моделей
    enable_t5_description: bool = True
    enable_t5_price_assistant: bool = True

    description_model_name: str | None = str(
        Path(__file__).resolve().parent.parent / "data" / "models" / "description_model"
    )

    price_assistant_model_name: str | None = str(
        Path(__file__).resolve().parent.parent / "data" / "models" / "price_assistant_model"
    )

    # Параметры генерации текста: длина входа/выхода и количество лучей поиска
    seq2seq_max_source_length: int = 768
    seq2seq_max_target_length: int = 128
    seq2seq_num_beams: int = 4

    # Ключевые слова для определения надёжности источника
    trusted_domains_keywords: tuple = (
        "gov", "government", "official", "admin", "minprom", "economy",
        "tass", "interfax", "kommersant", "vedomosti", "rbc",
        "ria", "rg.ru", "roscongress", "invest", "mining", "industry",
        "metalinfo", "dp", "expert", "forbes", "akm"
    )

    # Список колонок, которые система записывает в Excel после обработки проекта
    output_columns: tuple = (
        "actual_stage_ru",
        "last_news_date",
        "generated_description",
        "interest_score",
        "interest_label",
        "source_urls",
        "evidence_titles",
        "evidence_dates",
        "news_count",
        "updated_at",
        "core_fields_updated",
        "update_reason",
        "actual_price_found",
        "actual_price_mln_rub",
        "price_source_excerpt",
        "price_updated",
    )

    def ensure_dirs(self) -> None:
        """Создаёт все нужные директории при запуске, если они ещё не существуют."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.stage_model_dir.mkdir(parents=True, exist_ok=True)
        self.interest_model_dir.mkdir(parents=True, exist_ok=True)
        self.description_model_dir.mkdir(parents=True, exist_ok=True)
        self.price_assistant_model_dir.mkdir(parents=True, exist_ok=True)
        self.articles_archive_dir.mkdir(parents=True, exist_ok=True)