from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ProjectRecord:
    row_index: int
    project_id: str
    name: str
    start_date: Optional[datetime]
    update_date: Optional[datetime]
    price_raw: str
    price_mln_rub: float
    stage_raw: str
    industry: str
    place: str
    project_type: str
    description_raw: str


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    domain: str
    score: float
    query: str


@dataclass
class Article:
    project_name: str
    title: str
    text: str
    published_at: Optional[datetime]
    url: str
    domain: str
    query: str
    reliability_score: float = 0.0


@dataclass
class AnalysisResult:
    project_name: str
    actual_stage: str
    last_news_date: Optional[datetime]
    generated_description: str
    interest_score: float
    interest_label: str

    source_urls: list[str] = field(default_factory=list)
    evidence_titles: list[str] = field(default_factory=list)
    evidence_dates: list[str] = field(default_factory=list)
    news_count: int = 0

    # Поля для цены
    actual_price_raw: str | None = None
    actual_price_mln_rub: float | None = None
    price_source_excerpt: str | None = None
    price_updated: bool = False