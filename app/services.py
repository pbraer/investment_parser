import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from tqdm import tqdm

from app.config import Settings
from app.excel_repository import (
    load_projects_from_excel,
    save_projects_to_excel,
    update_project_row,
)
from app.logger import setup_logger
from app.nlp import (
    DescriptionGenerator,
    InterestPredictor,
    PriceExtractionHint,
    T5DescriptionGenerator,
    T5PriceAssistant,
    TransformerStageClassifier,
    stage_to_russian,
)
from app.schemas import AnalysisResult, Article, ProjectRecord
from app.scraper import ArticleExtractor, deduplicate_articles
from app.search_engine import SearchClient


logger = setup_logger(__name__)


INVESTMENT_KEYWORDS = [
    "инвестиции",
    "объем инвестиций",
    "объём инвестиций",
    "стоимость проекта",
    "проект оценивается",
    "капвложения",
    "капитальные вложения",
    "вложат",
    "вложит",
    "инвестор направит",
    "инвестор вложит",
    "финансирование проекта",
    "объем вложений",
    "стоимость строительства",
    "строительство оценивается",
    "реализация проекта потребует",
]

BAD_PRICE_KEYWORDS = [
    "цена номера",
    "стоимость номера",
    "номер от",
    "за ночь",
    "за сутки",
    "сутки",
    "ночь",
    "проживание",
    "стоимость проживания",
    "тариф",
    "скидка",
    "аренда",
    "цена билета",
    "стоимость билета",
    "меню",
    "доставка",
    "руб/сутки",
    "рублей в сутки",
    "койко-место",
    "глэмпинг",
    "booking",
    "tripadvisor",
    "отзывы",
]


@dataclass
class PriceCandidate:
    raw_text: str
    normalized_mln_rub: float
    context: str
    score: float
    article_date: Optional[datetime]
    article_url: str


def safe_filename(value: str) -> str:
    value = re.sub(r"[^\w\-_\. ]+", "_", value.strip(), flags=re.U)
    value = re.sub(r"\s+", "_", value)
    return value[:120]


def parse_money_to_mln_rub(number_text: str, unit_text: str) -> float:
    number = float(number_text.replace(" ", "").replace(",", "."))
    unit = unit_text.lower()

    if "трлн" in unit:
        return number * 1_000_000
    if "млрд" in unit:
        return number * 1_000
    if "млн" in unit:
        return number
    if "тыс" in unit:
        return number / 1_000

    return number


def format_price_from_mln_rub(value_mln_rub: float) -> str:
    if value_mln_rub >= 1_000_000:
        value = value_mln_rub / 1_000_000
        return f"{value:.2f} трлн руб".replace(".", ",")
    if value_mln_rub >= 1_000:
        value = value_mln_rub / 1_000
        return f"{value:.2f} млрд руб".replace(".", ",")
    return f"{value_mln_rub:.2f} млн руб".replace(".", ",")


def _context_has_project_price_signal(context_l: str) -> bool:
    """
    Цена должна относиться именно к проекту или инвестициям
    """
    if any(bad_kw in context_l for bad_kw in BAD_PRICE_KEYWORDS):
        return False

    if any(good_kw in context_l for good_kw in INVESTMENT_KEYWORDS):
        return True

    strong_signals = 0
    if "проект" in context_l:
        strong_signals += 1
    if "строительство" in context_l:
        strong_signals += 1
    if "объект" in context_l or "объекта" in context_l:
        strong_signals += 1
    if "инвест" in context_l:
        strong_signals += 1

    return strong_signals >= 2


def find_price_candidates_in_text(
    text: str,
    article_date: Optional[datetime],
    article_url: str,
) -> list[PriceCandidate]:
    """
    Ищет суммы, похожие именно на инвестиционную стоимость проекта
    """
    candidates: list[PriceCandidate] = []
    if not text:
        return candidates

    clean_text = re.sub(r"\s+", " ", text)

    pattern = re.compile(
        r"(\d{1,3}(?:[\s\u00A0]?\d{3})*(?:[.,]\d+)?)\s*(трлн|млрд|млн|тыс)\s*руб",
        flags=re.I,
    )

    for match in pattern.finditer(clean_text):
        raw_number = match.group(1)
        raw_unit = match.group(2)

        start = max(0, match.start() - 140)
        end = min(len(clean_text), match.end() + 140)
        context = clean_text[start:end]
        context_l = context.lower()

        if not _context_has_project_price_signal(context_l):
            continue

        try:
            mln_rub = parse_money_to_mln_rub(raw_number, raw_unit)
        except Exception:
            continue

        if mln_rub < 10:
            continue

        score = 0.0
        if any(good_kw in context_l for good_kw in INVESTMENT_KEYWORDS):
            score += 3.0
        if "проект" in context_l:
            score += 1.0
        if "строительство" in context_l:
            score += 1.0
        if "объект" in context_l or "объекта" in context_l:
            score += 0.5
        if "инвест" in context_l:
            score += 1.0

        candidates.append(
            PriceCandidate(
                raw_text=f"{raw_number} {raw_unit} руб",
                normalized_mln_rub=mln_rub,
                context=context,
                score=score,
                article_date=article_date,
                article_url=article_url,
            )
        )

    return candidates


def build_price_candidate_from_hint(
    hint: PriceExtractionHint | None,
    article: Article,
) -> PriceCandidate | None:
    """
    Преобразует подсказку от T5 в обычный PriceCandidate
    """
    if hint is None or not hint.raw_text:
        return None

    evidence_text = hint.evidence or f"{article.title}. {article.text[:700]}"

    candidates = find_price_candidates_in_text(
        text=evidence_text,
        article_date=article.published_at,
        article_url=article.url,
    )

    if candidates:
        normalized_hint = re.sub(r"\s+", "", hint.raw_text.lower())
        matched = [
            c for c in candidates
            if normalized_hint in re.sub(r"\s+", "", c.raw_text.lower())
            or re.sub(r"\s+", "", c.raw_text.lower()) in normalized_hint
        ]
        candidate = max(matched or candidates, key=lambda c: c.score)
        candidate.score += hint.confidence * 2.0
        return candidate

    raw_match = re.search(
        r"(\d{1,3}(?:[\s\u00A0]?\d{3})*(?:[.,]\d+)?)\s*(трлн|млрд|млн|тыс)\s*руб",
        hint.raw_text,
        flags=re.I,
    )
    if not raw_match:
        return None

    try:
        mln_rub = parse_money_to_mln_rub(raw_match.group(1), raw_match.group(2))
    except Exception:
        return None

    context = evidence_text[:400]
    if not _context_has_project_price_signal(context.lower()):
        return None

    return PriceCandidate(
        raw_text=f"{raw_match.group(1)} {raw_match.group(2)} руб",
        normalized_mln_rub=mln_rub,
        context=context,
        score=2.0 + hint.confidence * 2.0,
        article_date=article.published_at,
        article_url=article.url,
    )


def choose_best_price_candidate(
    project: ProjectRecord,
    articles: list[Article],
    extra_candidates: list[PriceCandidate] | None = None,
) -> PriceCandidate | None:
    """
    Выбирает лучшую найденную цену проекта
    """
    all_candidates: list[PriceCandidate] = []

    for article in articles:
        combined_text = f"{article.title}. {article.text}"
        all_candidates.extend(
            find_price_candidates_in_text(
                text=combined_text,
                article_date=article.published_at,
                article_url=article.url,
            )
        )

    if extra_candidates:
        all_candidates.extend(extra_candidates)

    if not all_candidates:
        return None

    current_price = project.price_mln_rub

    def candidate_rank(c: PriceCandidate):
        recency_dt = c.article_date or datetime(1900, 1, 1)

        closeness_bonus = 0.0
        if current_price > 0:
            ratio = c.normalized_mln_rub / current_price
            if 0.7 <= ratio <= 1.5:
                closeness_bonus = 2.0
            elif 0.5 <= ratio <= 2.0:
                closeness_bonus = 0.5

        # Бонус если имя проекта встречается рядом с ценой
        name_tokens = [
            t.lower() for t in re.findall(r"\w+", project.name)
            if len(t) >= 4
        ]
        context_l = c.context.lower()
        name_hits = sum(1 for t in name_tokens if t in context_l)
        name_bonus = min(name_hits * 0.8, 2.0)

        return (
            c.score + closeness_bonus + name_bonus,
            recency_dt,
            c.normalized_mln_rub,
        )

    all_candidates.sort(key=candidate_rank, reverse=True)
    best = all_candidates[0]

    if best.score < 3.5:
        return None

    return best


class ArticleCollectionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.search_client = SearchClient(settings)
        self.extractor = ArticleExtractor(settings)

    def collect_articles(self, project: ProjectRecord) -> List[Article]:
        search_results = self.search_client.search_project(project)
        logger.info(f"[{project.project_id}] search_results={len(search_results)}")

        articles = []
        for result in search_results:
            article = self.extractor.extract_article(result, project)
            if article:
                articles.append(article)

        logger.info(f"[{project.project_id}] extracted_articles_before_dedup={len(articles)}")
        articles = deduplicate_articles(articles)
        logger.info(f"[{project.project_id}] extracted_articles_after_dedup={len(articles)}")

        return articles[: self.settings.max_articles_per_project]

    def save_articles_cache(self, project: ProjectRecord, articles: List[Article]) -> None:
        cache_file = self.settings.cache_dir / "articles_cache.jsonl"
        with open(cache_file, "a", encoding="utf-8") as f:
            for article in articles:
                row = {
                    "project_id": project.project_id,
                    "project_name": project.name,
                    "title": article.title,
                    "text": article.text,
                    "published_at": article.published_at.isoformat() if article.published_at else None,
                    "url": article.url,
                    "domain": article.domain,
                    "query": article.query,
                    "reliability_score": article.reliability_score,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save_project_archive(self, project: ProjectRecord, articles: List[Article]) -> None:
        filename = f"{safe_filename(project.project_id)}__{safe_filename(project.name)}.json"
        path = self.settings.articles_archive_dir / filename

        payload = {
            "project_id": project.project_id,
            "project_name": project.name,
            "place": project.place,
            "industry": project.industry,
            "saved_at": datetime.now().isoformat(),
            "articles": [
                {
                    "title": a.title,
                    "url": a.url,
                    "published_at": a.published_at.isoformat() if a.published_at else None,
                    "domain": a.domain,
                    "query": a.query,
                    "reliability_score": a.reliability_score,
                    "text": a.text,
                }
                for a in articles
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


class AnalysisService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.stage_model = TransformerStageClassifier(settings)
        self.interest_predictor = InterestPredictor(settings)

        self.extractive_description_generator = DescriptionGenerator()
        self.t5_description_generator = T5DescriptionGenerator(settings)
        self.t5_price_assistant = T5PriceAssistant(settings)

    def assert_models_ready(self) -> None:
        self.stage_model.assert_trained()
        self.interest_predictor.assert_trained()

    def build_project_corpus(self, project: ProjectRecord, articles: List[Article]) -> str:
        parts = [
            f"Проект: {project.name}",
            f"Регион: {project.place}",
            f"Отрасль: {project.industry}",
            f"Исходная стадия: {project.stage_raw}",
            f"Описание проекта из Excel: {project.description_raw}",
        ]
        for article in articles:
            parts.append(f"Заголовок: {article.title}")
            parts.append(f"Текст статьи: {article.text[:2500]}")

        corpus = " ".join([p for p in parts if p]).strip()
        return corpus[:20000]

    def choose_last_news_date(self, articles: List[Article]) -> datetime | None:
        dates = [a.published_at for a in articles if a.published_at]
        if not dates:
            return None
        return max(dates)

    def analyze_project(self, project: ProjectRecord, articles: List[Article]) -> AnalysisResult:
        self.assert_models_ready()

        if not articles:
            raise ValueError("analyze_project вызван без статей")

        corpus = self.build_project_corpus(project, articles)
        last_news_date = self.choose_last_news_date(articles)

        actual_stage_en = self.stage_model.predict_one(corpus)
        actual_stage_ru = stage_to_russian(actual_stage_en)

        # Порядок стадий, так как стадия не может идти назад
        STAGE_ORDER = {
            "Инициация": 0,
            "Проектирование": 1,
            "Согласование / экспертиза": 2,
            "Финансирование / поиск инвестора": 3,
            "Строительство": 4,
            "Строительство (внутренние и инженерные работы)": 4,
            "Запуск / ввод в эксплуатацию": 5,
            "Введен в эксплуатацию": 6,
            "Заморожен / приостановлен": 7,
            "Отменен / закрыт": 8,
        }

        current_order = STAGE_ORDER.get(project.stage_raw, -1)
        new_order = STAGE_ORDER.get(actual_stage_ru, -1)

        # Если новая стадия раньше текущей — оставляем текущую
        if new_order < current_order:
            logger.info(
                f"[{project.project_id}] Регресс стадии заблокирован: "
                f"{project.stage_raw} → {actual_stage_ru}, оставляем {project.stage_raw}"
            )
            actual_stage_ru = project.stage_raw

        article_texts = [a.text for a in articles]

        generated_description = self.t5_description_generator.generate(project.name, article_texts)
        if not generated_description:
            generated_description = self.extractive_description_generator.generate(project.name, article_texts)
        if not generated_description:
            generated_description = project.description_raw[:900] if project.description_raw else ""

        interest_score, interest_label = self.interest_predictor.predict(corpus)

        assistant_candidates: list[PriceCandidate] = []
        for article in articles[:5]:
            hint = self.t5_price_assistant.extract(
                project_name=project.name,
                article_title=article.title,
                article_text=article.text,
            )
            candidate = build_price_candidate_from_hint(hint, article)
            if candidate is not None:
                assistant_candidates.append(candidate)

        best_price = choose_best_price_candidate(
            project=project,
            articles=articles,
            extra_candidates=assistant_candidates,
        )

        actual_price_raw = None
        actual_price_mln_rub = None
        price_source_excerpt = None
        price_updated = False

        if best_price is not None:
            actual_price_mln_rub = best_price.normalized_mln_rub
            actual_price_raw = format_price_from_mln_rub(best_price.normalized_mln_rub)
            price_source_excerpt = best_price.context[:400]
            price_updated = True

        top_articles = articles[:5]
        source_urls = [a.url for a in top_articles]
        evidence_titles = [a.title for a in top_articles]
        evidence_dates = [
            a.published_at.strftime("%d.%m.%Y") if a.published_at else ""
            for a in top_articles
        ]

        logger.info(
            f"[{project.project_id}] last_news_date="
            f"{last_news_date.strftime('%d.%m.%Y') if last_news_date else 'None'} | "
            f"pred_stage={actual_stage_ru} | "
            f"interest={interest_label}:{interest_score} | "
            f"price_found={actual_price_raw}"
        )

        return AnalysisResult(
            project_name=project.name,
            actual_stage=actual_stage_ru,
            last_news_date=last_news_date,
            generated_description=generated_description,
            interest_score=interest_score,
            interest_label=interest_label,
            source_urls=source_urls,
            evidence_titles=evidence_titles,
            evidence_dates=evidence_dates,
            news_count=len(articles),
            actual_price_raw=actual_price_raw,
            actual_price_mln_rub=actual_price_mln_rub,
            price_source_excerpt=price_source_excerpt,
            price_updated=price_updated,
        )


class ProjectUpdateService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.collector = ArticleCollectionService(settings)
        self.analysis = AnalysisService(settings)

    @staticmethod
    def _should_overwrite_core_fields(project: ProjectRecord, result: AnalysisResult) -> tuple[bool, str]:
        if result.news_count == 0:
            return False, "no_articles_found"

        if result.last_news_date is None:
            return False, "articles_found_but_no_date"

        if project.update_date is None:
            return True, "current_update_missing"

        if result.last_news_date >= project.update_date:
            return True, "found_same_or_newer_news"

        return False, "found_only_older_news"

    def update_excel(self, input_path: str, output_path: str) -> None:
        self.analysis.assert_models_ready()

        df, projects, mapping = load_projects_from_excel(input_path)
        logger.info(f"Загружено проектов: {len(projects)}")
        logger.info(f"Определено сопоставление колонок: {mapping}")

        for project in tqdm(projects, desc="Обновление проектов"):
            try:
                logger.info(f"Обработка: {project.project_id} | {project.name}")

                articles = self.collector.collect_articles(project)
                self.collector.save_articles_cache(project, articles)
                self.collector.save_project_archive(project, articles)

                if not articles:
                    logger.warning(f"[{project.project_id}] Не найдено ни одной релевантной статьи")
                    continue

                result = self.analysis.analyze_project(project, articles)
                overwrite_core_fields, reason = self._should_overwrite_core_fields(project, result)

                logger.info(
                    f"[{project.project_id}] overwrite_core_fields={overwrite_core_fields} | reason={reason}"
                )

                df = update_project_row(
                    df=df,
                    row_index=project.row_index,
                    result=result,
                    mapping=mapping,
                    overwrite_core_fields=overwrite_core_fields,
                    update_reason=reason,
                )

            except Exception as exc:
                logger.exception(f"Ошибка при обработке проекта {project.name}: {exc}")

        save_projects_to_excel(df, output_path)
        logger.info(f"Обновленный Excel сохранен: {output_path}")