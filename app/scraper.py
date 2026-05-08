import json
import re
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

from app.config import Settings
from app.schemas import Article, ProjectRecord, SearchResult


# Словарь для преобразования русских названий месяцев в числовой формат
RU_MONTHS = {
    "января": "01",
    "февраля": "02",
    "марта": "03",
    "апреля": "04",
    "мая": "05",
    "июня": "06",
    "июля": "07",
    "августа": "08",
    "сентября": "09",
    "октября": "10",
    "ноября": "11",
    "декабря": "12",
}


def normalize_whitespace(text: str) -> str:
    """Заменяет любые последовательности пробельных символов на одиночный пробел"""
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_datetime(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Приводит дату и время к UTC.
    """
    if dt is None:
        return None

    if dt.tzinfo is not None and dt.utcoffset() is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)

    return dt


def is_reasonable_article_date(dt: Optional[datetime]) -> bool:
    """
    Фильтр нереалистичных дат публикации.
    Отклоняет даты ранее 2000 года и даты в будущем.
    """
    if dt is None:
        return False

    dt = normalize_datetime(dt)
    today = datetime.now()

    # Отсекаем слишком старые и будущие даты
    if dt.year < 2000:
        return False
    if dt > today:
        return False

    return True


def sanitize_article_date(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Нормализует дату и отбрасывает нереалистичные значения.
    Возвращает None, если дата не прошла проверку is_reasonable_article_date.
    """
    dt = normalize_datetime(dt)
    if not is_reasonable_article_date(dt):
        return None
    return dt


def extract_meta_content(soup: BeautifulSoup, attrs_list: list[dict]) -> str | None:
    """
    Ищет первый подходящий <meta>-тег из списка атрибутов и возвращает его content.
    Перебирает кандидатов по приоритету от наиболее специфичных к общим.
    """
    for attrs in attrs_list:
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return str(tag.get("content")).strip()
    return None


def parse_date_safe(value: str | None) -> Optional[datetime]:
    """
    Парсит строку даты.
    Возвращает None при любой ошибке или нереалистичной дате.
    """
    if not value:
        return None

    value = str(value).strip()
    if not value:
        return None

    try:
        parsed = date_parser.parse(value, dayfirst=True, fuzzy=True)
        return sanitize_article_date(parsed)
    except Exception:
        pass

    # Резервный парсинг русскоязычных дат через regex + словарь месяцев
    m = re.search(
        r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})",
        value.lower()
    )
    if m:
        day, month_ru, year = m.groups()
        iso = f"{year}-{RU_MONTHS[month_ru]}-{int(day):02d}"
        try:
            return sanitize_article_date(datetime.strptime(iso, "%Y-%m-%d"))
        except Exception:
            pass

    return None


def parse_json_ld_dates(soup: BeautifulSoup) -> Optional[datetime]:
    """
    Извлекает дату публикации из блоков JSON-LD (schema.org).
    Проверяет поля datePublished, dateModified и uploadDate.
    Возвращает первую валидную дату или None.
    """
    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        text = script.string or script.get_text(" ", strip=True)
        if not text:
            continue
        try:
            data = json.loads(text)
            candidates = data if isinstance(data, list) else [data]
            for item in candidates:
                if isinstance(item, dict):
                    for key in ("datePublished", "dateModified", "uploadDate"):
                        if key in item:
                            dt = parse_date_safe(item[key])
                            if dt:
                                return dt
        except Exception:
            continue
    return None


def parse_date_from_url(url: str) -> Optional[datetime]:
    """
    Пытается извлечь дату публикации из структуры URL.
    Поддерживает форматы /2024/03/15/, /2024-03-15/ и _2024_03_15.
    """
    patterns = [
        r"/(20\d{2})/(0[1-9]|1[0-2])/([0-3]\d)/",
        r"/(20\d{2})-(0[1-9]|1[0-2])-([0-3]\d)/",
        r"_(20\d{2})_(0[1-9]|1[0-2])_([0-3]\d)",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            year, month, day = m.groups()
            try:
                return sanitize_article_date(datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d"))
            except Exception:
                pass
    return None


def parse_article_date(
    soup: BeautifulSoup,
    html_text: str = "",
    fallback_texts: list[str] | None = None,
) -> Optional[datetime]:
    """
    Извлекает дату публикации статьи с каскадом стратегий по убыванию надёжности.
    Возвращает первую найденную валидную дату или None.
    """
    meta_candidates = [
        {"property": "article:published_time"},
        {"property": "article:modified_time"},
        {"name": "pubdate"},
        {"name": "publishdate"},
        {"name": "publication_date"},
        {"name": "date"},
        {"name": "DC.date.issued"},
        {"name": "parsely-pub-date"},
        {"itemprop": "datePublished"},
        {"itemprop": "dateModified"},
    ]
    content = extract_meta_content(soup, meta_candidates)
    if content:
        dt = parse_date_safe(content)
        if dt:
            return dt

    time_tag = soup.find("time")
    if time_tag:
        dt = parse_date_safe(time_tag.get("datetime") or time_tag.get_text(" ", strip=True))
        if dt:
            return dt

    dt = parse_json_ld_dates(soup)
    if dt:
        return dt

    plain = normalize_whitespace(html_text)
    patterns = [
        r"\b\d{1,2}\.\d{1,2}\.\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4}\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, plain, flags=re.I)
        if m:
            dt = parse_date_safe(m.group(0))
            if dt:
                return dt

    # Поиск даты в snippet-е и заголовке
    for extra in fallback_texts or []:
        for pattern in patterns:
            m = re.search(pattern, str(extra), flags=re.I)
            if m:
                dt = parse_date_safe(m.group(0))
                if dt:
                    return dt

    return None


def parse_article_title(soup: BeautifulSoup) -> str:
    """
    Извлекает заголовок статьи
    """
    og_title = extract_meta_content(soup, [{"property": "og:title"}])
    if og_title:
        return normalize_whitespace(og_title)

    h1 = soup.find("h1")
    if h1:
        return normalize_whitespace(h1.get_text(" ", strip=True))

    if soup.title:
        return normalize_whitespace(soup.title.get_text(" ", strip=True))

    return ""


def parse_article_text(soup: BeautifulSoup) -> str:
    """
    Извлекает основной текст статьи из HTML.
    Из всех найденных кандидатов возвращается наиболее длинный текст.
    """
    candidates = []

    article_tag = soup.find("article")
    if article_tag:
        paragraphs = article_tag.find_all("p")
        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        text = normalize_whitespace(text)
        if len(text) > 180:
            candidates.append(text)

    selector_candidates = [
        {"name": "div", "attrs": {"class": re.compile(r"(content|article|news|body|text|story)", re.I)}},
        {"name": "section", "attrs": {"class": re.compile(r"(content|article|news|body|text|story)", re.I)}},
        {"name": "main", "attrs": {}},
    ]

    for item in selector_candidates:
        blocks = soup.find_all(item["name"], attrs=item["attrs"])
        for block in blocks:
            paragraphs = block.find_all("p")
            if len(paragraphs) < 2:  # Блоки с одним абзацем не являются статьей
                continue
            text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
            text = normalize_whitespace(text)
            if len(text) > 180:
                candidates.append(text)

    if not candidates:
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        text = normalize_whitespace(text)
        if len(text) > 180:
            candidates.append(text)

    if not candidates:
        return ""

    return max(candidates, key=len)


def clean_article_text(text: str) -> str:
    """
    Удаляет рекламный и навигационный мусор из текста статьи
    Обрезает всё после ключевых маркеров, например: «подписывайтесь», «читайте также», «реклама» и пр.
    """
    text = normalize_whitespace(text)
    text = re.sub(r"(подписывайтесь|читайте также|все права защищены|реклама).*?$", "", text, flags=re.I)
    return normalize_whitespace(text)


def normalize_project_name(name: str) -> str:
    """
    Нормализует название проекта, убирает пометки вида «(2 проекта)» и лишние пробелы
    """
    name = normalize_whitespace(str(name or ""))
    name = re.sub(r"\(\d+\s*проект[а-я]*\)", "", name, flags=re.I).strip()
    name = re.sub(r"\s{2,}", " ", name)
    return name


def token_overlap_score(query_text: str, page_text: str) -> float:
    """
    Оценивает смысловое пересечение двух текстов через долю общих токенов.
    Учитываются только токены длиной ≥4 символа. Возвращает значение от 0 до 1
    """
    query_tokens = {t for t in re.findall(r"\w+", query_text.lower()) if len(t) >= 4}
    page_tokens = {t for t in re.findall(r"\w+", page_text.lower()) if len(t) >= 4}
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & page_tokens)
    return overlap / len(query_tokens)


class SeleniumFetcher:
    """
    Резервный загрузчик страниц через Chrome WebDriver.
    Используется для сайтов, недоступных через обычный HTTP-запрос (JS-рендеринг, защита от ботов).
    Экземпляр драйвера создаётся и уничтожается при каждом вызове fetch().
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def fetch(self, url: str) -> str:
        """
        Загружает HTML страницы через headless Chrome.
        Пауза 2 секунды после загрузки — ожидание завершения JS-рендеринга.
        Возвращает пустую строку при любой ошибке
        """
        options = Options()
        if self.settings.selenium_headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"user-agent={self.settings.user_agent}")

        driver = None
        try:
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            driver.set_page_load_timeout(self.settings.selenium_page_load_timeout)
            driver.get(url)
            time.sleep(2)  # Ждём завершения динамической загрузки страницы
            return driver.page_source
        except Exception:
            return ""
        finally:
            if driver:
                driver.quit()


class HTTPClient:
    """
    HTTP-клиент с повторными попытками и автоматическим fallback на Selenium.
    При коротком ответе (<1200 символов) или исчерпании попыток — переключается на SeleniumFetcher.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.user_agent})
        # Selenium-фетчер инициализируется только если включён в конфиге
        self.selenium_fetcher = SeleniumFetcher(settings) if settings.enable_selenium_fallback else None

    def fetch_html(self, url: str) -> str:
        """
        Загружает HTML по URL с повторными попытками.
        Задержка между попытками растёт линейно (1.2 * номер_попытки секунды).
        При слишком коротком ответе или финальной ошибке — пробует Selenium.
        """
        for attempt in range(1, self.settings.request_retries + 1):
            try:
                response = self.session.get(url, timeout=self.settings.request_timeout)
                response.raise_for_status()
                response.encoding = response.apparent_encoding or response.encoding
                html = response.text

                # Слишком короткий ответ — возможно, страница заблокировала запрос
                if len(html) < 1200 and self.selenium_fetcher:
                    html = self.selenium_fetcher.fetch(url)
                return html
            except Exception:
                if attempt == self.settings.request_retries and self.selenium_fetcher:
                    return self.selenium_fetcher.fetch(url)
                time.sleep(1.2 * attempt)
        return ""


class ArticleExtractor:
    """
    Извлекает и оценивает релевантность статьи для конкретного инвестиционного проекта.
    Отклоняет статьи с коротким текстом или слабым пересечением с данными проекта.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.http_client = HTTPClient(settings)

    def extract_article(self, result: SearchResult, project: ProjectRecord) -> Optional[Article]:
        """
        Загружает страницу по URL, парсит текст, заголовок и дату публикации.
        Оценивает релевантность через токенное пересечение с названием, регионом и отраслью проекта.
        Возвращает None, если текст слишком короткий или статья не связана с проектом.
        """
        html = self.http_client.fetch_html(result.url)
        if not html:
            return None

        soup = BeautifulSoup(html, "lxml")

        title = parse_article_title(soup) or result.title
        text = clean_article_text(parse_article_text(soup))
        published_at = parse_article_date(
            soup,
            html,
            fallback_texts=[result.snippet, title],
        )

        if published_at is None:
            published_at = parse_date_from_url(result.url)

        published_at = sanitize_article_date(published_at)

        if len(text) < 180:
            return None

        project_name = normalize_project_name(project.name)
        page_material = f"{title} {text} {result.snippet}"

        # Три независимых сигнала релевантности: название, регион, отрасль
        name_score = token_overlap_score(project_name, page_material)
        place_score = token_overlap_score(project.place, page_material) if project.place else 0.0
        industry_score = token_overlap_score(project.industry, page_material) if project.industry else 0.0

        # Взвешенный максимум — регион и отрасль усиливают, но не заменяют совпадение по названию
        relevance = max(name_score, place_score * 0.7, industry_score * 0.5)

        name_score = token_overlap_score(project_name, page_material)

        # Если имя проекта совсем не упоминается — отбрасываем
        if name_score < 0.08 and relevance < 0.20:
            return None

        # Для коротких или общих названий (< 3 значимых токенов) — требуем хотя бы регион
        name_tokens = [t for t in re.findall(r"\w+", project_name) if len(t) >= 4]
        if len(name_tokens) <= 2 and place_score < 0.15 and name_score < 0.15:
            return None

        return Article(
            project_name=project.name,
            title=title,
            text=text,
            published_at=published_at,
            url=result.url,
            domain=result.domain,
            query=result.query,
            reliability_score=result.score + relevance,
        )


def normalize_url(url: str) -> str:
    """
    Приводит URL к нормальному виду - убирает / и якорные фрагменты (#...).
    Используется для дедупликации по URL перед проверкой текстового сходства.
    """
    url = url.strip().rstrip("/")
    url = re.sub(r"#.*$", "", url)
    return url


def deduplicate_articles(articles: List[Article]) -> List[Article]:
    """
    Удаляет дублирующиеся статьи из списка.
    Дубликатом считается статья с совпадающим URL или высоким текстовым сходством
    (заголовок >92% или первые 1800 символов текста >95%).
    Итоговый список сортируется по дате публикации и надежности.
    """
    unique = []
    seen_urls = set()

    for article in articles:
        article.published_at = sanitize_article_date(article.published_at)

        url_norm = normalize_url(article.url)
        if url_norm in seen_urls:
            continue

        is_dup = False
        for existing in unique:
            title_sim = SequenceMatcher(None, article.title.lower(), existing.title.lower()).ratio()
            text_sim = SequenceMatcher(None, article.text[:1800].lower(), existing.text[:1800].lower()).ratio()
            if title_sim > 0.92 or text_sim > 0.95:
                is_dup = True
                break

        if not is_dup:
            seen_urls.add(url_norm)
            unique.append(article)

    # Сортировка от свежих статей до несвежих, при равной дате — по надёжности источника
    unique.sort(
        key=lambda x: (
            x.published_at or datetime(1900, 1, 1),
            x.reliability_score
        ),
        reverse=True
    )
    return unique