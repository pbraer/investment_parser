import re
import time
from typing import List
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from app.config import Settings
from app.logger import setup_logger
from app.schemas import ProjectRecord, SearchResult


logger = setup_logger(__name__)


# Домены, которые не нужны как результаты поиска
BLOCKED_RESULT_DOMAINS = {
    "google.com",
    "www.google.com",
    "webcache.googleusercontent.com",
    "support.google.com",
    "accounts.google.com",
    "policies.google.com",
    "maps.google.com",
    "translate.google.com",
}

# Явно мусорные типы ссылок
BLOCKED_URL_PATTERNS = [
    r"^https://www\.google\.",
    r"^https://google\.",
    r"^https://accounts\.google\.",
    r"^https://support\.google\.",
    r"^https://policies\.google\.",
    r"^https://maps\.google\.",
]

# Слова-маркеры, которые шумят в запросах
NOISE_TOKENS = {
    "част",
    "част.",
    "гос",
    "гос.",
    "проект",
    "проекты",
    "объект",
}


def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def is_blocked_url(url: str) -> bool:
    if not url:
        return True

    domain = extract_domain(url)
    if domain in BLOCKED_RESULT_DOMAINS:
        return True

    for pattern in BLOCKED_URL_PATTERNS:
        if re.search(pattern, url, flags=re.I):
            return True

    return False


def score_domain(domain: str, settings: Settings) -> float:
    """
    Эвристическая оценка доверия к домену.
    """
    domain = domain.lower()
    score = 0.0

    if domain.endswith(".gov.ru") or ".gov" in domain:
        score += 6.0

    if any(key in domain for key in ["official", "government", "adm", "admin", "invest", "economy", "minprom"]):
        score += 4.0

    if any(
        key in domain
        for key in [
            "tass",
            "interfax",
            "kommersant",
            "vedomosti",
            "rbc",
            "ria",
            "rg.ru",
            "forbes",
        ]
    ):
        score += 3.0

    if domain.endswith(".ru"):
        score += 1.0

    if any(key in domain for key in settings.trusted_domains_keywords):
        score += 1.5

    return score


def _dedupe_repeated_halves(text: str) -> str:
    """
    Если строка похожа на 'X X', оставляем один X.
    """
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return text

    parts = text.split()
    if len(parts) >= 6:
        half = len(parts) // 2
        left = " ".join(parts[:half]).strip().lower()
        right = " ".join(parts[half:]).strip().lower()
        if left == right:
            return " ".join(parts[:half]).strip()

    return text


def cleanup_text_tokens(text: str) -> str:
    text = str(text or "")
    text = text.replace("«", " ").replace("»", " ").replace('"', " ")
    text = re.sub(r"\(\d+\s*проект[а-я]*\)", "", text, flags=re.I)
    text = re.sub(r"[;|:,]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = re.findall(r"\w+|\S", text, flags=re.U)
    cleaned = []
    for token in tokens:
        low = token.lower().strip(".").strip()
        if low in NOISE_TOKENS:
            continue
        cleaned.append(token)

    text = " ".join(cleaned)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_project_name(name: str) -> str:
    """
    Чистит название проекта для поиска.
    """
    name = cleanup_text_tokens(name)
    name = re.sub(r"\bОЭЗ\s+ППТ\b", "", name, flags=re.I)
    name = re.sub(r"\bГруппа проектов\b", "", name, flags=re.I)
    name = re.sub(r"\bСвязанный проект\b", "", name, flags=re.I)
    name = re.sub(r"\bАрктическая зона РФ\b", "Арктика", name, flags=re.I)
    name = _dedupe_repeated_halves(name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def extract_description_hint(description: str) -> str:
    """
    Пытается вытащить осмысленный кусок из description.
    """
    description = str(description or "")
    parts = [p.strip() for p in re.split(r"[;|]", description) if p.strip()]

    candidates = []
    for part in parts:
        low = part.lower()
        if len(part) < 12:
            continue
        if any(
            x in low
            for x in [
                "топ-25",
                "холдинг",
                "связанный проект",
                "группа проектов",
                "площадка",
            ]
        ):
            continue
        candidates.append(part)

    if not candidates:
        return ""

    candidates.sort(key=len, reverse=True)
    return cleanup_text_tokens(candidates[0])


def is_generic_project_name(name: str) -> bool:
    """
    Слишком общие названия нуждаются в усилении location/industry/description.
    """
    name_l = str(name or "").lower().strip()
    generic_patterns = [
        r"^строительство\s+\w+",
        r"^гостиница$",
        r"^отель$",
        r"^строительство$",
        r"^проектирование$",
        r"^рудник$",
    ]
    return any(re.search(p, name_l) for p in generic_patterns)


def query_bonus(query: str, title: str, snippet: str) -> float:
    """
    Бонус к score, если слова из query есть в title/snippet.
    """
    bag = f"{title} {snippet}".lower()
    tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) >= 4]
    if not tokens:
        return 0.0

    hits = sum(1 for token in set(tokens) if token in bag)
    return min(hits * 0.3, 1.5)


def build_queries(project: ProjectRecord) -> List[str]:
    """
    Строит несколько 'человеческих' запросов для браузерного поиска.
    """
    queries = []

    base_name = normalize_project_name(project.name)
    place = cleanup_text_tokens(project.place)
    stage = cleanup_text_tokens(project.stage_raw)
    industry = cleanup_text_tokens(project.industry)
    desc_hint = extract_description_hint(project.description_raw)

    if base_name:
        queries.append(base_name)
        queries.append(f"{base_name} новости")
        queries.append(f"{base_name} инвестиционный проект")

    if base_name and place:
        queries.append(f"{base_name} {place}")

    if base_name and industry:
        queries.append(f"{base_name} {industry}")

    if base_name and project.start_date:
        queries.append(f"{base_name} {project.start_date.year}")

    if base_name and stage:
        queries.append(f"{base_name} {stage}")

    if base_name and desc_hint:
        queries.append(f"{base_name} {desc_hint}")

    if is_generic_project_name(base_name):
        if place:
            queries.append(f"{base_name} {place}")
        if industry:
            queries.append(f"{base_name} {industry}")
        if desc_hint:
            queries.append(f"{base_name} {desc_hint}")

    if desc_hint and place:
        queries.append(f"{desc_hint} {place}")

    if desc_hint and industry:
        queries.append(f"{desc_hint} {industry}")

    seen = set()
    unique_queries = []
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q not in seen:
            seen.add(q)
            unique_queries.append(q)

    return unique_queries


def clean_google_result_url(href: str) -> str:
    """
    Google search results часто имеют вид /url?q=...
    """
    href = str(href or "").strip()
    if not href:
        return ""

    if href.startswith("/url?"):
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        if "q" in qs and qs["q"]:
            return unquote(qs["q"][0])

    return href


class GoogleSeleniumSearchBackend:
    """
    Поиск через Selenium + Google.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def _make_driver(self):
        options = Options()
        if self.settings.selenium_headless:
            options.add_argument("--headless=new")

        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1600,1200")
        options.add_argument(f"user-agent={self.settings.user_agent}")

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options,
        )
        driver.set_page_load_timeout(self.settings.selenium_page_load_timeout)
        return driver

    @staticmethod
    def _try_accept_google_consent(driver) -> None:
        """
        Пытается принять consent/куки, если страница его показывает.
        """
        possible_selectors = [
            (By.ID, "L2AGLb"),
            (By.XPATH, "//button//*[contains(text(), 'Принять все')]/.."),
            (By.XPATH, "//button//*[contains(text(), 'Accept all')]/.."),
            (By.XPATH, "//button[contains(., 'Принять все')]"),
            (By.XPATH, "//button[contains(., 'Accept all')]"),
        ]

        for by, selector in possible_selectors:
            try:
                btn = WebDriverWait(driver, 2).until(
                    EC.element_to_be_clickable((by, selector))
                )
                btn.click()
                time.sleep(1)
                return
            except Exception:
                continue

    def _parse_google_html(self, html: str, query: str) -> List[SearchResult]:
        results: List[SearchResult] = []
        soup = BeautifulSoup(html, "lxml")

        seen_urls = set()

        # Ищем h3 внутри ссылок — это самый устойчивый способ для Google
        for a in soup.select("a[href]"):
            href = a.get("href", "").strip()
            url = clean_google_result_url(href)

            if not url or not url.startswith("http"):
                continue
            if is_blocked_url(url):
                continue
            if url in seen_urls:
                continue

            h3 = a.find("h3")
            if not h3:
                continue

            title = h3.get_text(" ", strip=True)
            if not title:
                continue

            seen_urls.add(url)

            snippet = ""
            parent = a.find_parent(["div"])
            if parent:
                text = parent.get_text(" ", strip=True)
                text = re.sub(r"\s+", " ", text).strip()
                if text and text != title:
                    snippet = text[:500]

            domain = extract_domain(url)
            score = score_domain(domain, self.settings)
            score += query_bonus(query, title, snippet)

            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    domain=domain,
                    score=score,
                    query=query,
                )
            )

        logger.info(f"Google parse | query='{query}' | parsed_results={len(results)}")
        return results

    def search(self, query: str) -> List[SearchResult]:
        driver = None
        try:
            driver = self._make_driver()
            google_url = f"https://www.google.com/search?q={quote_plus(query)}&hl=ru&gl=ru&num=10&pws=0"

            logger.info(f"Google Selenium | open query='{query}'")
            driver.get(google_url)
            time.sleep(2)

            self._try_accept_google_consent(driver)

            # Ждём body и немного выдачу
            WebDriverWait(driver, 8).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(1.5)

            html = driver.page_source
            logger.info(f"Google Selenium | query='{query}' | html_len={len(html)}")

            return self._parse_google_html(html, query)

        except Exception as exc:
            logger.warning(f"Google Selenium failed | query='{query}' | error={exc}")
            return []
        finally:
            if driver:
                driver.quit()


class DuckDuckGoFallbackBackend:
    """
    Fallback через DuckDuckGo HTML, если Google не дал результатов.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.user_agent})

    def search(self, query: str) -> List[SearchResult]:
        results: List[SearchResult] = []
        search_url = "https://html.duckduckgo.com/html/"

        try:
            response = self.session.post(
                search_url,
                data={"q": query},
                timeout=self.settings.request_timeout,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning(f"DDG fallback failed | query='{query}' | error={exc}")
            return results

        soup = BeautifulSoup(response.text, "lxml")
        links = soup.select("a.result__a, a.result-link, a[href*='uddg=']")

        seen = set()
        for link in links[: self.settings.max_results_per_query]:
            raw_url = link.get("href", "").strip()
            url = clean_google_result_url(raw_url)

            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            if "uddg" in qs:
                url = unquote(qs["uddg"][0])

            if not url.startswith("http"):
                continue
            if is_blocked_url(url):
                continue
            if url in seen:
                continue

            seen.add(url)

            title = link.get_text(" ", strip=True)
            parent = link.find_parent(["div", "td", "tr"])
            snippet = ""
            if parent:
                snippet_node = (
                    parent.select_one(".result__snippet")
                    or parent.select_one(".result-snippet")
                    or parent.select_one(".snippet")
                )
                if snippet_node:
                    snippet = snippet_node.get_text(" ", strip=True)

            domain = extract_domain(url)
            score = score_domain(domain, self.settings)
            score += query_bonus(query, title, snippet)

            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    domain=domain,
                    score=score,
                    query=query,
                )
            )

        logger.info(f"DDG fallback | query='{query}' | parsed_results={len(results)}")
        return results


class SearchClient:
    """
    Основной поисковый клиент:
    1. Google через Selenium
    2. если пусто — DuckDuckGo fallback
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.google_backend = GoogleSeleniumSearchBackend(settings)
        self.ddg_backend = DuckDuckGoFallbackBackend(settings)

    def search(self, query: str) -> List[SearchResult]:
        results = self.google_backend.search(query)
        if results:
            logger.info(f"Search backend=google | query='{query}' | results={len(results)}")
            return results

        results = self.ddg_backend.search(query)
        if results:
            logger.info(f"Search backend=ddg_fallback | query='{query}' | results={len(results)}")
            return results

        logger.info(f"Search backend=none | query='{query}' | results=0")
        return []

    def search_project(self, project: ProjectRecord) -> List[SearchResult]:
        queries = build_queries(project)[: self.settings.max_queries_per_project]
        logger.info(f"[{project.project_id}] built_queries={queries}")

        all_results: List[SearchResult] = []

        for query in queries:
            query_results = self.search(query)
            logger.info(f"[{project.project_id}] query='{query}' -> results={len(query_results)}")
            all_results.extend(query_results)

        unique_by_url = {}
        for item in all_results:
            if item.url not in unique_by_url or item.score > unique_by_url[item.url].score:
                unique_by_url[item.url] = item

        ranked = sorted(unique_by_url.values(), key=lambda x: x.score, reverse=True)
        return ranked[: self.settings.max_articles_per_project * 3]