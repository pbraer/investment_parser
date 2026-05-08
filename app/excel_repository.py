import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from dateutil import parser as date_parser

from app.schemas import ProjectRecord

# Словарь синонимов к словам, характеризующим проект
COLUMN_ALIASES = {
    "project_id": ["id", "project_id", "код", "ид"],
    "start_date": ["start", "date_start", "дата старта", "старт", "начало"],
    "update_date": ["update", "updated", "дата обновления", "обновлено", "последнее обновление"],
    "price":       ["price", "investment", "стоимость", "инвестиции", "объем инвестиций"],
    "stage":       ["stage", "стадия", "этап"],
    "name":        ["name", "project_name", "название", "название проекта", "проект"],
    "industry":    ["otr", "industry", "отрасль", "сфера"],
    "place":       ["place", "location", "регион", "место", "местоположение"],
    "project_type": ["type", "тип", "форма", "вид"],
    "description": ["description", "описание", "comment", "комментарий"],
}


def _normalize_column_name(name: str) -> str:
    # Убирается всё лишнее из названия колонки - пробелы, спецсимволы, регистр
    return re.sub(r"[^a-zA-Zа-яА-Я0-9]+", "", str(name).strip().lower())


def detect_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    # Приведение всех заголовков таблицы к нормализованному виду для сравнения
    normalized_columns = {_normalize_column_name(col): col for col in df.columns}
    mapping: Dict[str, str] = {}

    # Перебор синонимов и поиск, какая колонка Excel соответствует каждому полю
    for target_field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalize_column_name(alias)
            if alias_norm in normalized_columns:
                mapping[target_field] = normalized_columns[alias_norm]
                break

    # Без идентификатора и названия работать не получится
    required = ["project_id", "name"]
    missing_required = [x for x in required if x not in mapping]
    if missing_required:
        raise ValueError(f"В Excel не найдены обязательные колонки: {missing_required}")

    return mapping


def parse_date_safe(value) -> datetime | None:
    # Если значение пустое или NaN - возвращается None
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value

    text = str(value).strip()
    if not text:
        return None

    try:
        # fuzzy=True позволяет распознавать даты даже внутри произвольного текста
        return date_parser.parse(text, dayfirst=True, fuzzy=True)
    except Exception:
        return None


def parse_price_to_mln_rub(value) -> float:
    # Если стоимость не указана - то считаем как 0
    if value is None or pd.isna(value):
        return 0.0

    text = str(value).replace(" ", "").replace(",", ".").lower()
    number_match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not number_match:
        return 0.0

    number = float(number_match.group(1))

    # Привожу к миллионам рублей в зависимости от единицы измерения в строке
    if "трлн" in text:
        return number * 1_000_000
    if "млрд" in text:
        return number * 1_000
    if "млн" in text:
        return number
    if "тыс" in text:
        return number / 1_000

    # Если единица не указана — предполагаем, что большое число уже в рублях, так как проект РФ
    return number / 1_000_000 if number > 1_000_000 else number


def load_projects_from_excel(path: str | Path) -> Tuple[pd.DataFrame, List[ProjectRecord], Dict[str, str]]:
    df = pd.read_excel(path, dtype=str)
    mapping = detect_column_mapping(df)

    records: List[ProjectRecord] = []
    for idx, row in df.iterrows():
        project = ProjectRecord(
            row_index=idx,
            project_id=str(row.get(mapping.get("project_id", ""), "")).strip(),
            name=str(row.get(mapping.get("name", ""), "")).strip(),
            start_date=parse_date_safe(row.get(mapping.get("start_date"))),
            update_date=parse_date_safe(row.get(mapping.get("update_date"))),
            price_raw=str(row.get(mapping.get("price", ""), "")).strip(),
            price_mln_rub=parse_price_to_mln_rub(row.get(mapping.get("price"))),
            stage_raw=str(row.get(mapping.get("stage", ""), "")).strip(),
            industry=str(row.get(mapping.get("industry", ""), "")).strip(),
            place=str(row.get(mapping.get("place", ""), "")).strip(),
            project_type=str(row.get(mapping.get("project_type", ""), "")).strip(),
            description_raw=str(row.get(mapping.get("description", ""), "")).strip(),
        )
        records.append(project)

    return df, records, mapping


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Добавляю служебные колонки, если их ещё нет в таблице
    output_columns = [
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
    ]
    for col in output_columns:
        if col not in df.columns:
            df[col] = None
    return df


def update_project_row(
    df: pd.DataFrame,
    row_index: int,
    result,
    mapping: dict | None = None,
    overwrite_core_fields: bool = False,
    update_reason: str = "",
) -> pd.DataFrame:
    df = ensure_output_columns(df)

    # Результаты анализа записываются в служебные колонки
    df.at[row_index, "actual_stage_ru"] = result.actual_stage
    df.at[row_index, "last_news_date"] = (
        result.last_news_date.strftime("%d.%m.%Y") if result.last_news_date else None
    )
    df.at[row_index, "generated_description"] = result.generated_description
    df.at[row_index, "interest_score"] = (
        round(result.interest_score, 2) if result.interest_score is not None else None
    )
    df.at[row_index, "interest_label"] = result.interest_label
    df.at[row_index, "source_urls"] = "\n".join(result.source_urls or [])
    df.at[row_index, "evidence_titles"] = " | ".join(result.evidence_titles or [])
    df.at[row_index, "evidence_dates"] = " | ".join(result.evidence_dates or [])
    df.at[row_index, "news_count"] = result.news_count
    df.at[row_index, "updated_at"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    df.at[row_index, "core_fields_updated"] = "yes" if overwrite_core_fields else "no"
    df.at[row_index, "update_reason"] = update_reason

    # Отдельно записываются данные о стоимости проекта
    df.at[row_index, "actual_price_found"] = result.actual_price_raw
    df.at[row_index, "actual_price_mln_rub"] = result.actual_price_mln_rub
    df.at[row_index, "price_source_excerpt"] = result.price_source_excerpt
    df.at[row_index, "price_updated"] = "yes" if result.price_updated else "no"

    # Перезаписываются основные рабочие поля только если это явно разрешено
    if overwrite_core_fields and mapping:
        if "update_date" in mapping and result.last_news_date:
            df.at[row_index, mapping["update_date"]] = result.last_news_date.strftime("%d.%m.%Y")

        if "stage" in mapping and result.actual_stage:
            df.at[row_index, mapping["stage"]] = result.actual_stage

        if "description" in mapping and result.generated_description:
            df.at[row_index, mapping["description"]] = result.generated_description

        if "price" in mapping and result.actual_price_raw and result.price_updated:
            df.at[row_index, mapping["price"]] = result.actual_price_raw

    return df


def save_projects_to_excel(df: pd.DataFrame, path: str | Path) -> None:
    # Создается папка, если её нет, таблица сохраняется без индексов строк
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)