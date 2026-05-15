from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "dataset_split",
    "project_id",
    "project_name",
    "location",
    "industry",
    "article_title",
    "article_text",
    "stage_label_en_manual",
    "interest_label",
    "review_status",
]

DESCRIPTION_REQUIRED_COLUMNS = [
    "dataset_split",
    "project_id",
    "project_name",
    "location",
    "industry",
    "article_title",
    "article_text",
    "target_description",
    "review_status",
]

PRICE_REQUIRED_COLUMNS = [
    "dataset_split",
    "project_id",
    "project_name",
    "article_title",
    "article_text",
    "target_price_json",
    "review_status",
]


def _load_tabular_file(path: str | Path, sheet_name: str | None = None) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, dtype=str).fillna("")

    if suffix in {".xlsx", ".xls"}:
        if sheet_name is None:
            return pd.read_excel(path, dtype=str).fillna("")
        return pd.read_excel(path, sheet_name=sheet_name, dtype=str).fillna("")

    raise ValueError(
        f"Неподдерживаемый формат файла: {path.suffix}. "
        f"Ожидается .csv, .xlsx или .xls"
    )


def _validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"В обучающем файле отсутствуют обязательные колонки: {missing}")


def _normalize_flag_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _apply_ready_filters(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "review_status" not in work.columns:
        raise ValueError("В обучающем файле отсутствует обязательная колонка review_status")

    work["review_status"] = _normalize_flag_series(work["review_status"])

    if "ready_for_train" in work.columns:
        work["ready_for_train"] = _normalize_flag_series(work["ready_for_train"])
    else:
        work["ready_for_train"] = ""

    work = work[work["review_status"] == "approved"].copy()

    if work["ready_for_train"].str.strip().ne("").any():
        work = work[work["ready_for_train"].isin(["yes", "1", "true", "y"])].copy()

    return work


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()

    if "dataset_split" not in work.columns:
        raise ValueError("В обучающем датасете отсутствует колонка dataset_split")

    work["dataset_split"] = work["dataset_split"].astype(str).str.strip().str.lower()

    train_df = work[work["dataset_split"] == "train"].copy()
    val_df = work[work["dataset_split"] == "val"].copy()
    test_df = work[work["dataset_split"] == "test"].copy()

    if train_df.empty:
        raise ValueError("В обучающем датасете нет строк с dataset_split = train")

    return train_df, val_df, test_df

def load_training_dataset(path: str | Path, sheet_name: str = "TRAIN_REAL_ARTICLES") -> pd.DataFrame:
    df = _load_tabular_file(path, sheet_name=sheet_name)
    _validate_columns(df, REQUIRED_COLUMNS)
    return df


def filter_ready_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = _apply_ready_filters(df)

    work = work[
        (work["article_text"].astype(str).str.strip() != "")
        & (work["stage_label_en_manual"].astype(str).str.strip() != "")
        & (work["interest_label"].astype(str).str.strip() != "")
    ].copy()

    return work


def build_training_text(df: pd.DataFrame) -> list[str]:
    texts: list[str] = []

    for _, row in df.iterrows():
        parts = [
            f"Проект: {row.get('project_name', '').strip()}",
            f"Регион: {row.get('location', '').strip()}",
            f"Отрасль: {row.get('industry', '').strip()}",
            f"Заголовок: {row.get('article_title', '').strip()}",
            f"Текст статьи: {row.get('article_text', '').strip()}",
        ]
        texts.append(" ".join([p for p in parts if p.strip()]))

    return texts

def load_t5_description_dataset(
    path: str | Path,
    sheet_name: str = "TRAIN_T5_DESCRIPTION",
) -> pd.DataFrame:
    df = _load_tabular_file(path, sheet_name=sheet_name)
    _validate_columns(df, DESCRIPTION_REQUIRED_COLUMNS)
    return df


def filter_ready_description_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = _apply_ready_filters(df)

    work = work[
        (work["article_text"].astype(str).str.strip() != "")
        & (work["target_description"].astype(str).str.strip() != "")
    ].copy()

    return work


def build_t5_description_sources(df: pd.DataFrame) -> list[str]:
    # Группируем по project_id и собираем все статьи одного проекта вместе
    sources: list[str] = []
    grouped = df.groupby("project_id", sort=False)

    for project_id, group in grouped:
        row = group.iloc[0]
        articles_text = "\n\n".join(
            f"Статья {i+1}: {str(r.get('article_title', '')).strip()}\n"
            f"{str(r.get('article_text', '')).strip()[:1400]}"
            for i, (_, r) in enumerate(group.iterrows())
        )
        source = (
            "Составь краткое и связное описание инвестиционного проекта на русском языке. "
            "Пиши так, как будто один человек коротко объясняет другому, о чем проект. "
            "Стиль должен быть естественным, но нейтральным и деловым. "
            "Не выдумывай факты и не добавляй того, чего нет в тексте.\n\n"
            f"Название проекта: {str(row.get('project_name', '')).strip()}\n"
            f"Регион: {str(row.get('location', '')).strip()}\n"
            f"Отрасль: {str(row.get('industry', '')).strip()}\n\n"
            f"Материалы:\n{articles_text}\n\n"
            "Описание:"
        )
        sources.append(source)

    return sources


def build_t5_description_targets(df: pd.DataFrame) -> list[str]:
    targets: list[str] = []
    for _, group in df.groupby("project_id", sort=False):
        targets.append(str(group.iloc[0].get("target_description", "")).strip())
    return targets


def build_t5_description_pairs(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    return build_t5_description_sources(df), build_t5_description_targets(df)


# цена - тренировка

def load_t5_price_dataset(
    path: str | Path,
    sheet_name: str = "TRAIN_T5_PRICE",
) -> pd.DataFrame:
    df = _load_tabular_file(path, sheet_name=sheet_name)
    _validate_columns(df, PRICE_REQUIRED_COLUMNS)
    return df


def filter_ready_price_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = _apply_ready_filters(df)

    work = work[
        (work["article_text"].astype(str).str.strip() != "")
        & (work["target_price_json"].astype(str).str.strip() != "")
    ].copy()

    return work


def _normalize_target_price_json(value: str) -> str:
    text = str(value or "").strip()

    if not text:
        return json.dumps(
            {"price_raw": "", "evidence": "", "confidence": 0.0},
            ensure_ascii=False,
        )

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("target_price_json должен быть JSON-объектом")

        normalized = {
            "price_raw": str(parsed.get("price_raw", "") or ""),
            "evidence": str(parsed.get("evidence", "") or ""),
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        }
        normalized["confidence"] = max(0.0, min(normalized["confidence"], 1.0))

        return json.dumps(normalized, ensure_ascii=False)
    except Exception:
        raise ValueError(f"Некорректный target_price_json: {text}")


def build_t5_price_sources(df: pd.DataFrame) -> list[str]:
    sources: list[str] = []

    for _, row in df.iterrows():
        source = (
            "Извлеки только стоимость инвестиционного проекта, если она явно указана в тексте. "
            "Игнорируй цены за номер, сутки, проживание, билеты, скидки, аренду, тарифы и бытовые цены. "
            "Ответь строго JSON без пояснений.\n"
            'Формат ответа: {"price_raw": "...", "evidence": "...", "confidence": 0.0}\n\n'
            f"Проект: {str(row.get('project_name', '')).strip()}\n"
            f"Заголовок: {str(row.get('article_title', '')).strip()}\n"
            f"Текст: {str(row.get('article_text', '')).strip()}\n"
        )
        sources.append(source)

    return sources


def build_t5_price_targets(df: pd.DataFrame) -> list[str]:
    targets: list[str] = []

    for _, row in df.iterrows():
        targets.append(_normalize_target_price_json(row.get("target_price_json", "")))

    return targets


def build_t5_price_pairs(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    return build_t5_price_sources(df), build_t5_price_targets(df)

import logging as _logging
_repo_logger = _logging.getLogger(__name__)

def check_split_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Предупреждает о пересечении project_id между сплитами.
    Пересечение train - val/test = data leakage означает что метрики нереальны
    """
    train_ids = set(train_df["project_id"].astype(str))
    val_ids   = set(val_df["project_id"].astype(str))
    test_ids  = set(test_df["project_id"].astype(str))

    val_leak  = train_ids & val_ids
    test_leak = train_ids & test_ids

    _repo_logger.info(
        f"Размеры сплитов — train: {len(train_df)}, "
        f"val: {len(val_df)}, test: {len(test_df)}"
    )

    if val_leak:
        _repo_logger.warning(
            f"DATA LEAKAGE: {len(val_leak)} project_id(s) "
            f"одновременно в train и val: {sorted(val_leak)[:10]}"
        )
    else:
        _repo_logger.info("val не пересекается с train по project_id")

    if test_leak:
        _repo_logger.warning(
            f"DATA LEAKAGE: {len(test_leak)} project_id(s) "
            f"одновременно в train и test: {sorted(test_leak)[:10]}"
        )
    else:
        _repo_logger.info("test не пересекается с train по project_id")

    if len(val_df) < 30:
        _repo_logger.warning(
            f"Val выборка маленькая ({len(val_df)} строк) — "
            f"метрики могут быть нерепрезентативны"
        )
    if len(test_df) < 30:
        _repo_logger.warning(
            f"Test выборка маленькая ({len(test_df)} строк) — "
            f"метрики могут быть нерепрезентативны"
        )
