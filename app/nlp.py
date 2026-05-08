import json
import re
from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from app.config import Settings


# Все возможные стадии жизненного цикла инвестиционного проекта
STAGE_LABELS = [
    "initiation",
    "design",
    "approval",
    "financing",
    "construction",
    "launch",
    "operation",
    "frozen",
    "cancelled",
]

# Прямое и обратное отображение: метка ↔ числовой идентификатор класса
STAGE_TO_ID = {label: idx for idx, label in enumerate(STAGE_LABELS)}
ID_TO_STAGE = {idx: label for label, idx in STAGE_TO_ID.items()}

# Перевод технических меток стадий на русский язык для отображения
STAGE_RU_MAP = {
    "initiation": "Инициация",
    "design": "Проектирование",
    "approval": "Согласование / экспертиза",
    "financing": "Финансирование / поиск инвестора",
    "construction": "Строительство",
    "launch": "Запуск / ввод в эксплуатацию",
    "operation": "Введен в эксплуатацию",
    "frozen": "Заморожен / приостановлен",
    "cancelled": "Отменен / закрыт",
}


def stage_to_russian(stage_en: str) -> str:
    """Возвращает русскоязычное название стадии. При неизвестной метке — 'Инициация'."""
    return STAGE_RU_MAP.get(stage_en, "Инициация")


@dataclass
class PriceExtractionHint:
    """
    Результат извлечения стоимости проекта из текста статьи.
    Не применяется к данным напрямую, а используется как кандидат для последующей валидации.
    """
    raw_text: str | None = None # Сырое значение стоимости, как оно указано в тексте
    evidence: str | None = None # Фрагмент текста, из которого извлечена стоимость
    confidence: float = 0.0 # Уверенность модели в правильности извлечения (0.0–1.0)


class StageDataset(Dataset):
    """
    PyTorch Dataset для обучения трансформерного классификатора стадий.
    Токенизирует тексты при инициализации и хранит метки в числовом виде.
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __getitem__(self, idx):
        # Формируем батч-элемент: токены + метка класса
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class TransformerStageClassifier:
    """
    Классификатор стадии инвестиционного проекта на базе трансформера (XLM-RoBERTa).
    При наличии сохранённой модели — загружается автоматически при инициализации.
    Обучение выполняется отдельно через метод train().
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_dir = settings.stage_model_dir
        self.tokenizer = None
        self.model = None
        self.is_loaded = False

        if self._model_exists():
            self.load()

    def _model_exists(self) -> bool:
        """Проверяет наличие сохранённой модели по файлу config.json в директории модели."""
        return (self.model_dir / "config.json").exists()

    def assert_trained(self) -> None:
        """Выбрасывает исключение, если модель ещё не обучена и не сохранена."""
        if not self._model_exists():
            raise FileNotFoundError(
                f"Stage model не найдена в {self.model_dir}. Сначала запустите train."
            )

    def load(self) -> None:
        """Загружает токенизатор и модель из сохранённой директории."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.is_loaded = True

    def predict_one(self, text: str) -> str:
        """
        Предсказывает стадию для одного текста.
        Возвращает строковую метку.
        """
        self.assert_trained()

        if not self.is_loaded:
            self.load()

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.settings.max_text_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            pred_id = int(torch.argmax(outputs.logits, dim=1).item())

        return ID_TO_STAGE[pred_id]

    def train(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: List[str] | None = None,
        val_labels: List[str] | None = None,
    ) -> dict:
        """
        Дообучает трансформер на обучающей выборке.
        При наличии валидационной выборки вычисляет accuracy и macro F1.
        Сохраняет модель, токенизатор и метрики в model_dir.
        Возвращает словарь с метриками валидации (или пустой словарь).
        """
        train_label_ids = [STAGE_TO_ID[x] for x in train_labels]

        tokenizer = AutoTokenizer.from_pretrained(self.settings.transformer_model_name, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.settings.transformer_model_name,
            num_labels=len(STAGE_LABELS),
        )

        train_dataset = StageDataset(
            texts=train_texts,
            labels=train_label_ids,
            tokenizer=tokenizer,
            max_length=self.settings.max_text_length,
        )

        eval_dataset = None
        if val_texts and val_labels:
            val_label_ids = [STAGE_TO_ID[x] for x in val_labels]
            eval_dataset = StageDataset(
                texts=val_texts,
                labels=val_label_ids,
                tokenizer=tokenizer,
                max_length=self.settings.max_text_length,
            )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "macro_f1": f1_score(labels, preds, average="macro"),
            }

        import inspect
        from transformers import TrainingArguments

        training_kwargs = dict(
            output_dir=str(self.model_dir / "checkpoints"),
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=20,
            save_strategy="epoch",
            remove_unused_columns=False,
            report_to=[],
        )

        sig = inspect.signature(TrainingArguments.__init__)
        params = sig.parameters
        if "evaluation_strategy" in params:
            training_kwargs["evaluation_strategy"] = "epoch" if eval_dataset else "no"
        elif "eval_strategy" in params:
            training_kwargs["eval_strategy"] = "epoch" if eval_dataset else "no"

        training_args = TrainingArguments(**training_kwargs)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics if eval_dataset is not None else None,
        )

        trainer.train()

        metrics = {}
        if eval_dataset is not None:
            metrics = trainer.evaluate()

        # Удаляем старые веса перед сохранением, чтобы не накапливались устаревшие файлы
        import shutil, os
        for f in self.model_dir.glob("*.safetensors"):
            try:
                f.unlink()
            except Exception:
                pass

        model.save_pretrained(self.model_dir)
        tokenizer.save_pretrained(self.model_dir)

        # Обновляем экземпляр — модель готова к инференсу без перезагрузки
        self.tokenizer = tokenizer
        self.model = model
        self.is_loaded = True

        metrics_path = self.model_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        return metrics


class InterestPredictor:
    """
    Классификатор инвестиционной перспективности проекта.
    Использует TF-IDF + логистическую регрессию.
    При наличии сохранённых файлов загружается автоматически при инициализации.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_path = settings.interest_model_dir / "interest_model.joblib"
        self.vectorizer_path = settings.interest_model_dir / "vectorizer.joblib"
        self.metrics_path = settings.interest_model_dir / "metrics.json"
        self.model = None
        self.vectorizer = None

        if self.model_path.exists() and self.vectorizer_path.exists():
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)

    def assert_trained(self) -> None:
        """Выбрасывает исключение, если модель или векторизатор не найдены на диске."""
        if not (self.model_path.exists() and self.vectorizer_path.exists()):
            raise FileNotFoundError(
                f"Interest model не найдена в {self.settings.interest_model_dir}. Сначала запустите train."
            )

    def predict(self, text: str) -> Tuple[float, str]:
        """
        Предсказывает метку перспективности и вероятность для переданного текста.
        Возвращает (score, label), где score — уверенность модели в процентах (0–100).
        """
        self.assert_trained()

        if self.model is None or self.vectorizer is None:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)

        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        labels = list(self.model.classes_)
        label = str(labels[idx])
        score = round(float(proba[idx]) * 100, 1)
        return score, label

    def train(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: List[str] | None = None,
        val_labels: List[str] | None = None,
    ) -> dict:
        """
        Обучает TF-IDF векторизатор и логистическую регрессию.
        При наличии валидационной выборки вычисляет accuracy, macro F1 и classification_report.
        Сохраняет модель, векторизатор и метрики на диск. Возвращает словарь метрик.
        """
        # Биграммный TF-IDF с ограничением словаря
        self.vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
        X_train = self.vectorizer.fit_transform(train_texts)

        self.model = LogisticRegression(max_iter=400)
        self.model.fit(X_train, train_labels)

        metrics = {}
        if val_texts and val_labels:
            X_val = self.vectorizer.transform(val_texts)
            preds = self.model.predict(X_val)
            metrics = {
                "accuracy": accuracy_score(val_labels, preds),
                "macro_f1": f1_score(val_labels, preds, average="macro"),
                "report": classification_report(val_labels, preds, output_dict=True),
            }

        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)

        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        return metrics


class DescriptionGenerator:
    """
    Генератор описания проекта.
    Отбирает наиболее релевантные предложения из переданных текстов по эвристическому скорингу.
    Не использует нейронные модели.
    """

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Разбивает текст на предложения, отфильтровывая слишком короткие фрагменты (<30 символов)."""
        text = re.sub(r"\s+", " ", text or "").strip()
        if not text:
            return []
        parts = re.split(r"(?<=[\.\!\?])\s+", text)
        return [p.strip() for p in parts if len(p.strip()) > 30]

    def generate(self, project_name: str, texts: List[str]) -> str:
        """
        Формирует краткое описание проекта из списка текстов.
        Дедуплицирует предложения и ранжирует их по релевантности названию проекта
        и наличию ключевых слов. Возвращает строку до 1000 символов.
        """
        all_sentences = []
        for text in texts:
            all_sentences.extend(self.split_sentences(text))

            # Дедупликация предложений с сохранением порядка
            seen = set()
            unique_sentences = []
            for s in all_sentences:
                key = s.strip().lower()
                if key not in seen:
                    seen.add(key)
                    unique_sentences.append(s)
            all_sentences = unique_sentences

        if not all_sentences:
            return ""

        # Токены названия проекта для поиска совпадений в предложениях
        name_tokens = [t.lower() for t in re.findall(r"\w+", project_name) if len(t) >= 4]

        scored = []
        for sent in all_sentences:
            s = sent.lower()
            score = 0
            score += min(len(sent) / 120, 3.0)  # Небольшой бонус за длину — более полные предложения предпочтительнее

            # Бонус за предметную лексику, характерную для инвестиционных проектов
            for kw in [
                "строительство",
                "производство",
                "мощность",
                "инвестиции",
                "запуск",
                "объект",
                "предприятие",
                "рудник",
                "завод",
                "комплекс",
                "площадка",
                "проект",
            ]:
                if kw in s:
                    score += 1.3

            # Бонус за совпадение токенов названия проекта — повышает тематическую точность
            token_hits = sum(1 for token in name_tokens if token in s)
            score += token_hits * 1.8

            # Штраф за рекламный и навигационный мусор
            if any(x in s for x in ["подписывайтесь", "читайте также", "реклама"]):
                score -= 3.0

            scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Берём топ-3 уникальных предложения по убыванию скора
        selected = []
        for _, sentence in scored:
            if len(selected) >= 3:
                break
            if sentence not in selected:
                selected.append(sentence)

        return " ".join(selected).strip()[:1000]


class Seq2SeqRuntime:
    """
    Runtime для T5-подобных seq2seq-моделей.
    Поддерживает beam search и sampling; ленивая загрузка модели при первом вызове generate_text.
    """

    def __init__(
        self,
        model_name_or_path: str | None,
        max_source_length: int = 768,
        max_target_length: int = 128,
        num_beams: int = 4,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_beams = num_beams

        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def is_enabled(self) -> bool:
        """Возвращает True, если задано имя или путь модели."""
        return bool(self.model_name_or_path)

    def _load(self) -> None:
        """Ленивая загрузка модели и токенизатора при первом обращении."""
        if not self.is_enabled():
            return
        if self.tokenizer is not None and self.model is not None:
            return  # Уже загружено

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

    def generate_text(self, prompt: str, do_sample: bool = False,
                      temperature: float = 0.85, top_p: float = 0.92) -> str:
        """
        Генерирует текст по промпту
        """
        if not self.is_enabled():
            return ""
        self._load()

        inputs = self.tokenizer(
            prompt, truncation=True,
            max_length=self.max_source_length, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=self.max_target_length,
            no_repeat_ngram_size=3,
            early_stopping=not do_sample,
        )
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs["num_beams"] = self.num_beams

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return re.sub(r"\s+", " ", text).strip()


class T5DescriptionGenerator:
    """
    Генератор описания проекта на базе T5-подобной seq2seq-модели.
    При отключённой модели или неудачной генерации автоматически переключается
    на DescriptionGenerator.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        model_name = (
            getattr(settings, "description_model_name", None)
            or getattr(settings, "generator_model_name", None)
        )

        self.runtime = Seq2SeqRuntime(
            model_name_or_path=model_name,
            max_source_length=getattr(settings, "seq2seq_max_source_length", 768),
            max_target_length=min(getattr(settings, "seq2seq_max_target_length", 128), 180),
            num_beams=getattr(settings, "seq2seq_num_beams", 4),
        )
        self.fallback = DescriptionGenerator()

    @staticmethod
    def _build_prompt(project_name: str, texts: List[str]) -> str:
        """Формирует промпт для модели из названия проекта и до 5 исходных текстов."""
        joined = "\n\n".join(t[:1400] for t in texts[:5] if t.strip())
        return (
            "Составь краткое и связное описание инвестиционного проекта на русском языке. "
            "Пиши так, как будто один человек коротко объясняет другому, о чем проект. "
            "Стиль должен быть естественным, но нейтральным и деловым. "
            "Не выдумывай факты и не добавляй того, чего нет в тексте. "
            "Нужен один абзац на 2-4 предложения.\n\n"
            f"Название проекта: {project_name}\n\n"
            f"Материалы:\n{joined}\n\n"
            "Описание:"
        )

    @staticmethod
    def _postprocess(text: str) -> str:
        """Убирает служебные префиксы, нормализует пробелы, обрезает до 1000 символов."""
        text = re.sub(r"^\s*(описание|краткое описание)\s*:\s*", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:1000]

    @staticmethod
    def _looks_valid(text: str) -> bool:
        """
        Проверяет качество сгенерированного текста.
        Отклоняет слишком короткие строки и тексты, похожие на JSON-артефакты модели.
        """
        if not text:
            return False
        if len(text) < 40:
            return False
        if len(text.split()) < 8:
            return False

        # JSON-артефакты возникают, если модель путает задачу с задачей T5PriceAssistant
        bad_markers = [
            '"price_raw"',
            '"confidence"',
            "{",
            "}",
            "json",
        ]
        text_l = text.lower()
        if any(marker in text_l for marker in bad_markers):
            return False

        return True

    def generate(self, project_name: str, texts: List[str]) -> str:
        """
        Генерирует описание проекта.
        Если T5 отключён, недоступен или вернул некачественный результат — возвращает extractive fallback.
        """
        fallback = self.fallback.generate(project_name, texts)

        if not getattr(self.settings, "enable_t5_description", False) or not self.runtime.is_enabled():
            return fallback

        try:
            prompt = self._build_prompt(project_name, texts)
            generated = self.runtime.generate_text(
                prompt, do_sample=True, temperature=0.85, top_p=0.92
            )
            generated = self._postprocess(generated)

            if not self._looks_valid(generated):
                return fallback

            return generated
        except Exception:
            return fallback


class T5PriceAssistant:
    """
    Помощник по извлечению стоимости инвестиционного проекта из текста статьи.
    Возвращает PriceExtractionHint как кандидата для внешней валидации.
    Работает только при включённом флаге enable_t5_price_assistant.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        model_name = (
            getattr(settings, "price_assistant_model_name", None)
            or getattr(settings, "generator_model_name", None)
        )

        self.runtime = Seq2SeqRuntime(
            model_name_or_path=model_name,
            max_source_length=getattr(settings, "seq2seq_max_source_length", 768),
            max_target_length=min(getattr(settings, "seq2seq_max_target_length", 128), 96),
            num_beams=getattr(settings, "seq2seq_num_beams", 4),
        )

    @staticmethod
    def _build_prompt(project_name: str, article_title: str, article_text: str) -> str:
        """
        Формирует промпт для извлечения стоимости.
        Явно указывает модели игнорировать бытовые цены — анализирует только цены, касающиеся вложения в проект.
        """
        return (
            "Извлеки только стоимость инвестиционного проекта, если она явно указана в тексте. "
            "Игнорируй цены за номер, сутки, проживание, билеты, скидки, аренду, тарифы и бытовые цены. "
            "Ответь строго JSON без пояснений.\n"
            'Формат ответа: {"price_raw": "...", "evidence": "...", "confidence": 0.0}\n\n'
            f"Проект: {project_name}\n"
            f"Заголовок: {article_title}\n"
            f"Текст: {article_text[:1800]}\n"
        )

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Ищет первый JSON-объект в тексте и десериализует его. При ошибке возвращает None."""
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def extract(self, project_name: str, article_title: str, article_text: str) -> PriceExtractionHint | None:
        """
        Извлекает стоимость проекта из текста статьи через T5-модель.
        При отключённом флаге, недоступной модели или любой ошибке возвращает None.
        """
        if not getattr(self.settings, "enable_t5_price_assistant", False) or not self.runtime.is_enabled():
            return None

        try:
            prompt = self._build_prompt(project_name, article_title, article_text)
            raw = self.runtime.generate_text(prompt)
            data = self._extract_json(raw)
            if not data:
                return None

            raw_text = str(data.get("price_raw") or "").strip() or None
            evidence = str(data.get("evidence") or "").strip() or None

            try:
                confidence = float(data.get("confidence", 0.0))
            except Exception:
                confidence = 0.0

            # Зажимаем значение уверенности в допустимый диапазон
            confidence = max(0.0, min(confidence, 1.0))

            if not raw_text:
                return None

            return PriceExtractionHint(
                raw_text=raw_text,
                evidence=evidence,
                confidence=confidence,
            )
        except Exception:
            return None