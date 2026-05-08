import argparse
import inspect
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, classification_report

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from app.config import Settings
from app.logger import setup_logger
from app.nlp import TransformerStageClassifier, InterestPredictor
from app.services import ProjectUpdateService
from app.training_repository import (
    load_training_dataset,
    filter_ready_rows,
    build_training_text,
    split_dataset,
    load_t5_description_dataset,
    filter_ready_description_rows,
    build_t5_description_pairs,
    load_t5_price_dataset,
    filter_ready_price_rows,
    build_t5_price_pairs,
)

logger = setup_logger(__name__)


class Seq2SeqTextDataset(Dataset):
    """
    PyTorch Dataset для обучения seq2seq-моделей (T5 и аналогов).
    Токенизирует пары (источник → цель) и заменяет padding в метках на -100,
    чтобы исключить их из вычисления функции потерь.
    """

    def __init__(
        self,
        sources: list[str],
        targets: list[str],
        tokenizer,
        max_source_length: int = 768,
        max_target_length: int = 128,
    ):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int):
        source = self.sources[idx]
        target = self.targets[idx]

        model_inputs = self.tokenizer(
            source,
            max_length=self.max_source_length,
            truncation=True,
            padding="max_length",
        )

        labels = self.tokenizer(
            text_target=target,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
        )

        # Padding-токены в метках заменяются на -100 — трансформер игнорирует их при расчёте loss
        label_ids = labels["input_ids"]
        label_ids = [
            token_id if token_id != self.tokenizer.pad_token_id else -100
            for token_id in label_ids
        ]

        model_inputs["labels"] = label_ids
        return {k: torch.tensor(v) for k, v in model_inputs.items()}


def _normalize_text_for_em(text: str) -> str:
    """Нормализует текст для сравнения"""
    return " ".join(str(text).strip().lower().split())


def _make_seq2seq_training_args(
    output_dir: str | Path,
    eval_dataset,
    settings: Settings,
    max_target_length: int,
    num_train_epochs: int,
    learning_rate: float,
    batch_size: int,
):
    """
    Формирует Seq2SeqTrainingArguments с совместимостью для разных версий transformers.
    Имя параметра определяется через inspect,
    чтобы код работал как на старых, так и на новых версиях библиотеки.
    """
    training_kwargs = {
        "output_dir": str(Path(output_dir) / "checkpoints"),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "predict_with_generate": True,
        "generation_max_length": max_target_length,
        "generation_num_beams": settings.seq2seq_num_beams,
        "logging_steps": 20,
        "save_strategy": "epoch",
        "save_total_limit": 2, # Хранит не более 2 последних чекпоинтов
        "report_to": [],
    }

    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    params = sig.parameters

    if "evaluation_strategy" in params:
        training_kwargs["evaluation_strategy"] = "epoch" if eval_dataset is not None else "no"
    elif "eval_strategy" in params:
        training_kwargs["eval_strategy"] = "epoch" if eval_dataset is not None else "no"

    return Seq2SeqTrainingArguments(**training_kwargs)


def _make_seq2seq_trainer(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    data_collator,
    tokenizer,
    compute_metrics,
):
    """
    Создаёт Seq2SeqTrainer с совместимостью по параметру tokenizer/processing_class
    """
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics if eval_dataset is not None else None,
    }

    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    trainer_params = trainer_sig.parameters

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    return Seq2SeqTrainer(**trainer_kwargs)


def _train_seq2seq_model(
    train_sources: list[str],
    train_targets: list[str],
    val_sources: list[str],
    val_targets: list[str],
    base_model_name: str,
    output_dir: str | Path,
    settings: Settings,
    run_name: str,
    max_target_length: int,
    num_train_epochs: int = 3,
    learning_rate: float = 3e-5,
    batch_size: int = 2,
) -> dict:
    """
    Функция обучения seq2seq-модели (T5 и аналогов)
    Используется как для description, так и для price assistant моделей

    Возвращает словарь с финальными метриками rouge1, rouge2, rougeL или exact_match.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[{run_name}] Запуск обучения seq2seq | "
        f"base_model={base_model_name} | output_dir={output_dir}"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    train_dataset = Seq2SeqTextDataset(
        sources=train_sources,
        targets=train_targets,
        tokenizer=tokenizer,
        max_source_length=settings.seq2seq_max_source_length,
        max_target_length=max_target_length,
    )

    eval_dataset = None
    if val_sources and val_targets:
        eval_dataset = Seq2SeqTextDataset(
            sources=val_sources,
            targets=val_targets,
            tokenizer=tokenizer,
            max_source_length=settings.seq2seq_max_source_length,
            max_target_length=max_target_length,
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    from rouge_score import rouge_scorer as rs_module

    def compute_metrics(eval_pred):
        """Вычисляет ROUGE-1/2/L по декодированным предсказаниям и эталонам."""
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        import numpy as np
        if isinstance(predictions, np.ndarray) and predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        # Восстанавливаем padding-токены перед декодированием (заменённые -100 при подготовке меток)
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        try:
            from rouge_score import rouge_scorer as rs_module
            scorer = rs_module.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
            r1, r2, rL = [], [], []
            for pred, gold in zip(pred_texts, label_texts):
                s = scorer.score(gold, pred)
                r1.append(s["rouge1"].fmeasure)
                r2.append(s["rouge2"].fmeasure)
                rL.append(s["rougeL"].fmeasure)

            return {
                "rouge1": float(np.mean(r1)),
                "rouge2": float(np.mean(r2)),
                "rougeL": float(np.mean(rL)),
            }
        except ImportError:
            exact_matches = sum(1 for p, g in zip(pred_texts, label_texts) if p.strip() == g.strip())
            return {"exact_match": exact_matches / max(len(pred_texts), 1)}

    training_args = _make_seq2seq_training_args(
        output_dir=output_dir,
        eval_dataset=eval_dataset,
        settings=settings,
        max_target_length=max_target_length,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    trainer = _make_seq2seq_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = {}
    if eval_dataset is not None:
        raw_metrics = trainer.evaluate()
        # Конвертируем numpy-скаляры в стандартные Python float
        metrics = {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in raw_metrics.items()
        }

        logger.info(f"[{run_name}] Запуск predict() для вычисления ROUGE")
        pred_output = trainer.predict(eval_dataset)

        raw_preds = pred_output.predictions
        if isinstance(raw_preds, tuple):
            raw_preds = raw_preds[0]
        if isinstance(raw_preds, np.ndarray) and raw_preds.ndim == 3:
            raw_preds = np.argmax(raw_preds, axis=-1)

        pred_ids = np.where(raw_preds != -100, raw_preds, tokenizer.pad_token_id)
        label_ids = np.where(pred_output.label_ids != -100,
                             pred_output.label_ids, tokenizer.pad_token_id)

        pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        for i in range(min(3, len(pred_texts))):
            logger.info(f"[{run_name}] PRED[{i}]: {pred_texts[i][:200]!r}")
            logger.info(f"[{run_name}] GOLD[{i}]: {label_texts[i][:200]!r}")

        try:
            from rouge_score import rouge_scorer as rs_module
            scorer = rs_module.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=False
            )
            r1, r2, rL = [], [], []
            for pred, gold in zip(pred_texts, label_texts):
                s = scorer.score(gold, pred)
                r1.append(s["rouge1"].fmeasure)
                r2.append(s["rouge2"].fmeasure)
                rL.append(s["rougeL"].fmeasure)
            metrics["rouge1"] = float(np.mean(r1))
            metrics["rouge2"] = float(np.mean(r2))
            metrics["rougeL"] = float(np.mean(rL))
            logger.info(
                f"[{run_name}] ROUGE → R1={metrics['rouge1']:.3f} "
                f"R2={metrics['rouge2']:.3f} RL={metrics['rougeL']:.3f}"
            )
        except ImportError:
            logger.warning(f"[{run_name}] rouge_score не получилось установить, ROUGE пропущен")

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info(f"[{run_name}] Обучение завершено успешно metrics={metrics}")
    return metrics


def train_models(dataset_path: str, sheet_name: str = "TRAIN_REAL_ARTICLES") -> None:
    """
    Обучает классификаторы стадии и привлекательности на датасете TRAIN_REAL_ARTICLES.
    Выполняет  сплит train/val/test, проверяет утечку данных между сплитами.
    При наличии test-сплита вычисляет и сохраняет test_metrics.json для stage-модели.
    """
    settings = Settings()
    settings.ensure_dirs()

    logger.info(f"Загрузка обучающего датасета: {dataset_path} | sheet={sheet_name}")
    df = load_training_dataset(dataset_path, sheet_name=sheet_name)
    df = filter_ready_rows(df)

    if df.empty:
        raise ValueError("После фильтрации approved/ready_for_train не осталось строк для обучения")

    from app.training_repository import check_split_leakage

    train_df, val_df, test_df = split_dataset(df)
    check_split_leakage(train_df, val_df, test_df)  # одни проекты не должны попасть в несколько сплитов

    train_texts = build_training_text(train_df)
    val_texts = build_training_text(val_df) if not val_df.empty else []
    _ = build_training_text(test_df) if not test_df.empty else []  # подготовка тест-текстов

    train_stage_labels = (
        train_df["stage_label_en_manual"].astype(str).str.strip().str.lower().tolist()
    )
    val_stage_labels = (
        val_df["stage_label_en_manual"].astype(str).str.strip().str.lower().tolist()
        if not val_df.empty else []
    )

    train_interest_labels = (
        train_df["interest_label"].astype(str).str.strip().str.lower().tolist()
    )
    val_interest_labels = (
        val_df["interest_label"].astype(str).str.strip().str.lower().tolist()
        if not val_df.empty else []
    )

    logger.info(
        f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}"
    )

    logger.info("Обучение stage model")
    stage_model = TransformerStageClassifier(settings)
    stage_metrics = stage_model.train(
        train_texts=train_texts,
        train_labels=train_stage_labels,
        val_texts=val_texts,
        val_labels=val_stage_labels,
    )
    logger.info(f"Stage model metrics: {stage_metrics}")

    if not test_df.empty:
        test_texts_eval = build_training_text(test_df)
        test_stage_labels_eval = (
            test_df["stage_label_en_manual"]
            .astype(str).str.strip().str.lower().tolist()
        )
        # Перезагружаем модель с диска — проверяем реальный инференс, а не состояние в памяти
        stage_model_for_test = TransformerStageClassifier(settings)
        test_preds = [stage_model_for_test.predict_one(t) for t in test_texts_eval]

        test_acc = accuracy_score(test_stage_labels_eval, test_preds)
        test_f1 = f1_score(
            test_stage_labels_eval, test_preds, average="macro",
            labels=list(set(test_stage_labels_eval)), zero_division=0
        )
        test_report = classification_report(
            test_stage_labels_eval, test_preds,
            output_dict=True, zero_division=0
        )
        test_metrics = {
            "test_size": len(test_texts_eval),
            "test_accuracy": round(test_acc, 4),
            "test_macro_f1": round(test_f1, 4),
            "test_report": test_report,
        }
        test_path = settings.stage_model_dir / "test_metrics.json"
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Stage model TEST metrics (n={len(test_texts_eval)}): "
            f"accuracy={test_acc:.3f}, macro_f1={test_f1:.3f}"
        )
    else:
        logger.warning("test сплит пуст — реальная оценка stage-модели недоступна")

    logger.info("Обучение interest model")
    interest_model = InterestPredictor(settings)
    interest_metrics = interest_model.train(
        train_texts=train_texts,
        train_labels=train_interest_labels,
        val_texts=val_texts,
        val_labels=val_interest_labels,
    )
    logger.info(f"Interest model metrics: {interest_metrics}")

    logger.info("Обучение stage/interest завершено успешно.")


def train_description_model(
    dataset_path: str,
    base_model_name: str,
    sheet_name: str = "TRAIN_T5_DESCRIPTION",
    output_dir: str | None = None,
    epochs: int = 3,
    learning_rate: float = 3e-5,
    batch_size: int = 2,
) -> None:
    """
    Дообучает T5-like модель на задаче генерации описания инвестиционного проекта.
    Датасет: пары (промпт с контекстом → целевое описание).
    Результат сохраняется в settings.description_model_dir или в переданный output_dir.
    """
    settings = Settings()
    settings.ensure_dirs()

    # Используем путь из настроек как fallback, если output_dir не задан явно
    final_output_dir = output_dir or str(settings.description_model_dir)

    logger.info(
        f"Загрузка T5 description датасета: {dataset_path} | "
        f"sheet={sheet_name} | model={base_model_name}"
    )

    df = load_t5_description_dataset(dataset_path, sheet_name=sheet_name)
    df = filter_ready_description_rows(df)

    if df.empty:
        raise ValueError("После фильтрации не осталось строк для обучения description model.")

    train_df, val_df, test_df = split_dataset(df)

    train_sources, train_targets = build_t5_description_pairs(train_df)
    val_sources, val_targets = (
        build_t5_description_pairs(val_df) if not val_df.empty else ([], [])
    )

    logger.info(
        f"[description] Train rows: {len(train_df)} | "
        f"Val rows: {len(val_df)} | Test rows: {len(test_df)}"
    )

    metrics = _train_seq2seq_model(
        train_sources=train_sources,
        train_targets=train_targets,
        val_sources=val_sources,
        val_targets=val_targets,
        base_model_name=base_model_name,
        output_dir=final_output_dir,
        settings=settings,
        run_name="description",
        max_target_length=300,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    logger.info(f"Description model metrics: {metrics}")
    logger.info(f"Description model saved to: {final_output_dir}")


def train_price_model(
    dataset_path: str,
    base_model_name: str,
    sheet_name: str = "TRAIN_T5_PRICE",
    output_dir: str | None = None,
    epochs: int = 3,
    learning_rate: float = 3e-5,
    batch_size: int = 2,
) -> None:
    """
    Дообучает T5-like модель на задаче извлечения стоимости инвестиционного проекта.
    Датасет: пары (промпт с текстом статьи и JSON с price_raw, evidence, confidence).
    Результат сохраняется в settings.price_assistant_model_dir или в переданный output_dir.
    """
    settings = Settings()
    settings.ensure_dirs()

    final_output_dir = output_dir or str(settings.price_assistant_model_dir)

    logger.info(
        f"Загрузка T5 price датасета: {dataset_path} | "
        f"sheet={sheet_name} | model={base_model_name}"
    )

    df = load_t5_price_dataset(dataset_path, sheet_name=sheet_name)
    df = filter_ready_price_rows(df)

    if df.empty:
        raise ValueError("После фильтрации не осталось строк для обучения price model")

    train_df, val_df, test_df = split_dataset(df)

    train_sources, train_targets = build_t5_price_pairs(train_df)
    val_sources, val_targets = (
        build_t5_price_pairs(val_df) if not val_df.empty else ([], [])
    )

    logger.info(
        f"[price] Train rows: {len(train_df)} | "
        f"Val rows: {len(val_df)} | Test rows: {len(test_df)}"
    )

    metrics = _train_seq2seq_model(
        train_sources=train_sources,
        train_targets=train_targets,
        val_sources=val_sources,
        val_targets=val_targets,
        base_model_name=base_model_name,
        output_dir=final_output_dir,
        settings=settings,
        run_name="price_assistant",
        max_target_length=96,    # Короткий JSON-ответ — 96 токенов достаточно
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    logger.info(f"Price assistant metrics: {metrics}")
    logger.info(f"Price assistant model saved to: {final_output_dir}")


def train_all_models(
    train_dataset_path: str,
    desc_dataset_path: str,
    price_dataset_path: str,
    base_desc_model_name: str,
    base_price_model_name: str,
    train_sheet: str = "TRAIN_REAL_ARTICLES",
    desc_sheet: str = "TRAIN_T5_DESCRIPTION",
    price_sheet: str = "TRAIN_T5_PRICE",
    desc_output_dir: str | None = None,
    price_output_dir: str | None = None,
    epochs: int = 3,
    learning_rate: float = 3e-5,
    batch_size: int = 2,
) -> None:
    """
    Запускает полный цикл обучения всех трёх моделей последовательно:
    1. stage + interest (TransformerStageClassifier + InterestPredictor)
    2. description (T5-like seq2seq)
    3. price assistant (T5-like seq2seq)
    """
    train_models(train_dataset_path, sheet_name=train_sheet)
    train_description_model(
        dataset_path=desc_dataset_path,
        base_model_name=base_desc_model_name,
        sheet_name=desc_sheet,
        output_dir=desc_output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    train_price_model(
        dataset_path=price_dataset_path,
        base_model_name=base_price_model_name,
        sheet_name=price_sheet,
        output_dir=price_output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )


def run_update(input_path: str, output_path: str) -> None:
    """
    Запускает обновление эксель-файла с проектами обученными моделями.
    Предполагает, что все три модели уже обучены и сохранены в settings.
    Создаёт родительские директории для input и output, если они не существуют.
    """
    settings = Settings()
    settings.ensure_dirs()

    Path(input_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    service = ProjectUpdateService(settings)
    service.update_excel(input_path, output_path)


def selfcheck(dataset_path: str | None = None) -> None:
    """
    Проверяет директории и наличие всех четырёх моделей.
    При передаче пути к датасету дополнительно логирует путь к датасету.
    """
    settings = Settings()
    settings.ensure_dirs()

    logger.info("Selfcheck OK")
    logger.info(f"cache_dir={settings.cache_dir}")
    logger.info(f"models_dir={settings.models_dir}")
    logger.info(f"articles_archive_dir={settings.articles_archive_dir}")

    # Проверяем наличие артефактов каждой модели по характерному файлу
    stage_exists = (settings.stage_model_dir / "config.json").exists()
    interest_exists = (settings.interest_model_dir / "interest_model.joblib").exists()
    description_exists = (settings.description_model_dir / "config.json").exists()
    price_assistant_exists = (settings.price_assistant_model_dir / "config.json").exists()

    logger.info(f"stage_model_exists={stage_exists}")
    logger.info(f"interest_model_exists={interest_exists}")
    logger.info(f"description_model_exists={description_exists}")
    logger.info(f"price_assistant_model_exists={price_assistant_exists}")

    if dataset_path:
        logger.info(f"dataset_path={dataset_path}")


def main():
    """
    CLI-точка входа. Поддерживает следующие подкоманды:
    - train          — обучение stage + interest моделей
    - train-desc     — обучение T5 description модели
    - train-price    — обучение T5 price assistant модели
    - train-all      — полный цикл обучения всех трёх моделей
    - update         — обновление project Excel обученными моделями
    - selfcheck      — диагностика окружения и наличия моделей
    """
    parser = argparse.ArgumentParser(
        description=(
            "Investment parser: обучение stage/interest моделей, "
            "обучение T5-like description/price моделей и обновление project Excel"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Обучение stage и interest моделей на TRAIN_REAL_ARTICLES",
    )
    train_parser.add_argument("--dataset", required=True, help="Путь к training Excel/CSV")
    train_parser.add_argument(
        "--sheet",
        default="TRAIN_REAL_ARTICLES",
        help="Имя листа с обучающими данными (игнорируется для CSV)",
    )

    desc_parser = subparsers.add_parser(
        "train-desc",
        help="Обучение T5-like модели генерации описания",
    )
    desc_parser.add_argument("--dataset", required=True, help="Путь к CSV/XLSX датасету")
    desc_parser.add_argument(
        "--sheet",
        default="TRAIN_T5_DESCRIPTION",
        help="Имя листа TRAIN_T5_DESCRIPTION (игнорируется для CSV)",
    )
    desc_parser.add_argument(
        "--model",
        required=True,
        help="Базовая seq2seq модель: HF model id или путь к модели",
    )
    desc_parser.add_argument(
        "--output-dir",
        required=False,
        help="Куда сохранить обученную description model",
    )
    desc_parser.add_argument("--epochs", type=int, default=3, help="Количество эпох")
    desc_parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    desc_parser.add_argument("--batch-size", type=int, default=2, help="Batch size")

    price_parser = subparsers.add_parser(
        "train-price",
        help="Обучение T5-like модели помощника по цене",
    )
    price_parser.add_argument("--dataset", required=True, help="Путь к CSV/XLSX датасету")
    price_parser.add_argument(
        "--sheet",
        default="TRAIN_T5_PRICE",
        help="Имя листа TRAIN_T5_PRICE (игнорируется для CSV)",
    )
    price_parser.add_argument(
        "--model",
        required=True,
        help="Базовая seq2seq модель - HF model id или путь к модели",
    )
    price_parser.add_argument(
        "--output-dir",
        required=False,
        help="Куда сохранить обученную price assistant model",
    )
    price_parser.add_argument("--epochs", type=int, default=3, help="Количество эпох")
    price_parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    price_parser.add_argument("--batch-size", type=int, default=2, help="Batch size")

    all_parser = subparsers.add_parser(
        "train-all",
        help="Обучить stage/interest + description + price за один запуск",
    )
    all_parser.add_argument("--train-dataset", required=True, help="Путь к TRAIN_REAL_ARTICLES")
    all_parser.add_argument("--desc-dataset", required=True, help="Путь к TRAIN_T5_DESCRIPTION")
    all_parser.add_argument("--price-dataset", required=True, help="Путь к TRAIN_T5_PRICE")

    all_parser.add_argument("--train-sheet", default="TRAIN_REAL_ARTICLES")
    all_parser.add_argument("--desc-sheet", default="TRAIN_T5_DESCRIPTION")
    all_parser.add_argument("--price-sheet", default="TRAIN_T5_PRICE")

    all_parser.add_argument("--desc-model", required=True, help="Базовая модель для description")
    all_parser.add_argument("--price-model", required=True, help="Базовая модель для price assistant")

    all_parser.add_argument("--desc-output-dir", required=False)
    all_parser.add_argument("--price-output-dir", required=False)

    all_parser.add_argument("--epochs", type=int, default=3)
    all_parser.add_argument("--lr", type=float, default=3e-5)
    all_parser.add_argument("--batch-size", type=int, default=2)

    update_parser = subparsers.add_parser(
        "update",
        help="Обновление project Excel обученными моделями",
    )
    update_parser.add_argument("--input", required=True, help="Путь к исходному project Excel")
    update_parser.add_argument("--output", required=True, help="Путь к обновленному project Excel")

    check_parser = subparsers.add_parser(
        "selfcheck",
        help="Проверка структуры проекта и наличия моделей",
    )
    check_parser.add_argument("--dataset", required=False, help="Опционально путь к training Excel/CSV")

    args = parser.parse_args()

    if args.command == "train":
        train_models(args.dataset, sheet_name=args.sheet)

    elif args.command == "train-desc":
        train_description_model(
            dataset_path=args.dataset,
            base_model_name=args.model,
            sheet_name=args.sheet,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

    elif args.command == "train-price":
        train_price_model(
            dataset_path=args.dataset,
            base_model_name=args.model,
            sheet_name=args.sheet,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

    elif args.command == "train-all":
        train_all_models(
            train_dataset_path=args.train_dataset,
            desc_dataset_path=args.desc_dataset,
            price_dataset_path=args.price_dataset,
            base_desc_model_name=args.desc_model,
            base_price_model_name=args.price_model,
            train_sheet=args.train_sheet,
            desc_sheet=args.desc_sheet,
            price_sheet=args.price_sheet,
            desc_output_dir=args.desc_output_dir,
            price_output_dir=args.price_output_dir,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

    elif args.command == "update":
        run_update(args.input, args.output)

    elif args.command == "selfcheck":
        selfcheck(args.dataset)


if __name__ == "__main__":
    main()