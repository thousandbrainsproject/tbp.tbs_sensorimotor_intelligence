import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Optional


def generate_predictions(trainer, model, datamodule, ckpt_path):
    """Generate predictions using the trained model."""
    return trainer.predict(
        model=model,
        dataloaders=datamodule.test_dataloader(),
        ckpt_path=ckpt_path,
    )


def compute_quaternion_error(predicted, target, reduction=None):
    from src.losses.loss import quaternion_geodesic_loss

    error_radians = quaternion_geodesic_loss(predicted, target, reduction=reduction)
    return error_radians * 180 / np.pi


def analyze_predictions(
    predictions,
    class_masker: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> pd.DataFrame:
    """Analyze model predictions and compute evaluation metrics."""
    df = pd.DataFrame(
        columns=[
            "real_class",
            "predicted_class",
            "real_quaternion",
            "predicted_quaternion",
            "quaternion_error_degs",
        ]
    )
    for batch in predictions:
        class_probs = batch["class_probabilities"]
        if class_masker:
            predicted_class = class_masker(class_probs)
        else:
            predicted_class = class_probs.argmax(dim=1)
        predicted_quaternion = batch["predicted_quaternion"]
        object_id = batch["object_ids"]
        unit_quaternion = batch["unit_quaternion"]
        quaternion_error = compute_quaternion_error(
            predicted_quaternion, unit_quaternion, reduction=None
        )
        batch_df = pd.DataFrame(
            {
                "real_class": object_id.tolist(),
                "predicted_class": predicted_class.tolist(),
                "real_quaternion": unit_quaternion.tolist(),
                "predicted_quaternion": predicted_quaternion.tolist(),
                "quaternion_error_degs": quaternion_error.tolist(),
            }
        )
        df = pd.concat([df, batch_df], ignore_index=True)
    return df


def save_and_summarize_results(df, save_dir, filename="predictions.csv", logger=print):
    summarize_predictions(df, logger=logger)
    save_path = Path(save_dir) / filename
    df.to_csv(save_path, index=False)
    logger(f"Predictions saved to {save_path}")


def summarize_predictions(df, logger=print):
    accuracy = (df["predicted_class"] == df["real_class"]).mean()
    mean_quaternion_error = df["quaternion_error_degs"].mean()
    logger(f"Accuracy: {accuracy:.4f}")
    logger(f"Mean quaternion error: {mean_quaternion_error:.4f} degrees")
