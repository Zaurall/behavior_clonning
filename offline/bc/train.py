import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from offline.bc.dataset import BCDataset, get_segment_ids
from offline.bc.evaluate import evaluate_online
from offline.bc.model import BCModel
from offline.config import BCConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train(config: BCConfig) -> None:
    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get segment IDs and split
    logger.info("Loading segment IDs...")
    segment_ids = get_segment_ids(Path(config.pgto_data_dir))
    logger.info(f"Training on {len(segment_ids)} segments")

    # Load dataset
    logger.info("Loading training data...")
    train_dataset = BCDataset(segment_ids, config, verbose=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize model
    model = BCModel(config).to(device)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    # Loss function
    criterion = nn.MSELoss()

    # Tracking
    best_online_cost = float("inf")

    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_samples = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_samples += len(labels)

        train_loss /= train_samples

        # Log
        lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch + 1:2d}/{config.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"LR: {lr:.2e}"
        )

        # Online evaluation
        if (epoch + 1) % config.eval_every_n_epochs == 0:
            logger.info(
                f"  Running online evaluation on {config.eval_num_segments} segments..."
            )

            # Save current model temporarily
            temp_path = output_dir / "bc_temp.pt"
            torch.save(model.state_dict(), temp_path)

            online_cost = evaluate_online(
                model_path=temp_path,
                num_segments=config.eval_num_segments,
                config=config,
            )
            logger.info(f"  → Online cost: {online_cost:.2f}")

            if online_cost < best_online_cost:
                best_online_cost = online_cost
                save_path = output_dir / "bc_best_online.pt"
                torch.save(model.state_dict(), save_path)
                logger.info(f"  → New best online cost, saved to {save_path}")

        scheduler.step()

    logger.info(f"Training complete. Best online cost: {best_online_cost:.2f}")


if __name__ == "__main__":
    train(BCConfig())
