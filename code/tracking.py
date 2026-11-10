import os


def start(trainer):
    try:
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT", "tem-challenge"),
                   entity=os.environ.get("WANDB_ENTITY"),
                   name=f"{type(trainer).__name__}-{trainer.plans_manager.dataset_name}"
                        f"-{trainer.num_epochs}ep-fold{trainer.fold}",
                   group=os.environ.get("TANDA", "nnunet"),
                   dir=os.environ.get("WANDB_DIR", "/tmp"))
        return wandb
    except Exception:
        return None


def epoch(wandb_mod, trainer):
    if wandb_mod is None:
        return
    try:
        get = lambda k: trainer.logger.get_value(k, step=-1)
        names = [k for k in trainer.dataset_json["labels"] if k not in ("background", "ignore")]
        wandb_mod.log({"epoch": trainer.current_epoch + 1,
                       "train_loss": get("train_losses"), "val_loss": get("val_losses"),
                       "lr": get("lrs"),
                       **{f"dice_{n}": v for n, v in zip(names, get("dice_per_class_or_region"))}},
                      step=trainer.current_epoch + 1)
    except Exception:
        pass


def finish(wandb_mod):
    if wandb_mod is not None:
        wandb_mod.finish()
