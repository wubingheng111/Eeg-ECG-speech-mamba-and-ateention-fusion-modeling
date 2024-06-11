from modeling.configuration_bme import BMEConfig
from modeling.modeling_bme import BMEModelForSequenceClassification

from scripts.Dataset import Merged_Dataset

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import os
import logging
import argparse
import transformers
transformers.logging.set_verbosity(logging.ERROR)
from torch.utils.tensorboard import SummaryWriter


def evaluate_metrics(logits, labels):
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == labels).item() / len(labels)
    print(preds)
    return acc, preds

# 训练函数
def trainer(
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    epochs: int,
    logger: logging.Logger
):
    writer = SummaryWriter()


    torch.manual_seed(233)
    model.to(device)
    # 如果是Linux, 编译模型
    if os.name == 'posix':
        torch.compile(model)
    # 记录模型参数数量
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(model)
    logger.info(f"Total Parameters: {num_parameters}")
    train_loss = []

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            # 获取数据
    
            inputs = {
                "eeg_input_ids" : batch["eeg_input_ids"].to(device),
                "ecg_input_ids" : batch["ecg_input_ids"].to(device),
                # "speech_input_ids" : batch["speech_input_ids"].to(device),
                "labels" : batch["labels"].to(device)
            }
            # print(inputs["labels"].tolist()[:10])
            # continue    
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(**inputs)
            loss = outputs.loss
            # 反向传播
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 更新参数
            optimizer.step()
            # 更新学习率
            scheduler.step()
            # 记录损失
            train_loss.append(loss.item())

            if (step+1) % (len(train_loader)//10) == 0:
            # if (step+1) % 1 == 0:
                train_loss = sum(train_loss) / len(train_loss)
                logger.info(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], lr: {scheduler.get_last_lr()[0]:.8f}, Loss: {train_loss}")
                # print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], lr: {scheduler.get_last_lr()[0]:.8f}, Loss: {train_loss}")
                writer.add_scalar("Loss/train", train_loss, epoch*len(train_loader)+step)
                train_loss = []

        # 模型评估 ACC,
        if (epoch+1) % 1 == 0:
            model.eval()
            accs = []
    
            for step, batch in enumerate(eval_loader):
    
                inputs = {
                "eeg_input_ids" : batch["eeg_input_ids"].to(device),
                "ecg_input_ids" : batch["ecg_input_ids"].to(device),
                # "speech_input_ids" : batch["speech_input_ids"].to(device),
                "labels" : batch["labels"].to(device)
                }

                outputs = model(**inputs)
                logits = outputs.logits
                labels = inputs["labels"]
                acc = evaluate_metrics(logits, labels)
                accs.append(acc)
            
            acc = sum(accs) / len(accs)
            logger.info(f"\nEpoch [{epoch+1}/{epochs}], Eval ACC: {acc:.4f}")
            writer.add_scalar("ACC/eval", acc, epoch)
            
            # 保存模型
            # model.save_pretrained(f"./modeling/cheems_epoch_{epoch+1}")       

if __name__ == "__main__":
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    # lr
    parser.add_argument("--lr", type=float, default=2e-5)
    # epochs
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    handler = logging.FileHandler("./logs/train_cheems.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(args)

    # 学习率
    def lrate(
        step: int,
    ) -> float:
        if step == 0:
            return 0.0
        return args.lr
    
    # 加载数据集
    train_dataset = Merged_Dataset(r'datasets\dataset_BME_depression\final_dataset', "train")
    test_dataset = Merged_Dataset(r'datasets\dataset_BME_depression\final_dataset', "test")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载模型
    config = BMEConfig()
    config.num_labels = 32
    model = BMEModelForSequenceClassification(config)

    # 衰减与非衰减参数分组
    no_decay = ["bias", "LayerNorm.weight"]
    decay = ["weight"]
    # 衰减与非衰减参数分组
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not p.requires_grad == False],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not p.requires_grad == False],
            "weight_decay": 0.0,
        },
    ]

    # 优化器, 学习率调度器, 混合精度训练
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer = AdamW(optimizer_grouped_parameters, lr=1, betas=(0.9, 0.98), eps=1e-8)
    scheduler = LambdaLR(optimizer, lr_lambda=lrate)

    model = trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        logger=logger
    )

