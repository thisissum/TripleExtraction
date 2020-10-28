import os
import json
import json
import argparse
from collections import deque

from torch.utils.data import DataLoader

from .biaffine_extraction_model import BiaffineExtractor
from .util.func import to_device
from .modules.losses import BinaryFocalLoss

def args_parser():
    # start parser
    parser = argparse.ArgumentParser()

    # parameters
    parser.add_argument("--rela_num", default=49, type=int)
    parser.add_argument("--rela2id_path", default="./data/rela2id.json", type=str)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--LR", default=1e0-5, type=float)
    parser.add_argument("--pos_weight", default=5, type=float)
    parser.add_argument('--train_path', default="./data/train.json", type=str)
    parser.add_argument("--bert_dir", default=None, type=str)
    parser.add_argument("--seq_len", default=200, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--ckpt_dir", default="./checkpoint", type=str)
    parser.add_argument("--num_epoch", default=8, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)

    args = parser.parse_args()

    return args

def main():
    args = args_parser()
    with open(args.rela2id_path, 'r', encoding='utf-8') as f:
        rela2id = json.loads(f.readline())
    model = to_device(
        BiaffineExtractor(rela_num=args.rela_num, hidden_size=args.hidden_size, bert_dir=args.bert_dir),
        device=args.device
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion_with_weight = torch.nn.BCELoss(weight=torch.FloatTensor([1, args.pos_weight]))
    train_dataset = CustomDataset(
        data_path=args.train_path,
        bert_dir=args.bert_dir,
        rela2id=rela2id,
        seq_len=args.seq_len
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=var.BATCH_SIZE,
        shuffle=True
    )

    count = 0
    for epoch in range(args.num_epoch):
        if model.training == False:
            model.train()

        loss_queue = deque(maxlen=100)
        for batch_data in train_loader:
            batch_data = to_device(batch_data, args.device)

            sub_pred, obj_pred = model(batch_data)
            sub_golden, obj_golden = batch_data["sub_golden"], batch_data["obj_golden"]
            sub_loss = criterion_with_weight(sub_pred, sub_golden)
            obj_rela_loss = criterion_with_weight(obj_pred, obj_golden)
            loss = sub_loss + obj_rela_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_queue.append(loss.item())
            count += 1
            if count % 50 == 0:
                print("cur_loss: ", np.mean(loss_queue))

        # save model
        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "BiaffineExtractor_epoch_{}.pt".format(epoch)))
