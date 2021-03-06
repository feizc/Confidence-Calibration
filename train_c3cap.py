import argparse
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import torch 
from tqdm import tqdm 
from torch.nn import functional as nnf 
import os 
import multiprocessing 
import itertools
import numpy as np 
import random 
from torch.optim import Adam

from models import C3CapModel, TransformerConfig
from dataset import ClipCocoDataset 
from torch.utils.data import Dataset, DataLoader
import evaluation 
from evaluation import PTBTokenizer, Cider


import warnings
warnings.filterwarnings("ignore")

use_device = torch.cuda.is_available()
device = torch.device('cuda:0' if use_device else 'cpu') 
torch.backends.cudnn.benchmark = True

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)



def confidence_calibration_loss(logits, targets, smooth=0.0, alpha=1.0): 
    # logits [bsz, seq_len, vocab_size] 
    # targets [bsz, seq_len] 
    bsz, seq, vocab = logits.size()
    true_logprob = nnf.logsigmoid(logits) 
    false_logprob = torch.log(torch.maximum(1.0 - torch.exp(true_logprob), torch.tensor(1.0e-30).expand_as(true_logprob)))  
    tgt_true_logprob = torch.gather(true_logprob.view(bsz*seq, vocab), 1, targets.view(bsz*seq,1)).view(bsz, seq) # [bsz, seq]
    tgt_false_logprob = torch.gather(false_logprob.view(bsz*seq, vocab), 1, targets.view(bsz*seq,1)).view(bsz, seq)
    tgt_true_xent = -(alpha - 1) * tgt_true_logprob - 1 * tgt_false_logprob 
    tgt_false_xent = -(alpha - 1) * tgt_false_logprob - 1 * tgt_true_logprob 
    all_false_xent = - (alpha - 1) * false_logprob - 1 * true_logprob 
    loss = smooth * (torch.sum(all_false_xent, dim=-1) - tgt_false_xent) + tgt_true_xent 
    weights = torch.where(targets > 0, 1, 0).float() 
    return (loss * weights).sum() / weights.sum()



def evaluate_metrics(model, test_dataloader, tokenizer, epoch): 
    model.eval() 
    gen = {} 
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(test_dataloader)) as pbar:
        for idx, (_, tokens, _, img_features) in enumerate(test_dataloader):  
            img_features = img_features.to(device) 
            with torch.no_grad():
                text, _ = model.beam_search(img_features, beam_size=5, out_size=1) 

            caps_gt = tokenizer.batch_decode(tokens)
            caps_gen = tokenizer.batch_decode(text)

            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (idx, i)] = [gen_i, ]
                gts['%d_%d' % (idx, i)] = gts_i
            pbar.update()
            break
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_all_scores(gts, gen) 
    print(scores)
    return scores




def train_xe(model, train_dataloader, args, optimizer, scheduler, epoch): 
    model.train()
    running_loss = .0 
    progress = tqdm(total=len(train_dataloader), desc='C3CapModel') 
    for idx, (tokens, labels, _, img_features) in enumerate(train_dataloader):  
        model.zero_grad() 
        tokens, labels, img_features = tokens.to(device), labels.to(device), img_features.to(device, dtype=torch.float32) 
        outputs = model(img_features, tokens) 
        loss = confidence_calibration_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        progress.set_postfix({"loss": running_loss / (idx + 1)})
        progress.update()
        break 
    progress.close()
    return running_loss / len(train_dataloader)

    


def train_scst(model, train_dataloader, cider_train, args, optimizer, scheduler, epoch, tokenizer): 
    tokenizer_pool = multiprocessing.Pool() 
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    seq_len = model.language_decoder.max_len
    running_loss = .0
    beam_size = 5
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(train_dataloader)) as pbar:
        for it, (_, caps_gt, _, img_features) in enumerate(train_dataloader):
            img_features = img_features.to(device)
            outs, log_probs, logits = model.beam_search(img_features, beam_size=beam_size, out_size=beam_size, return_logits=True)
            optimizer.zero_grad()

            # Rewards
            caps_gen = tokenizer.batch_decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt))) 
            caps_gt = tokenizer.batch_decode(caps_gt)

            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider_train.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(img_features.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update() 
            break 

    loss = running_loss / len(train_dataloader)
    reward = running_reward / len(train_dataloader)
    reward_baseline = running_reward_baseline / len(train_dataloader)
    return loss, reward, reward_baseline




def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='./data/train.pkl') 
    parser.add_argument('--test_data_path', default='./data/test.pkl') 
    parser.add_argument('--tokenizer_path', default='./ckpt/gpt2') 
    parser.add_argument('--batch_size', default=5) 
    parser.add_argument('--lr', default=1e-2) 
    parser.add_argument('--epochs', default=10) 
    parser.add_argument('--warmup_steps', default=5000) 
    parser.add_argument('--out_dir', default='./ckpt') 
    parser.add_argument('--model_type', default='tic') 
    parser.add_argument('--phase', type=str, default='xs', choices=('xe', 'scst'))
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path) 
    train_dataset = ClipCocoDataset(args.train_data_path, tokenizer)  
    test_dataset = ClipCocoDataset(args.test_data_path, tokenizer)  

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    ref_caps_train = list(tokenizer.decode(text) for text in test_dataset.captions_tokens) 
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train)) 

    config = TransformerConfig()
    model = C3CapModel(config).to(device) 

    optimizer = AdamW(model.parameters(), lr=args.lr) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_dataloader)
    ) 

    use_rl = False
    best_cider = .0 
    patience = 0 


    for epoch in range(args.epochs): 
        if not use_rl: 
            train_loss = train_xe(model, train_dataloader, args, optimizer, scheduler, epoch) 
        else:
            train_loss, reward, reward_baseline = train_scst(model, train_dataloader, cider_train, args, optimizer, scheduler, epoch, tokenizer)
        
        scores = evaluate_metrics(model, test_dataloader, tokenizer, epoch)
        val_cider = scores['CIDEr'] 

        best = False 
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1 
        
        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True


        torch.save(
            model.state_dict(), 
            os.path.join(args.out_dir, f"{args.model_type}-{epoch:02d}.pt")
        )
        break 



if __name__ == '__main__': 
    main() 
