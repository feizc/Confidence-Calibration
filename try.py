import torch 
from torch.nn import functional as nnf 

def confidence_calibration_loss(logits, targets, smooth=0.0, alpha=1.0): 
    # logits [bsz, seq_len, vocab_size] 
    # targets [bsz, seq_len] 
    bsz, seq, _ = logits.size()
    true_logprob = nnf.logsigmoid(logits) 
    false_logprob = torch.log(torch.maximum(1.0 - torch.exp(true_logprob), torch.tensor(1.0e-30).expand_as(true_logprob)))  
    tgt_true_logprob = torch.gather(true_logprob.view(bsz*seq, -1), 1, targets.view(-1,1)).view(bsz, seq) # [bsz, seq]
    tgt_false_logprob = torch.gather(false_logprob.view(bsz*seq, -1), 1, targets.view(-1,1)).view(bsz, seq)
    tgt_true_xent = -(1.0 - 1) * tgt_true_logprob - 1 * tgt_false_logprob 
    tgt_false_xent = -(1.0 - 1) * tgt_false_logprob - 1 * tgt_true_logprob 
    all_false_xent = - (1.0 - 1) * false_logprob - 1 * true_logprob 
    loss = smooth * (torch.sum(all_false_xent, dim=-1) - tgt_false_xent) + tgt_true_xent 
    weights = torch.where(targets > 0, 1, 0).float() 
    return (loss * weights).sum() / weights.sum()



logits = torch.randn(5, 12, 100) 
targets = torch.ones(5, 12).long()
loss = confidence_calibration_loss(logits, targets) 
print(loss)