import torch
import torch.nn.functional as F
from torch import nn

class DPR(nn.Module):
    def __init__(self, query_encoder, doc_encoder, args):
        super(DPR, self).__init__()
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.args = args
        self.labels = torch.arange(args.batch_size, dtype=torch.long, device=args.device)
        print(self.labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def match(self, q_ids, q_mask, d_ids, d_mask):
        q_reps = self.query(q_ids, q_mask)
        d_reps = self.doc(d_ids, d_mask)
        scores = self.score(q_reps, d_reps)
        loss = self.loss_fct(scores, self.labels[:scores.size(0)])
        return loss

    def query(self, input_ids, attention_mask):
        enc_reps = self.query_encoder(input_ids, attention_mask=attention_mask).pooler_output
        return enc_reps

    def doc(self, input_ids, attention_mask):
        enc_reps = self.doc_encoder(input_ids, attention_mask=attention_mask).pooler_output
        return enc_reps

    def score(self, query_reps, document_reps):
        return query_reps.mm(document_reps.t())

    def save(self, path):
        state = self.state_dict()
        torch.save(state, path)

    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        if type(checkpoint) is dict:
            self.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            self.load_state_dict(checkpoint, strict=True)

class VanillaDPR(DPR):
    def forward(self, q_ids, q_mask, d_ids, d_mask):
        return self.match(q_ids, q_mask, d_ids, d_mask)

