#!/usr/bin/env python
# coding: utf-8
import torch
import os, sys
from models.dpr import VanillaDPR
from arguments import get_rerank_parser
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModel

from util.dataset import get_rerank_dataset
from util.util import MixedPrecisionManager, query_tokenizer, doc_tokenizer, hugface_models, set_seed, trec_eval, query_tokenizer_paragraph, document_tokenizer_paragraph
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from collections import defaultdict
import torch.nn.functional as F


def score_agg(scores, query_reps, doc_reps, args):
    if args.score_agg == "colbert":
        max_input, max_indx = torch.max(scores, dim=1)
        return torch.mean(max_input).item()
    
    elif args.score_agg == "vector_rrf" :
        final_query_rep = torch.mean(query_reps, dim =0, keepdim=True)
        
        rank_matrix = torch.zeros(score_matrix.shape)
        _argsort = torch.argsort(score_matrix, descending=True)
        for row in range(len(_argsort)):
            for col in range(len(_argsort[row])):
                indx = _argsort[row][col]
                rank_matrix[row][indx] = col + 1
            
        rrf = torch.add(rank_matrix, 60)   
        rrf = rrf.pow(-1)
        rrf = torch.mean(rrf, dim=0, keepdim=True)
        rrf = rrf.to(args.device)
        final_doc_rep = rrf.mm(doc_reps)
        return F.cosine_similarity(final_query_rep, final_doc_rep).item()
    
    else:
        raise NotImplementedError

def get_datasets(args):
    query_dataset, rerank_document_dataset, run = get_rerank_dataset(args)  
    
    query_loader = torch.utils.data.DataLoader(query_dataset,
                                                   batch_size=args.batch_size,
                                                   drop_last=False,
                                                   shuffle=False)
    
    document_loader = torch.utils.data.DataLoader(rerank_document_dataset,
                                                      batch_size=args.batch_size, 
                                                      drop_last=False)
    
    return query_loader, document_loader, run


def rerank(model, tokenizer, args, query_loader, document_loader, baseline_run):
    model.eval()
    rerank_run = defaultdict(dict)
    
    with torch.no_grad():
        qreps = {}
        for item in tqdm(query_loader, desc=f'encode rerank queries'):
            qid = item['qid'][0]  
            qtext = item['qtext'][0]
            q_chunck_ids, q_chunk_mask = query_tokenizer_paragraph(qtext, args, tokenizer)
            q_reps = model.query(q_chunck_ids, q_chunk_mask)
            qreps[qid] = q_reps.detach().cpu()
        
        dreps = {}
        for item in tqdm(document_loader, desc=f'encode rerank documents'):
            docid, doctext = item   
            docid = docid[0]
            doctext = doctext[0]
            d_chunk_ids, d_chunk_mask = document_tokenizer_paragraph(doctext, args, tokenizer)
            d_reps = model.doc(d_chunk_ids, d_chunk_mask)
            dreps[docid] = d_reps.detach().cpu()
                
        
        for qid in tqdm(qreps):
            rank_list = baseline_run[qid]
            query_rep = qreps[qid].to(args.device)
            for docid in rank_list:
                document_rep = dreps[docid].to(args.device)
                scores = model.score(query_rep, document_rep)
                score = score_agg(model.score(query_rep, document_rep), query_rep, document_rep, args)            
                rerank_run[qid][docid] = score
    
    # writing the re-ranked run
    runf = os.path.join(args.output_dir, f'DPR-Reranked.run')
    with open(runf, 'wt') as runfile:
        for qid in tqdm(rerank_run, desc="writing re-ranked run ..."):
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')
    
    # changing the baseline run data structure for reciprocal rank fusion
    baseline_list = {}
    for qid in baseline_run:
        ranklist = list(sorted(baseline_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
        baseline_list.setdefault(qid, [])
        for retdoc in ranklist:
            baseline_list[qid].append(retdoc[0])
    
    # reciprocal rank fusion of baseline and re-ranked run
    fusion_run = defaultdict(dict)
    for qid in tqdm(rerank_run):
        rerank_list = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
        for i in range(len(rerank_list)):
            did = rerank_list[i][0]
            rerank_at = i + 1
            baseline_at = baseline_list[qid].index(did) + 1
            fused_score = (1/(args.rrf_k + rerank_at)) + (1/(args.rrf_k + baseline_at))
            fusion_run[qid][did] = fused_score
    
    # writing the fused run
    fusion_file = os.path.join(args.output_dir, f'Fused_DPR_Reranked_w_Baseline_rrf_k_{args.rrf_k}.run')
    with open(fusion_file, 'wt') as runfile:
        for qid in tqdm(fusion_run, "writing fused run ..."):
            dummy_score = 100000
            fused_scores = list(sorted(fusion_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(fused_scores):
                runfile.write(f'{qid} 0 {did} {i+1} {dummy_score} run\n')
                dummy_score -= 1
        
def main(args):
    args.score_agg = "colbert"
    args.batch_size = 1 # this should be 1 for the code to run correctly
    args.stride = 10
    args.rrf_k = 60
    set_seed(args.seed)
    args.device = torch.cuda.current_device()
    os.makedirs(args.output_dir, exist_ok=True)


    _, tokenizer_class, model_class = hugface_models(args.model_name)
    query_encoder = model_class.from_pretrained(os.path.join('../weights/', args.model_name), local_files_only=True)
    doc_encoder = model_class.from_pretrained(os.path.join('../weights/', args.model_name), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join('../weights/', args.model_name), local_files_only=True)
    model = VanillaDPR(query_encoder, doc_encoder, args)
    model = model.to(args.device)

    # load checkpoint
    if args.checkpoint:
        state_dict = torch.load(os.path.join(args.checkpoint))["model_state_dict"]
        model.load_state_dict(state_dict)
    
    query_loader, document_loader, baseline_run = get_datasets(args)  
    rerank(model, tokenizer, args, query_loader, document_loader, baseline_run)
     
if __name__ == '__main__':
    parser = get_rerank_parser()
    args = parser.parse_args()
    main(args)