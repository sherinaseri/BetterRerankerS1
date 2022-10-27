from collections import defaultdict
import os, re
import random
from torch.utils.data import Dataset, DataLoader
import mmap
from tqdm import tqdm
import json

# reads the number of lines in a file
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def read_triple_ids(path):
    triple_ids = []
    with open(path, 'r') as f:
        for line in tqdm(f, total=get_num_lines(path), desc='read triple ids'):
            qid, pid, nid = line.rstrip('\n').split('\t')
            triple_ids.append([qid, pid, nid])
    return triple_ids

def read_collection(path):
    docs = {}
    with open(path, 'r') as f:
        for line in tqdm(f, desc='read docs'):
            data = line.rstrip('\n').split('\t')
            assert len(data) == 2
            docid, doctxt = data
            docs[docid] = doctxt
    return docs

def read_queries(path):
    queries = {}
    with open(path, 'r') as f:
        for line in tqdm(f, total=get_num_lines(path), desc='read queries'):
            data = line.rstrip('\n').split('\t')
            assert len(data) == 2
            qid, qtxt = data
            queries[qid] = qtxt
    return queries

def read_run(run_file, topK=500):
    run = defaultdict(dict)
    with open(run_file) as file:
        for line in tqdm(file, total=get_num_lines(run_file), desc='loading run data'):
            qid, _ , did, _ , score ,_ = line.split()
            score = float(score)
            run[qid][did] = score
            
    run_w_topk_retdocs = defaultdict(dict)
    for qid in run:
        retdocs = sorted(run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True)
        for retdoc in retdocs[:topK]:
            doc_id = retdoc[0]
            doc_score = retdoc[1]
            run_w_topk_retdocs[qid][doc_id] = doc_score
    
    return run_w_topk_retdocs

def read_qrels(qrel_file):
    qrels = dict()
    with open(qrel_file) as file:
        for line in tqdm(file, total=get_num_lines(qrel_file), desc='loading qrel data'):
            qid, _, did, rel = line.strip().split(' ')
            rel = int(rel)
            if qid in qrels:
                qrels[qid].append(did)
            else:
                qrels[qid] = [did]
    return qrels

def read_triples(triple_ids_path, queries_path, docs_path):
    triples = []
    triple_ids = read_triple_ids(triple_ids_path)
    queries = read_queries(queries_path)
    docs = read_collection(docs_path)
    for qid, pid, nid in triple_ids:
        triples.append([queries[qid], docs[pid], docs[nid]])
    return triples

def read_multilingual_triples(triple_ids_path, queries_path, list_docs_path):
    triples = []
    triple_ids = read_triple_ids(triple_ids_path)
    queries = read_queries(queries_path)
    list_docs = []
    for docs_path in list_docs_path:
        list_docs.append(read_collection(docs_path))
    
    for qid, pid, nid in tqdm(triple_ids, desc='collect triples'):
        query = queries[qid]
        pos, neg = [], []
        for docs in list_docs:
            pos.append(docs[pid])
            pos.append(docs[nid])
        triples.append([query] + pos + neg)
    return triples


def read_multilingual_triples_single_collection(triple_ids_path, queries_path, list_docs_path):
    triples = []
    triple_ids = read_triple_ids(triple_ids_path)
    queries = read_queries(queries_path)
    list_docs = []
    for docs_path in list_docs_path:
        list_docs.append(read_collection(docs_path))
    
    for qid, pid, nid in tqdm(triple_ids, desc='collect triples'):
        query = queries[qid]
        for docs in list_docs:
            triples.append([queries[qid], docs[pid], docs[nid]])
    return triples

def read_better_collection(args):
    collection_path = os.path.join(args.collection_file)
    docs = {}
    with open(collection_path, 'r') as file:
        for line in tqdm(file, desc='loading collection'):
            s = line.strip()
            metadata = json.loads(s)
            text = metadata['derived-metadata']['text'].strip()
            dtxt = " ".join(sentence_breaker(text))
            docid = metadata['derived-metadata']['id'].strip()
            docs[docid] = dtxt
    return docs

def sentence_breaker(text):
    text = text.replace('\\t', '').replace('\\r', '').replace('\\n', '')
    return [sent.strip() for sent in text.split('\n') if sent.strip() != '']

class RetrievalTriplesDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        content = self.dataset[idx]
        assert len(content) == 3
        query, pos, neg = content
        return [query, pos, neg]

class MultilingualRetrievalTriplesDataset(Dataset):
    def __init__(self, dataset, num_lang):
        self.dataset = dataset
        self.num_lang = num_lang
        assert self.num_lang > 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        content = self.dataset[idx]
        assert len(content) == 1 + self.num_lang * 2
        query = content[0:1]
        pos = content[1:1+self.num_lang]
        neg = content[1+self.num_lang:]
        return [query , pos , neg]
    
class CollectionDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        content = self.dataset[idx]
        assert len(content) == 2
        docid, text = content
        return [docid, text]


class QueryDataset(Dataset):
    def __init__(self, queries):
        self.queries = queries
        self.qids = list(self.queries.keys())
        
    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        return {'qid': qid, 'qtext':self.queries[qid]}
    

def get_retrieval_dataset(args):
    triples = read_triples(args.msm_triple_ids, args.msm_queries, args.msm_collection)
    triples_train = RetrievalTriplesDataset(triples[:args.num_train])
    triples_val = None
    if args.num_val:
        triples_val = RetrievalTriplesDataset(triples[args.num_train:args.num_train+args.num_val])
    print('in retrieval dataset')
    return triples_train, triples_val


def get_retrieval_dataset_multilingual_as_multi_label_loss_fn(args):
    triples = read_multilingual_triples(args.msm_triple_ids, args.msm_queries, args.msm_collections)
    triples_train = MultilingualRetrievalTriplesDataset(triples[:args.num_train], args.num_lang)
    triples_val = None
    if args.num_val:
        triples_val = MultilingualRetrievalTriplesDataset(triples[args.num_train:args.num_train+args.num_val], args.num_lang)
    print('in retrieval dataset')
    return triples_train, triples_val

def get_retrieval_dataset_multilingual_as_single_collection(args):
    triples = read_multilingual_triples_single_collection(args.msm_triple_ids, args.msm_queries, args.msm_collections)
    triples_train = RetrievalTriplesDataset(triples[:args.num_train])
    triples_val = None
    if args.num_val:
        triples_val = RetrievalTriplesDataset(triples[args.num_train:args.num_train+args.num_val])
    print('in retrieval dataset')
    return triples_train, triples_val

def get_collection_dataset(args):
    print("start loading collection!")
    docs = read_collection(args.collection)
    print("collection loaded!")
    collection = CollectionDataset(list(docs.items())[:200000])
    return collection


def get_rerank_dataset(args):
    queries = {}
    analytic_task = json.load(open(args.task_file))
    if args.mode == "AUTO":
        for task in analytic_task:
            task_docs = []
            for doc in task['task-docs']:
                task_docs.append(task['task-docs'][doc]['doc-text'])
            for req in task['requests']:
                req_id = req['req-num']
                req_docs = []
                for doc in req['req-docs']:
                    req_docs.append(req['req-docs'][doc]['doc-text'])
                if args.include_task_docs:
                    queries[req_id] = "\t".join(task_docs) + "\t" + "\t".join(req_docs)
                else:
                    queries[req_id] = "\t".join(req_docs)
    
    elif args.mode == "AUTO-HITL":
        for task in analytic_task:
            task_title = ''
            task_statement = ''
            task_narr = ''
            if 'task-title' in task:
                task_title = task['task-title']
            if 'task-stmt' in task:
                task_statement = task['task-stmt']
            if 'task-narr' in task:
                task_narr = task['task-narr']
            task_docs = []
            for doc in task['task-docs']:
                task_docs.append(task['task-docs'][doc]['doc-text'])
            task_docs_text = "\t".join(task_docs)
            for req in task['requests']:
                req_id = req['req-num']
                req_text = ''
                if 'req-text' in req:
                    if req['req-text'] is not None:
                        req_text = req['req-text']
                req_docs = []
                for doc in req['req-docs']:
                    req_docs.append(req['req-docs'][doc]['doc-text'])
                req_docs_text = "\t".join(req_docs)
                if args.include_task_docs:
                    queries[req_id] = "\t".join([task_title, task_statement, task_narr, task_docs_text, req_text, req_docs_text])
                else:
                    queries[req_id] = "\t".join([task_title, task_statement, task_narr, req_text, req_docs_text])
                queries[req_id] = re.sub(r'\t+', r'\t', queries[req_id])
    else:
        raise ValueError("Please pass the mode either as AUTO or AUTO-HITL")      
    
    docs = read_better_collection(args)
    qrels = read_qrels(args.test_qrels)
    runs = read_run(args.test_run, args.rerank_topK)   
    
    retrieved_docs = {}
    for q in runs:
        for doc in runs[q]:
            retrieved_docs[doc] = docs[doc]
    
    query_dataset = QueryDataset(queries)
    rerank_document_dataset = CollectionDataset(list(retrieved_docs.items()))

    return query_dataset, rerank_document_dataset, runs