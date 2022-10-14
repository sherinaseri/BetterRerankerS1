import argparse

def get_train_parser():
    """
    Generate a parameters parser for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_rank', type=int, default=0, metavar='N', help='node rank.')
    parser.add_argument('--local_rank', type=int, default=0, metavar='N', help='Local process rank.')
    parser.add_argument('--nproc_per_node', type=int, default=1, metavar='N', help='gpu process per node.')
    parser.add_argument("--job_name",default=None,type=str,required=True,help="wandb jobs name")
    parser.add_argument("--model_name",default=None,type=str,required=True,help="pretrained model name")
    parser.add_argument("--data_dir",default=None,type=str,required=True,help="dataset directory")
    
    parser.add_argument("--msm_collection",default=None,type=str,help="msmarco collection")
    parser.add_argument("--msm_queries",default=None,type=str,help="msmarco query")
    parser.add_argument("--msm_triple_ids",default=None,type=str,help="msmarco triples ids")
    parser.add_argument('--msm_collections', nargs='+', default=None, type=str, help="list of msmarco collections")
    
    parser.add_argument("--output_dir",default=None,type=str,required=True,help="output directory")
    parser.add_argument("--checkpoint",default=None,type=str,help="custom model init checkpoint")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="total number of training epochs.")
    parser.add_argument("--num_train", type=int, default=125000, help="training size")
    parser.add_argument("--num_val", type=int, default=1000, help="validation size")
    parser.add_argument("--num_lang", type=int, default=1, help="number of lan")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--logging_steps", type=int, default=1000, help="logging every n steps")
    parser.add_argument("--query_maxlen", type=int, default=32, help="max query token length")
    parser.add_argument("--doc_maxlen", type=int, default=180, help="max document token length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="gradient accumulation")
    parser.add_argument('--fp16', default=False, action='store_true', help="mixed precision training")
    return parser

def get_rerank_parser():
    """
    Generate a parameters parser for reranking.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--mode",default=None,type=str,required=True,help="AUTO or AUTO-HITL mode to run")
    parser.add_argument("--model_name",default=None,type=str,required=True,help="pretrained model name")
    parser.add_argument("--output_dir",default=None,type=str,required=True,help="output directory")
    parser.add_argument("--checkpoint",default=None,type=str,help="custom model init checkpoint")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--query_maxlen", type=int, default=32, help="max query token length")
    parser.add_argument("--doc_maxlen", type=int, default=180, help="max document token length")
    parser.add_argument("--rerank_topK", type=int, default=100, help="rerank topK from run file")
    parser.add_argument("--collection_file",default=None,type=str,required=True,help="The test corpus dir.")
    parser.add_argument("--task_file",default=None,type=str,required=True,help="The test query file.")
    parser.add_argument("--test_qrels",default=None,type=str,required=False,help="The test query relevant judgments in TREC style.")
    parser.add_argument("--test_run",default=None,type=str,required=False,help="The test rerank input data file in galago style.")
    # parser.add_argument("--trec_eval", type=str, required=True, help="trec eval file")
    parser.add_argument('--include_task_docs', default=False, action='store_true', help="including task documents in the query")
    return parser


def get_index_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_rank', type=int, default=0, metavar='N', help='node rank.')
    parser.add_argument('--local_rank', type=int, default=0, metavar='N', help='Local process rank.')
    parser.add_argument('--nproc_per_node', type=int, default=1, metavar='N', help='gpu process per node.')
    parser.add_argument("--model_name",default=None,type=str,required=True,help="pretrained model name")
    parser.add_argument("--checkpoint",default=None,type=str,help="custom model init checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size")
    parser.add_argument("--collection",default=None,type=str,help="collection in tsv file")
    parser.add_argument("--doc_maxlen", type=int, default=180, help="max document token length")
    parser.add_argument("--chunk_size", type=int, default=100000, help="cache vectors to file")
    parser.add_argument("--output_dir",default=None,type=str,required=True,help="output directory")
    return parser