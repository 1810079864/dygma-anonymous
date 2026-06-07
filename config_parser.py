import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_type", default='DyGMA', type=str,
                        help="which types of model you would use. model with multi-prompt(DyGMA) or single-prompt(base)")
    parser.add_argument("--model_name_or_path", default="./ckpts/bart-base", type=str,
                        help="pre-trained language model")
    parser.add_argument("--dataset_type", default="rams", type=str,
                        help="dataset type. Both sentence-level(ace_eeqa) and document-level(rams/wikievent)")
    parser.add_argument("--role_path", default='./data/dset_meta/description_rams.csv', type=str,
                        help="a file containing all role names. Read it to access all argument roles of this dataset")
    parser.add_argument("--prompt_path", default='./data/prompts/prompts_rams_full.csv', type=str,
                        help="a file containing all prompts we use for this dataset")
    parser.add_argument("--output_dir", default='./outputs', type=str,
                        help="output folder storing checkpoint and all sorts of log files")
    parser.add_argument("--keep_ratio", default=1.0, type=float,
                        help="The ratio of remaining traning samples. We drop the others. Used in Few-shot setting.")
    parser.add_argument('--inference_only', default=False, action="store_true",
                        help="The model will inference directly without training if it were set as True")
    parser.add_argument('--single', default=False, action="store_true",
                        help="The model will extract one event at a time if set as True")

    parser.add_argument("--pad_mask_token", default=0, type=int,
                        help="padding token id")
    parser.add_argument('--logging_steps', default=100, type=int,
                        help="step intervals for outputting log files")
    parser.add_argument('--eval_steps', default=500, type=int,
                        help="step intervals for validation")
    parser.add_argument("--max_span_length", default=10, type=int,
                        help="a heuristic constraint: the maximum length of extracted arguments")
    parser.add_argument("--batch_size", default=4, type=int, 
                        help="batch size during training. with BP")
    parser.add_argument("--infer_batch_size", default=32, type=int, 
                        help="batch size during inference. without BP")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_enc_seq_length", default=500, type=int,
                        help="maximum length for context")
    parser.add_argument("--window_size", default=260, type=int,
                        help="for document exceeding the length constraint, add a window centering at the trigger word and drop words outside this window")
    parser.add_argument("--encoder_layers", default=17, type=int,
                        help="encoder_layers")
    parser.add_argument('--context_representation', default="decoder", choices=['encoder', 'decoder'], type=str,
                        help="whether use the full BART (decoder) or only BART-encoder (encoder) to represent the context.")

    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=5.0, type=float)
    parser.add_argument("--max_steps", default=10000, type=int)
    parser.add_argument("--warmup_steps", default=0.1, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--lamb", default=1, type=float)
    parser.add_argument("--structural_type", default='biaffine', type=str)


    # setting only for the situation when inference_only
    parser.add_argument('--inference_model_path', default="./checkpoint", type=str,
                        help="The path of checkpoint used for inference.")
    parser.add_argument("--coref_model_path", default=None, type=str,
                        help="Optional local path to a fastcoref model. If omitted, DYGMA_COREF_MODEL or ./fcoref_model is used.")
    # setting only for base model.
    parser.add_argument("--max_dec_seq_length", default=20, type=int,
                        help="maximum length for single prompt")
    parser.add_argument("--max_span_num", default=1, type=int,
                        help="maximum arguments extracted for one role.")
    parser.add_argument('--th_delta', default=.0, type=float,
                        help="threshold controlling whether accept a candiate span as argument or not")
    # setting only for DyGMA model
    parser.add_argument("--max_prompt_seq_length", default=64, type=int,
                        help="maximum length for multi-prompt")
    parser.add_argument('--matching_method_train', default="max", choices=["max", 'accurate'], type=str,
                        help="start/end token matching method during training.")
    parser.add_argument('--bipartite', default=False, action="store_true",
                        help="whether use bipartite matching loss during training or not.")
    parser.add_argument("--schema_search_k",    default=2,   type=int,   help="每个触发词最多连接的同 Schema 候选实体数")
    parser.add_argument("--schema_edge_weight", default=0.3, type=float, help="Schema-aware 边的软连接权重")

    parser.add_argument("--schema_pos_bias", default=1.5,  type=float)
    parser.add_argument("--schema_neg_bias", default=-2.0, type=float)
    parser.add_argument("--null_score_weight", default=1.0, type=float,
                        help="版本6：no-answer/null score 加到位置0 logits上的缩放系数")
    parser.add_argument("--dep_order_weight", default=0.15, type=float,
                        help="版本4：句法可抽取性在角色顺序重排中的权重")
    parser.add_argument("--role_memory_coref_weight", default=0.10, type=float,
                        help="Memory共指优先级信号权重，供带Memory的后续版本使用")
    parser.add_argument('--role_planner_diff_weight', default=0.25, type=float,
                        help="角色顺序规划中的难度惩罚权重")
    parser.add_argument("--role_memory_write_threshold", default=0.8, type=float,
                        help="推理阶段memory写入的置信度阈值")
    parser.add_argument("--gat_query_fuse_weight", default=0.2, type=float,
                        help="GAT触发词拓扑特征正交投影后融合到query的残差权重")
    parser.add_argument("--memory_query_fuse_weight", default=0.2, type=float,
                        help="历史角色Memory正交投影后融合到query的残差权重")
    parser.add_argument("--use_memory_query_enhancement", default=True,
                        action="store_true",
                        help="启用历史角色Memory-aware query enhancement")
    parser.add_argument("--no_memory_query_enhancement",
                        dest="use_memory_query_enhancement",
                        action="store_false",
                        help="关闭历史角色Memory-aware query enhancement")
    parser.add_argument("--use_dep_role_order", default=True, action="store_true",
                        help="启用基于依存路径可抽取性的角色顺序重排")
    parser.add_argument("--no_dep_role_order", dest="use_dep_role_order",
                        action="store_false",
                        help="关闭基于依存路径可抽取性的角色顺序重排")
    parser.add_argument("--use_dep_memory", default=True, action="store_true",
                        help="有dep路径表示时启用三路Memory item融合")
    parser.add_argument("--no_dep_memory", dest="use_dep_memory",
                        action="store_false",
                        help="关闭三路Memory item融合")
      
    args = parser.parse_args()

    if args.inference_only:
        args.output_dir = "/".join(args.inference_model_path.split("/")[:-1])
    return args
