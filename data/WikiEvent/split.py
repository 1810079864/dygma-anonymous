import os
import copy
import jsonlines
import argparse
from itertools import chain
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. 改进后的 SemanticChunker (基于句子列表，而非生文本) ---
class SemanticChunker:
    def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.35):
        print(f"正在加载语义模型: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def chunk_sentences(self, sentence_list):
        """
        输入：句子列表 (List[str])，例如 ["I am a cat.", "He is a dog."]
        输出：每个句子对应的 chunk_id (List[int])，例如 [0, 0]
        """
        if not sentence_list:
            return []
        
        # 1. 计算所有句子的向量
        embeddings = self.model.encode(sentence_list)
        
        sentence_chunk_ids = [0] 
        current_chunk_id = 0

        # 2. 依次比较相邻句子
        for i in range(1, len(sentence_list)):
            prev_emb = embeddings[i-1].reshape(1, -1)
            curr_emb = embeddings[i].reshape(1, -1)
            
            # 计算余弦相似度
            similarity = cosine_similarity(prev_emb, curr_emb)[0][0]
            
            # 如果相似度低于阈值，说明话题转换，开启新 Chunk
            if similarity < self.threshold:
                current_chunk_id += 1
            
            sentence_chunk_ids.append(current_chunk_id)
            
        return sentence_chunk_ids

# --- 辅助函数 ---
def find_list(sub, full):
    sub_len = len(sub)
    full_len = len(full)
    pos = -1
    for i in range(full_len):
        if full[i: i+sub_len] == sub:
            pos = i
            break
    return pos

def overlap(span1, span2):
    return ((span1[1] >= span2[0] and span1[0] <= span2[1]) or
            (span2[1] >= span1[0] and span2[0] <= span1[1]))

# 初始化 Chunker
# 阈值建议：0.2 - 0.4 之间。太高会切得太碎，太低会分不开。
chunker = SemanticChunker(threshold=0.4) 

total_arg = 0
total_events = 0
total_instance = 0

def split(in_file, out_file, max_len=250):
    global total_arg
    global total_events
    global total_instance

    num_event_cnter = dict()
    reader = jsonlines.open(in_file)
    writer = jsonlines.open(out_file, mode='w')
    
    for line in reader:
        entity_dict = {entity['id']:entity for entity in line['entity_mentions']}
        if not line["event_mentions"]:
            continue
        doc_key = line["doc_id"]
        full_text = line['tokens']
        doc_length = len(full_text)
        
        # sents: List[List[str]] -> 文档中包含的所有句子，每个句子是 token 列表
        sents = [[s[0] for s in sent[0]] for sent in line['sentences']]
        sent_lens = [len(sent) for sent in sents]
        assert sum(sent_lens) == doc_length

        events = []
        event_tuples = []
        for i, event in enumerate(line["event_mentions"]):
            event_type = event['event_type']
            trigger = event['trigger']
            event_span = [trigger['sent_idx'], trigger['sent_idx']]
            trigger = [trigger['start'], trigger['end'], trigger['text']]
            
            args = []
            for arg_info in event['arguments']:
                arg_entity = entity_dict[arg_info['entity_id']]
                arg_sent_idx = arg_entity['sent_idx']
                arg = [arg_entity['start'], arg_entity['end'], arg_info['text'], arg_info['role']]
                args.append(arg)
                event_span = [min(event_span[0], arg_sent_idx), max(event_span[1], arg_sent_idx)]

            event = {'event_type': event_type, 'trigger': trigger, 'args': args}
            events.append(event)
            event_tuple = ([i], event_span)
            event_tuples.append(event_tuple)

        # --- Merging Logic (保持原样) ---
        num_event = len(event_tuples)
        tmp_tuples = copy.deepcopy(event_tuples)
        merged_indexs = []
        have_merged = True
        while have_merged:
            have_merged = False
            for i in range(num_event):
                if i in merged_indexs: continue
                tuple1 = tmp_tuples[i]; span1 = tuple1[-1]
                for j in range(i+1, num_event):
                    if j in merged_indexs: continue
                    tuple2 = tmp_tuples[j]; span2 = tuple2[-1]
                    if overlap(span1, span2):
                        merge_span = [min(span1[0], span2[0]), max(span1[1], span2[1])]
                        merge_tuple = (tuple1[0] + tuple2[0], merge_span)
                        tmp_tuples[i] = merge_tuple
                        tuple1 = merge_tuple
                        merged_indexs.append(j)
                        have_merged = True
        
        new_tuples = [tmp_tuples[i] for i in range(num_event) if i not in merged_indexs]

        # --- 处理每个切分后的样本 ---
        for i, (indice, merge_span) in enumerate(new_tuples):
            start_sent_index = merge_span[0]
            end_sent_index = merge_span[1]
            context_len = sum(sent_lens[start_sent_index: end_sent_index+1])

            # ... (Context Expansion Logic 保持原样) ...
            flag_expand_front = True
            flag_expand_back = True
            while flag_expand_front or flag_expand_back:
                index_front = start_sent_index - 1
                index_back = end_sent_index + 1
                if index_front >= 0:
                    if (context_len + sent_lens[index_front]) <= max_len:
                        start_sent_index = index_front
                        context_len += sent_lens[index_front]
                    else: flag_expand_front = False
                else: flag_expand_front = False

                if index_back < len(sents):
                    if (context_len + sent_lens[index_back]) <= max_len:
                        end_sent_index = index_back
                        context_len += sent_lens[index_back]
                    else: flag_expand_back = False
                else: flag_expand_back = False
            # ... (Context Expansion Logic End) ...

            # 获取当前上下文包含的句子列表 (List[List[str]])
            relevant_sents_tokens = sents[start_sent_index : end_sent_index+1]
            
            # 展平为 token 列表用于保存 context
            context_tokens = list(chain(*relevant_sents_tokens))
            
            # --- 2. 意群划分 (Sentence-Level -> Token-Level) ---
            
            # A. 将 token 列表转回 string 列表，供 BERT 编码
            relevant_sents_strs = [" ".join(sent) for sent in relevant_sents_tokens]
            
            # B. 获取每个句子的 Chunk ID
            # 例如: [0, 0, 1, 2, 2]
            sent_chunk_ids = chunker.chunk_sentences(relevant_sents_strs)
            
            # C. 将句子级 ID 扩展为 Token 级 ID (完美对齐)
            # 这是一个 List[int]，长度等于 len(context_tokens)
            token_chunk_ids = []
            for s_idx, sent_tokens in enumerate(relevant_sents_tokens):
                c_id = sent_chunk_ids[s_idx]
                # 这句话有多少个 token，就重复多少次这个 ID
                token_chunk_ids.extend([c_id] * len(sent_tokens))
            
            assert len(token_chunk_ids) == len(context_tokens), "Chunk ID 对齐错误！"

            # --- 构建 Event ---
            new_events = []
            offset = sum(sent_lens[0: start_sent_index]) if start_sent_index > 0 else 0
            
            for index in indice:
                event = copy.deepcopy(events[index])
                trigger = event['trigger']
                args = event['args']
                
                trigger[0] -= offset
                trigger[1] -= offset
                for k in range(len(args)):
                    args[k][0] -= offset
                    args[k][1] -= offset
                
                event['trigger'] = trigger
                event['args'] = args
                new_events.append(event)

                total_arg += len(event['args'])
                total_events += 1
            total_instance += 1
            
            # --- 3. 保存 ---
            # 直接保存 token_chunk_ids，Dataset 类里不需要再做任何对齐计算！
            # 只需要 lookup: chunk_id = sample['token_chunk_ids'][role_start_index]
            sample = {
                'id': '%d-%s' % (i, doc_key), 
                'context': context_tokens, 
                'events': new_events, 
                'token_chunk_ids': token_chunk_ids  # <--- 核心新特征
            }
            writer.write(sample)

            num_event = len(new_events)
            if num_event not in num_event_cnter: num_event_cnter[num_event] = 1
            else: num_event_cnter[num_event] += 1

    return num_event_cnter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    in_dir = args.indir
    out_dir = args.outdir
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    fnames = os.listdir(in_dir)
    for fn in fnames:
        fpath = os.path.join(in_dir, fn)
        out_fpath = os.path.join(out_dir, fn)
        print('\n', fpath)
        split(fpath, out_fpath, max_len=250)