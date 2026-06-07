# import os
# import re
# import sys

# sys.path.append("../")
# import torch
# import numpy as np
# from copy import deepcopy
# from torch.utils.data import Dataset
# from processors.processor_base import DSET_processor, SyntaxProvider, _get_event_group
# from utils import EXTERNAL_TOKENS, _PREDEFINED_QUERY_TEMPLATE


# class InputFeatures(object):
#     """A single set of features of data."""

#     def __init__(self, example_id, feature_id,
#                  event_type, event_trigger,
#                  enc_text, enc_input_ids, enc_mask_ids, all_ids, all_mask_ids,
#                  dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
#                  arg_quries, arg_joint_prompt, target_info, enc_attention_mask,
#                  old_tok_to_new_tok_index=None, full_text=None, arg_list=None,
#                  # ── 新增三个字段 ─────────────────────────────────────────
#                  dep_heads=None,
#                  dep_rels=None,
#                  event_groups=None,
#                  ):

#         self.example_id = example_id
#         self.feature_id = feature_id
#         self.event_type = event_type
#         self.event_trigger = event_trigger
#         self.num_events = len(event_trigger)

#         self.enc_text = enc_text
#         self.enc_input_ids = enc_input_ids
#         self.enc_mask_ids = enc_mask_ids

#         self.all_ids = all_ids
#         self.all_mask_ids = all_mask_ids
#         self.enc_attention_mask = enc_attention_mask

#         self.dec_prompt_texts = dec_prompt_text
#         self.dec_prompt_ids = dec_prompt_ids
#         self.dec_prompt_mask_ids = dec_prompt_mask_ids

#         if arg_quries is not None:
#             self.dec_arg_query_ids = [v[0] for k, v in arg_quries.items()]
#             self.dec_arg_query_masks = [v[1] for k, v in arg_quries.items()]
#             self.dec_arg_start_positions = [v[2] for k, v in arg_quries.items()]
#             self.dec_arg_end_positions = [v[3] for k, v in arg_quries.items()]
#             self.start_position_ids = [v['span_s'] for k, v in target_info.items()]
#             self.end_position_ids = [v['span_e'] for k, v in target_info.items()]
#         else:
#             self.dec_arg_query_ids = None
#             self.dec_arg_query_masks = None

#         self.arg_joint_prompt = arg_joint_prompt
#         self.target_info = target_info
#         self.old_tok_to_new_tok_index = old_tok_to_new_tok_index
#         self.full_text = full_text
#         self.arg_list = arg_list

#         # ── 新增字段赋值 ──────────────────────────────────────────────
#         # dep_heads / dep_rels：subword 粒度，长度 = len(all_ids)（含 prompt）
#         # event_groups：每个事件对应的意群 group id，长度 = num_events
#         self.dep_heads   = dep_heads   if dep_heads   is not None else []
#         self.dep_rels    = dep_rels    if dep_rels    is not None else []
#         self.event_groups = event_groups if event_groups is not None else []

#     def find_idx(self, target, list):
#         for i, item in enumerate(list):
#             if item == target:
#                 return i

#     def init_pred(self):
#         self.pred_dict_tok = [dict() for _ in range(self.num_events)]
#         self.pred_dict_word = [dict() for _ in range(self.num_events)]

#     def add_pred(self, role, span, event_index):
#         pred_dict_tok = self.pred_dict_tok[event_index]
#         pred_dict_word = self.pred_dict_word[event_index]
#         if role not in pred_dict_tok:
#             pred_dict_tok[role] = list()
#         if span not in pred_dict_tok[role]:
#             pred_dict_tok[role].append(span)
#             if span != (0, 0):
#                 if role not in pred_dict_word:
#                     pred_dict_word[role] = list()
#                 word_span = self.get_word_span(span)
#                 if word_span not in pred_dict_word[role]:
#                     pred_dict_word[role].append(word_span)

#     def set_gt(self):
#         self.gt_dict_tok = [dict() for _ in range(self.num_events)]
#         for i, target_info in enumerate(self.target_info):
#             for k, v in target_info.items():
#                 self.gt_dict_tok[i][k] = [(s, e) for (s, e) in zip(v["span_s"], v["span_e"])]

#         self.gt_dict_word = [dict() for _ in range(self.num_events)]
#         for i, gt_dict_tok in enumerate(self.gt_dict_tok):
#             gt_dict_word = self.gt_dict_word[i]
#             for role, spans in gt_dict_tok.items():
#                 for span in spans:
#                     if span != (0, 0):
#                         if role not in gt_dict_word:
#                             gt_dict_word[role] = list()
#                         word_span = self.get_word_span(span)
#                         gt_dict_word[role].append(word_span)

#     @property
#     def old_tok_index(self):
#         new_tok_index_to_old_tok_index = dict()
#         for old_tok_id, (new_tok_id_s, new_tok_id_e) in enumerate(self.old_tok_to_new_tok_index):
#             for j in range(new_tok_id_s, new_tok_id_e):
#                 new_tok_index_to_old_tok_index[j] = old_tok_id
#         return new_tok_index_to_old_tok_index

#     def get_word_span(self, span):
#         if span == (0, 0):
#             raise AssertionError()
#         offset = 0
#         span = list(span)
#         span[0] = min(span[0], max(self.old_tok_index.keys()))
#         span[1] = max(span[1] - 1, min(self.old_tok_index.keys()))
#         while span[0] not in self.old_tok_index:
#             span[0] += 1
#         span_s = self.old_tok_index[span[0]] + offset
#         while span[1] not in self.old_tok_index:
#             span[1] -= 1
#         span_e = self.old_tok_index[span[1]] + offset
#         while span_e < span_s:
#             span_e += 1
#         return (span_s, span_e)

#     def __repr__(self):
#         s = ""
#         s += "example_id: {}\n".format(self.example_id)
#         s += "event_type: {}\n".format(self.event_type)
#         s += "trigger_word: {}\n".format(self.event_trigger)
#         s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
#         s += "enc_input_ids: {}\n".format(self.enc_input_ids)
#         s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
#         s += "dec_prompt_ids: {}\n".format(self.dec_prompt_ids)
#         s += "dec_prompt_mask_ids: {}\n".format(self.dec_prompt_mask_ids)
#         s += "event_groups: {}\n".format(self.event_groups)
#         s += "dep_heads[:10]: {}\n".format(self.dep_heads[:10])
#         s += "dep_rels[:10]: {}\n".format(self.dep_rels[:10])
#         return s


# class ArgumentExtractionDataset(Dataset):
#     def __init__(self, features):
#         self.features = features

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx]

#     @staticmethod
#     def collate_fn(batch):
#         enc_input_ids = torch.tensor([f.enc_input_ids for f in batch])
#         enc_mask_ids = torch.tensor([f.enc_mask_ids for f in batch])

#         all_ids = torch.tensor([f.all_ids for f in batch])
#         all_mask_ids = torch.tensor([f.all_mask_ids for f in batch])

#         if batch[0].dec_prompt_ids is not None:
#             dec_prompt_ids = torch.tensor([f.dec_prompt_ids for f in batch])
#             dec_prompt_mask_ids = torch.tensor([f.dec_prompt_mask_ids for f in batch])
#         else:
#             dec_prompt_ids = None
#             dec_prompt_mask_ids = None

#         example_idx = [f.example_id for f in batch]
#         feature_idx = torch.tensor([f.feature_id for f in batch])

#         if batch[0].dec_arg_query_ids is not None:
#             dec_arg_query_ids = [torch.LongTensor(f.dec_arg_query_ids) for f in batch]
#             dec_arg_query_mask_ids = [torch.LongTensor(f.dec_arg_query_masks) for f in batch]
#             dec_arg_start_positions = [torch.LongTensor(f.dec_arg_start_positions) for f in batch]
#             dec_arg_end_positions = [torch.LongTensor(f.dec_arg_end_positions) for f in batch]
#             start_position_ids = [torch.FloatTensor(f.start_position_ids) for f in batch]
#             end_position_ids = [torch.FloatTensor(f.end_position_ids) for f in batch]
#         else:
#             dec_arg_query_ids = None
#             dec_arg_query_mask_ids = None
#             dec_arg_start_positions = None
#             dec_arg_end_positions = None
#             start_position_ids = None
#             end_position_ids = None

#         target_info = [f.target_info for f in batch]
#         old_tok_to_new_tok_index = [f.old_tok_to_new_tok_index for f in batch]
#         arg_joint_prompt = [f.arg_joint_prompt for f in batch]
#         arg_lists = [f.arg_list for f in batch]
#         event_trigger = [f.event_trigger for f in batch]
#         enc_attention_mask = [f.enc_attention_mask for f in batch]

#         # ── 新增：三个图字段直接以 list 传出，无需 padding ──────────────
#         dep_heads_batch    = [f.dep_heads    for f in batch]   # list[list[int]]
#         dep_rels_batch     = [f.dep_rels     for f in batch]   # list[list[str]]
#         event_groups_batch = [f.event_groups for f in batch]   # list[list[int]]

#         return enc_input_ids, enc_mask_ids, all_ids, all_mask_ids, \
#                dec_arg_query_ids, dec_arg_query_mask_ids, \
#                dec_prompt_ids, dec_prompt_mask_ids, \
#                target_info, old_tok_to_new_tok_index, arg_joint_prompt, arg_lists, \
#                example_idx, feature_idx, \
#                dec_arg_start_positions, dec_arg_end_positions, \
#                start_position_ids, end_position_ids, event_trigger, enc_attention_mask, \
#                dep_heads_batch, dep_rels_batch, event_groups_batch   # ← 新增


# class MultiargProcessor(DSET_processor):
#     def __init__(self, args, tokenizer):
#         super().__init__(args, tokenizer)
#         self.set_dec_input()
#         self.collate_fn = ArgumentExtractionDataset.collate_fn

#         # ── 新增：句法解析器（全局单例，避免重复加载）────────────────────
#         # SyntaxProvider 已在 processor_base 中定义，此处直接复用父类实例
#         # 如果父类 __init__ 中没有初始化，在这里初始化一次即可
#         if not hasattr(self, 'syntax_provider'):
#             self.syntax_provider = SyntaxProvider()

#     def set_dec_input(self):
#         self.arg_query = False
#         self.prompt_query = False
#         if self.args.model_type == "base":
#             self.arg_query = True
#         elif "DyGMA" in self.args.model_type:
#             self.prompt_query = True
#         else:
#             raise NotImplementedError(f"Unexpected setting {self.args.model_type}")

#     @staticmethod
#     def _read_prompt_group(prompt_path):
#         with open(prompt_path) as f:
#             lines = f.readlines()
#         prompts = dict()
#         for line in lines:
#             if not line:
#                 continue
#             event_type, prompt = line.split(":")
#             prompts[event_type] = prompt
#         return prompts

#     def create_dec_qury(self, arg, event_trigger):
#         dec_text = _PREDEFINED_QUERY_TEMPLATE.format(arg=arg, trigger=event_trigger)
#         dec = self.tokenizer(dec_text)
#         dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]
#         while len(dec_input_ids) < self.args.max_dec_seq_length:
#             dec_input_ids.append(self.tokenizer.pad_token_id)
#             dec_mask_ids.append(self.args.pad_mask_token)
#         matching_result = re.search(arg, dec_text)
#         char_idx_s, char_idx_e = matching_result.span()
#         char_idx_e -= 1
#         tok_prompt_s = dec.char_to_token(char_idx_s)
#         tok_prompt_e = dec.char_to_token(char_idx_e) + 1
#         return dec_input_ids, dec_mask_ids, tok_prompt_s, tok_prompt_e

#     def find_idx(self, target, list):
#         for i, item in enumerate(list):
#             if item == target:
#                 return i

#     # ── 新增：依存解析 + subword 对齐（专为本 processor 的数据格式定制）──
#     def _build_dep_fields(
#         self,
#         context: list,                    # 原始 token 列表（无 marker）
#         marked_context: list,             # 加了 <t-i> marker 的 token 列表
#         old_tok_to_new_tok_index: list,   # word_i → [new_tok_s, new_tok_e]
#         enc: object,                      # tokenizer 编码结果（带 offset_mapping）
#         seq_len: int,                     # all_ids 的有效长度（含 prompt 前的部分）
#     ):
#         """
#         对原始 context（无 marker）做依存解析，
#         再通过 old_tok_to_new_tok_index 映射到 marked_context 的 subword 粒度。

#         old_tok_to_new_tok_index[i] = [new_tok_s, new_tok_e]，对应原始第 i 个词
#         （已跳过 EXTERNAL_TOKENS，与 context 等长）。

#         Returns:
#             dep_heads : list[int]  长度 = seq_len
#             dep_rels  : list[str]  长度 = seq_len
#         """
#         # 默认：每个 subword 自指，关系为 'none'
#         dep_heads = list(range(seq_len))
#         dep_rels  = ['none'] * seq_len

#         # 用原始 context（无 marker）做解析，避免 <t-0> 等特殊符号干扰 spacy
#         plain_text = " ".join(context)
#         try:
#             doc = self.syntax_provider.nlp(plain_text)
#         except Exception as e:
#             import logging
#             logging.getLogger(__name__).warning(f"[dep_parse] spacy 解析失败: {e}")
#             return dep_heads, dep_rels

#         # 构建 char_start → word_idx 的反查表（基于原始 context）
#         char_start_to_word_idx = {}
#         char_pos = 0
#         for w_i, tok in enumerate(context):
#             char_start_to_word_idx[char_pos] = w_i
#             char_pos += len(tok) + 1   # +1 for space

#         # word_i → (head_word_i, deprel)
#         word_dep = {}
#         for spacy_tok in doc:
#             w_i = char_start_to_word_idx.get(spacy_tok.idx, None)
#             if w_i is None:
#                 continue
#             head_w_i = char_start_to_word_idx.get(spacy_tok.head.idx, w_i)
#             word_dep[w_i] = (head_w_i, spacy_tok.dep_)

#         # 映射到 subword 层
#         # old_tok_to_new_tok_index[w_i] = [new_tok_s, new_tok_e]
#         for w_i, (head_w_i, rel) in word_dep.items():
#             if w_i >= len(old_tok_to_new_tok_index):
#                 continue

#             sw_range   = old_tok_to_new_tok_index[w_i]       # [s, e)
#             sw_s, sw_e = sw_range[0], sw_range[1]

#             # head 的第一个 subword
#             if head_w_i < len(old_tok_to_new_tok_index):
#                 head_sw = old_tok_to_new_tok_index[head_w_i][0]
#             else:
#                 head_sw = sw_s   # fallback 自指

#             # 当前 word 的所有 subword 共享同一 head/rel
#             for sw in range(sw_s, min(sw_e, seq_len)):
#                 dep_heads[sw] = min(head_sw, seq_len - 1)
#                 dep_rels[sw]  = rel

#         return dep_heads, dep_rels

#     def convert_examples_to_features(self, examples, role_name_mapping=None):
#         features = []
#         if self.prompt_query:
#             prompts = self._read_prompt_group(self.args.prompt_path)

#         if os.environ.get("DEBUG", False): counter = [0, 0, 0]
#         over_nums = 0

#         for example in examples:
#             example_id         = example.doc_id
#             context            = example.context           # 原始 token 列表
#             event_type_2_events = example.event_type_2_events

#             list_event_type = []
#             triggers = []
#             for event_type, events in event_type_2_events.items():
#                 list_event_type += [e['event_type'] for e in events]
#                 triggers        += [tuple(e['trigger']) for e in events]

#             set_triggers = sorted(list(set(triggers)))

#             trigger_overlap = False
#             for t1 in set_triggers:
#                 for t2 in set_triggers:
#                     if t1[0] == t2[0] and t1[1] == t2[1]:
#                         continue
#                     if (t1[0] < t2[1] and t2[0] < t1[1]) or (t2[0] < t1[1] and t1[0] < t2[1]):
#                         trigger_overlap = True
#                         break
#             if trigger_overlap:
#                 print('[trigger_overlap]', event_type_2_events)
#                 exit(0)

#             # ── 构建 marked_context（与原逻辑完全一致）──────────────────
#             offset = 0
#             marked_context = deepcopy(context)
#             marker_indice  = list(range(len(triggers)))
#             for i, t in enumerate(set_triggers):
#                 t_start = t[0]; t_end = t[1]
#                 marked_context = (
#                     marked_context[:(t_start + offset)]
#                     + ['<t-%d>' % marker_indice[i]]
#                     + context[t_start: t_end]
#                     + ['</t-%d>' % marker_indice[i]]
#                     + context[t_end:]
#                 )
#                 offset += 2
#             enc_text = " ".join(marked_context)

#             # ── old_tok_to_new_tok_index（与原逻辑完全一致）──────────────
#             old_tok_to_char_index    = []
#             old_tok_to_new_tok_index = []

#             curr = 0
#             enc  = self.tokenizer(enc_text, add_special_tokens=True)
#             trigger_list = [[] for _ in range(len(triggers))]
#             for tok in marked_context:
#                 if tok not in EXTERNAL_TOKENS:
#                     old_tok_to_char_index.append([curr, curr + len(tok) - 1])
#                 curr += len(tok) + 1

#             enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
#             if len(enc_input_ids) > self.args.max_enc_seq_length:
#                 raise ValueError(f"Please increase max_enc_seq_length above {len(enc_input_ids)}")

#             all_ids      = enc_input_ids.copy()
#             all_mask_ids = enc_mask_ids.copy()
#             type_ids     = enc_mask_ids.copy()

#             offset_prompt = len(enc_input_ids)

#             while len(enc_input_ids) < self.args.max_enc_seq_length:
#                 enc_input_ids.append(self.tokenizer.pad_token_id)
#                 enc_mask_ids.append(self.args.pad_mask_token)

#             for old_tok_idx, (char_idx_s, char_idx_e) in enumerate(old_tok_to_char_index):
#                 new_tok_s = enc.char_to_token(char_idx_s)
#                 new_tok_e = enc.char_to_token(char_idx_e) + 1
#                 old_tok_to_new_tok_index.append([new_tok_s, new_tok_e])

#             trigger_enc_token_index = []
#             for t in triggers:
#                 new_t_start = old_tok_to_new_tok_index[t[0]][0]
#                 new_t_end   = old_tok_to_new_tok_index[t[1] - 1][1]
#                 trigger_enc_token_index.append([new_t_start, new_t_end])
#             for ii, it in enumerate(trigger_enc_token_index):
#                 type_ids[it[0] - 1] = ii + 2

#             dec_table_ids  = []
#             dec_table_mask = []

#             list_arg_2_prompt_slots    = []
#             list_num_prompt_slots      = []
#             list_dec_prompt_ids        = []
#             list_arg_2_prompt_slot_spans = []
#             offset_prompt_ = 0
#             kk = 0
#             enc_attention_mask = torch.zeros(
#                 (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
#                 dtype=torch.float32
#             )

#             for i, event_type in enumerate(event_type_2_events):
#                 events     = event_type_2_events[event_type]
#                 event_name = event_type.split('.')
#                 event_name = ['<e-%d>' % (i)] + event_name + ['</e-%d>' % (i)]
#                 for event in events:
#                     enc_trigger_start = trigger_enc_token_index[kk][0] - 1
#                     enc_trigger_end   = trigger_enc_token_index[kk][1] + 1
#                     kk += 1
#                     dec_prompt_text = prompts[event_type].strip()
#                     assert dec_prompt_text
#                     dec_prompt_text = ' '.join(event_name) + ' ' + dec_prompt_text
#                     dec_prompt      = self.tokenizer(dec_prompt_text, add_special_tokens=True)
#                     dec_prompt_ids_, dec_prompt_mask_ids_ = dec_prompt["input_ids"], dec_prompt["attention_mask"]

#                     arg_list = self.argument_dict[event_type.replace(':', '.')]
#                     arg_2_prompt_slots      = dict()
#                     arg_2_prompt_slot_spans = dict()
#                     num_prompt_slots = 0
#                     if os.environ.get("DEBUG", False): arg_set = set()
#                     for arg in arg_list:
#                         prompt_slots      = {"tok_s": [], "tok_e": [], "tok_s_off": [], "tok_e_off": []}
#                         prompt_slot_spans = []
#                         if role_name_mapping is not None:
#                             arg_ = role_name_mapping[event_type][arg]
#                         else:
#                             arg_ = arg
#                         for matching_result in re.finditer(
#                             r'\b' + re.escape(arg_) + r'\b',
#                             dec_prompt_text.split('.')[0]
#                         ):
#                             char_idx_s, char_idx_e = matching_result.span()
#                             char_idx_e -= 1
#                             tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
#                             tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
#                             prompt_slot_spans.append((tok_prompt_s, tok_prompt_e))
#                             prompt_slots["tok_s"].append(tok_prompt_s + offset_prompt_)
#                             prompt_slots["tok_e"].append(tok_prompt_e + offset_prompt_)
#                             prompt_slots["tok_s_off"].append(tok_prompt_s + offset_prompt + offset_prompt_)
#                             prompt_slots["tok_e_off"].append(tok_prompt_e + offset_prompt + offset_prompt_)
#                             num_prompt_slots += 1
#                         arg_2_prompt_slots[arg]      = prompt_slots
#                         arg_2_prompt_slot_spans[arg] = prompt_slot_spans

#                     list_arg_2_prompt_slots.append(arg_2_prompt_slots)
#                     list_num_prompt_slots.append(num_prompt_slots)
#                     list_dec_prompt_ids.append(dec_prompt_ids_)
#                     list_arg_2_prompt_slot_spans.append(arg_2_prompt_slot_spans)

#                     enc_attention_mask[0, enc_trigger_start:enc_trigger_end,
#                         offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_)] = 1
#                     enc_attention_mask[0,
#                         offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                         enc_trigger_start:enc_trigger_end] = 1
#                     enc_attention_mask[1, enc_trigger_start:enc_trigger_end,
#                         offset_prompt:offset_prompt + offset_prompt_] = 1
#                     enc_attention_mask[1, enc_trigger_start:enc_trigger_end,
#                         offset_prompt + offset_prompt_ + len(dec_prompt_ids_):] = 1
#                     enc_attention_mask[1, offset_prompt:offset_prompt + offset_prompt_,
#                         enc_trigger_start:enc_trigger_end] = 1
#                     enc_attention_mask[1, offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
#                         enc_trigger_start:enc_trigger_end] = 1

#                 enc_attention_mask[0,
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_)] = 1
#                 enc_attention_mask[1,
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                     offset_prompt:offset_prompt + offset_prompt_] = 1
#                 enc_attention_mask[1,
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                     offset_prompt + offset_prompt_ + len(dec_prompt_ids_):] = 1
#                 enc_attention_mask[1, offset_prompt:offset_prompt + offset_prompt_,
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_)] = 1
#                 enc_attention_mask[1, offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_)] = 1

#                 offset_prompt_ += len(dec_prompt_ids_)
#                 dec_table_ids  += dec_prompt_ids_
#                 dec_table_mask += dec_prompt_mask_ids_

#             all_ids.extend(dec_table_ids)
#             all_mask_ids.extend(dec_table_mask)
#             if len(all_ids) > self.args.max_enc_seq_length:
#                 over_nums += 1

#             while len(all_ids) < self.args.max_enc_seq_length:
#                 all_ids.append(self.tokenizer.pad_token_id)
#                 all_mask_ids.append(self.args.pad_mask_token)

#             # ════════════════════════════════════════════════════════════
#             # ▶▶ 新增①②：依存解析
#             #    此处 enc_text / old_tok_to_new_tok_index 已全部就绪
#             #    seq_len 取 all_ids padding 前的实际长度（offset_prompt + prompt长度）
#             #    以覆盖 context + prompt 的完整 subword 序列
#             # ════════════════════════════════════════════════════════════
#             actual_seq_len = offset_prompt + offset_prompt_   # context + 所有 prompt 的实际 token 数
#             actual_seq_len = min(actual_seq_len, self.args.max_enc_seq_length)

#             dep_heads, dep_rels = self._build_dep_fields(
#                 context=context,                            # 原始无 marker token 列表
#                 marked_context=marked_context,              # 含 marker 的列表（备用）
#                 old_tok_to_new_tok_index=old_tok_to_new_tok_index,  # word→[sw_s, sw_e]
#                 enc=enc,                                    # tokenizer 编码对象
#                 seq_len=actual_seq_len,
#             )

#             # ════════════════════════════════════════════════════════════
#             # ▶▶ 新增③：意群分组
#             #    遍历本样本所有事件，每个事件对应一个 group id
#             #    顺序与 list_event_type / trigger_enc_token_index 保持一致
#             # ════════════════════════════════════════════════════════════
#             event_groups = [_get_event_group(etype) for etype in list_event_type]

#             # ── 处理 target arguments（与原逻辑完全一致）────────────────
#             row_index       = 0
#             list_trigger_pos = []
#             list_arg_slots   = []
#             list_target_info = []
#             list_roles       = []
#             k = 0
#             for i, (event_type, events) in enumerate(event_type_2_events.items()):
#                 for event in events:
#                     arg_2_prompt_slots = list_arg_2_prompt_slots[k]
#                     num_prompt_slots   = list_num_prompt_slots[k]
#                     dec_prompt_ids_    = list_dec_prompt_ids[k]
#                     k += 1
#                     row_index += 1

#                     list_trigger_pos.append(len(dec_table_ids))
#                     arg_slots = []
#                     cursor    = len(dec_table_ids) + 1
#                     event_args      = event['args']
#                     event_args_name = [arg[-1] for arg in event_args]
#                     target_info     = dict()

#                     for arg, prompt_slots in arg_2_prompt_slots.items():
#                         num_slots = len(prompt_slots['tok_s'])
#                         arg_slots.append([cursor + x for x in range(num_slots)])
#                         cursor += num_slots

#                         arg_target = {"text": [], "span_s": [], "span_e": []}
#                         if arg in event_args_name:
#                             if os.environ.get("DEBUG", False): counter[0] += 1
#                             arg_idxs = [j for j, x in enumerate(event_args_name) if x == arg]
#                             if os.environ.get("DEBUG", False): counter[1] += len(arg_idxs)
#                             for arg_idx in arg_idxs:
#                                 event_arg_info = event_args[arg_idx]
#                                 answer_text    = event_arg_info[2]
#                                 start_old, end_old = event_arg_info[0], event_arg_info[1]
#                                 start_position = old_tok_to_new_tok_index[start_old][0]
#                                 end_position   = old_tok_to_new_tok_index[end_old - 1][1]
#                                 arg_target["text"].append(answer_text)
#                                 arg_target["span_s"].append(start_position)
#                                 arg_target["span_e"].append(end_position)

#                         target_info[arg] = arg_target

#                     assert sum([len(slots) for slots in arg_slots]) == num_prompt_slots
#                     list_arg_slots.append(arg_slots)
#                     list_target_info.append(target_info)
#                     roles = self.argument_dict[event_type.replace(':', '.')]
#                     assert len(roles) == len(arg_slots)
#                     list_roles.append(roles)

#             max_dec_seq_len = self.args.max_prompt_seq_length
#             while len(dec_table_ids) < max_dec_seq_len:
#                 dec_table_ids.append(self.tokenizer.pad_token_id)
#                 dec_table_mask.append(self.args.pad_mask_token)

#             if len(all_ids) > self.args.max_enc_seq_length:
#                 enc_attention_mask = torch.zeros(
#                     (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
#                     dtype=torch.float32
#                 )
#             if len(all_ids) > self.args.max_enc_seq_length:
#                 all_ids      = all_ids[:self.args.max_enc_seq_length]
#                 all_mask_ids = all_mask_ids[:self.args.max_enc_seq_length]
#             if len(list_arg_2_prompt_slots) == 1:
#                 enc_attention_mask = torch.zeros(
#                     (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
#                     dtype=torch.float32
#                 )

#             feature_idx = len(features)
#             features.append(
#                 InputFeatures(
#                     example_id, feature_idx,
#                     list_event_type, trigger_enc_token_index,
#                     enc_text, enc_input_ids, enc_mask_ids, all_ids, all_mask_ids,
#                     dec_prompt_text, dec_table_ids, dec_table_mask, None,
#                     list_arg_2_prompt_slots, list_target_info, enc_attention_mask,
#                     old_tok_to_new_tok_index=old_tok_to_new_tok_index,
#                     full_text=example.context,
#                     arg_list=list_roles,
#                     # ── 新增三个字段 ──────────────────────────────────
#                     dep_heads=dep_heads,
#                     dep_rels=dep_rels,
#                     event_groups=event_groups,
#                 )
#             )

#         print(over_nums)
#         if os.environ.get("DEBUG", False):
#             print('\033[91m' + f"distinct/tot arg_role: {counter[0]}/{counter[1]} ({counter[2]})" + '\033[0m')
#         return features

#     def convert_features_to_dataset(self, features):
#         dataset = ArgumentExtractionDataset(features)
#         return dataset

# #版本1采用意群来构建事件之间的关联

# import os
# import re
# import sys

# sys.path.append("../")
# import torch
# import numpy as np
# from copy import deepcopy
# from torch.utils.data import Dataset
# from processors.processor_base import DSET_processor, SyntaxProvider
# from utils import EXTERNAL_TOKENS, _PREDEFINED_QUERY_TEMPLATE


# class InputFeatures(object):
#     """A single set of features of data."""

#     def __init__(self, example_id, feature_id,
#                  event_type, event_trigger,
#                  enc_text, enc_input_ids, enc_mask_ids, all_ids, all_mask_ids,
#                  dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
#                  arg_quries, arg_joint_prompt, target_info, enc_attention_mask,
#                  old_tok_to_new_tok_index=None, full_text=None, arg_list=None,
#                  # ── 新增三个字段 ─────────────────────────────────────────
#                  dep_heads=None,
#                  dep_rels=None,
#                  event_groups=None,
#                  ):

#         self.example_id = example_id
#         self.feature_id = feature_id
#         self.event_type = event_type
#         self.event_trigger = event_trigger
#         self.num_events = len(event_trigger)

#         self.enc_text = enc_text
#         self.enc_input_ids = enc_input_ids
#         self.enc_mask_ids = enc_mask_ids

#         self.all_ids = all_ids
#         self.all_mask_ids = all_mask_ids
#         self.enc_attention_mask = enc_attention_mask

#         self.dec_prompt_texts = dec_prompt_text
#         self.dec_prompt_ids = dec_prompt_ids
#         self.dec_prompt_mask_ids = dec_prompt_mask_ids

#         if arg_quries is not None:
#             self.dec_arg_query_ids = [v[0] for k, v in arg_quries.items()]
#             self.dec_arg_query_masks = [v[1] for k, v in arg_quries.items()]
#             self.dec_arg_start_positions = [v[2] for k, v in arg_quries.items()]
#             self.dec_arg_end_positions = [v[3] for k, v in arg_quries.items()]
#             self.start_position_ids = [v['span_s'] for k, v in target_info.items()]
#             self.end_position_ids = [v['span_e'] for k, v in target_info.items()]
#         else:
#             self.dec_arg_query_ids = None
#             self.dec_arg_query_masks = None

#         self.arg_joint_prompt = arg_joint_prompt
#         self.target_info = target_info
#         self.old_tok_to_new_tok_index = old_tok_to_new_tok_index
#         self.full_text = full_text
#         self.arg_list = arg_list

#         # ── 新增字段赋值 ──────────────────────────────────────────────
#         # dep_heads / dep_rels：subword 粒度，长度 = len(all_ids)（含 prompt）
#         # event_groups：每个事件对应的意群 group id，长度 = num_events
#         self.dep_heads   = dep_heads   if dep_heads   is not None else []
#         self.dep_rels    = dep_rels    if dep_rels    is not None else []
#         self.event_groups = event_groups if event_groups is not None else []

#     def find_idx(self, target, list):
#         for i, item in enumerate(list):
#             if item == target:
#                 return i

#     def init_pred(self):
#         self.pred_dict_tok = [dict() for _ in range(self.num_events)]
#         self.pred_dict_word = [dict() for _ in range(self.num_events)]

#     def add_pred(self, role, span, event_index):
#         pred_dict_tok = self.pred_dict_tok[event_index]
#         pred_dict_word = self.pred_dict_word[event_index]
#         if role not in pred_dict_tok:
#             pred_dict_tok[role] = list()
#         if span not in pred_dict_tok[role]:
#             pred_dict_tok[role].append(span)
#             if span != (0, 0):
#                 if role not in pred_dict_word:
#                     pred_dict_word[role] = list()
#                 word_span = self.get_word_span(span)
#                 if word_span not in pred_dict_word[role]:
#                     pred_dict_word[role].append(word_span)

#     def set_gt(self):
#         self.gt_dict_tok = [dict() for _ in range(self.num_events)]
#         for i, target_info in enumerate(self.target_info):
#             for k, v in target_info.items():
#                 self.gt_dict_tok[i][k] = [(s, e) for (s, e) in zip(v["span_s"], v["span_e"])]

#         self.gt_dict_word = [dict() for _ in range(self.num_events)]
#         for i, gt_dict_tok in enumerate(self.gt_dict_tok):
#             gt_dict_word = self.gt_dict_word[i]
#             for role, spans in gt_dict_tok.items():
#                 for span in spans:
#                     if span != (0, 0):
#                         if role not in gt_dict_word:
#                             gt_dict_word[role] = list()
#                         word_span = self.get_word_span(span)
#                         gt_dict_word[role].append(word_span)

#     @property
#     def old_tok_index(self):
#         new_tok_index_to_old_tok_index = dict()
#         for old_tok_id, (new_tok_id_s, new_tok_id_e) in enumerate(self.old_tok_to_new_tok_index):
#             for j in range(new_tok_id_s, new_tok_id_e):
#                 new_tok_index_to_old_tok_index[j] = old_tok_id
#         return new_tok_index_to_old_tok_index

#     def get_word_span(self, span):
#         if span == (0, 0):
#             raise AssertionError()
#         offset = 0
#         span = list(span)
#         span[0] = min(span[0], max(self.old_tok_index.keys()))
#         span[1] = max(span[1] - 1, min(self.old_tok_index.keys()))
#         while span[0] not in self.old_tok_index:
#             span[0] += 1
#         span_s = self.old_tok_index[span[0]] + offset
#         while span[1] not in self.old_tok_index:
#             span[1] -= 1
#         span_e = self.old_tok_index[span[1]] + offset
#         while span_e < span_s:
#             span_e += 1
#         return (span_s, span_e)

#     def __repr__(self):
#         s = ""
#         s += "example_id: {}\n".format(self.example_id)
#         s += "event_type: {}\n".format(self.event_type)
#         s += "trigger_word: {}\n".format(self.event_trigger)
#         s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
#         s += "enc_input_ids: {}\n".format(self.enc_input_ids)
#         s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
#         s += "dec_prompt_ids: {}\n".format(self.dec_prompt_ids)
#         s += "dec_prompt_mask_ids: {}\n".format(self.dec_prompt_mask_ids)
#         s += "event_groups: {}\n".format(self.event_groups)
#         s += "dep_heads[:10]: {}\n".format(self.dep_heads[:10])
#         s += "dep_rels[:10]: {}\n".format(self.dep_rels[:10])
#         return s


# class ArgumentExtractionDataset(Dataset):
#     def __init__(self, features):
#         self.features = features

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx]

#     @staticmethod
#     def collate_fn(batch):
#         enc_input_ids = torch.tensor([f.enc_input_ids for f in batch])
#         enc_mask_ids = torch.tensor([f.enc_mask_ids for f in batch])

#         all_ids = torch.tensor([f.all_ids for f in batch])
#         all_mask_ids = torch.tensor([f.all_mask_ids for f in batch])

#         if batch[0].dec_prompt_ids is not None:
#             dec_prompt_ids = torch.tensor([f.dec_prompt_ids for f in batch])
#             dec_prompt_mask_ids = torch.tensor([f.dec_prompt_mask_ids for f in batch])
#         else:
#             dec_prompt_ids = None
#             dec_prompt_mask_ids = None

#         example_idx = [f.example_id for f in batch]
#         feature_idx = torch.tensor([f.feature_id for f in batch])

#         if batch[0].dec_arg_query_ids is not None:
#             dec_arg_query_ids = [torch.LongTensor(f.dec_arg_query_ids) for f in batch]
#             dec_arg_query_mask_ids = [torch.LongTensor(f.dec_arg_query_masks) for f in batch]
#             dec_arg_start_positions = [torch.LongTensor(f.dec_arg_start_positions) for f in batch]
#             dec_arg_end_positions = [torch.LongTensor(f.dec_arg_end_positions) for f in batch]
#             start_position_ids = [torch.FloatTensor(f.start_position_ids) for f in batch]
#             end_position_ids = [torch.FloatTensor(f.end_position_ids) for f in batch]
#         else:
#             dec_arg_query_ids = None
#             dec_arg_query_mask_ids = None
#             dec_arg_start_positions = None
#             dec_arg_end_positions = None
#             start_position_ids = None
#             end_position_ids = None

#         target_info = [f.target_info for f in batch]
#         old_tok_to_new_tok_index = [f.old_tok_to_new_tok_index for f in batch]
#         arg_joint_prompt = [f.arg_joint_prompt for f in batch]
#         arg_lists = [f.arg_list for f in batch]
#         event_trigger = [f.event_trigger for f in batch]
#         enc_attention_mask = [f.enc_attention_mask for f in batch]

#         # ── 新增：三个图字段直接以 list 传出，无需 padding ──────────────
#         dep_heads_batch    = [f.dep_heads    for f in batch]   # list[list[int]]
#         dep_rels_batch     = [f.dep_rels     for f in batch]   # list[list[str]]
#         event_groups_batch = [f.event_groups for f in batch]   # list[list[int]]

#         return enc_input_ids, enc_mask_ids, all_ids, all_mask_ids, \
#                dec_arg_query_ids, dec_arg_query_mask_ids, \
#                dec_prompt_ids, dec_prompt_mask_ids, \
#                target_info, old_tok_to_new_tok_index, arg_joint_prompt, arg_lists, \
#                example_idx, feature_idx, \
#                dec_arg_start_positions, dec_arg_end_positions, \
#                start_position_ids, end_position_ids, event_trigger, enc_attention_mask, \
#                dep_heads_batch, dep_rels_batch, event_groups_batch   # ← 新增


# class MultiargProcessor(DSET_processor):
#     def __init__(self, args, tokenizer):
#         super().__init__(args, tokenizer)
#         self.set_dec_input()
#         self.collate_fn = ArgumentExtractionDataset.collate_fn

#         # ── 新增：句法解析器（全局单例，避免重复加载）────────────────────
#         # SyntaxProvider 已在 processor_base 中定义，此处直接复用父类实例
#         # 如果父类 __init__ 中没有初始化，在这里初始化一次即可
#         if not hasattr(self, 'syntax_provider'):
#             self.syntax_provider = SyntaxProvider()

#     def set_dec_input(self):
#         self.arg_query = False
#         self.prompt_query = False
#         if self.args.model_type == "base":
#             self.arg_query = True
#         elif "DyGMA" in self.args.model_type:
#             self.prompt_query = True
#         else:
#             raise NotImplementedError(f"Unexpected setting {self.args.model_type}")

#     @staticmethod
#     def _read_prompt_group(prompt_path):
#         with open(prompt_path) as f:
#             lines = f.readlines()
#         prompts = dict()
#         for line in lines:
#             if not line:
#                 continue
#             event_type, prompt = line.split(":")
#             prompts[event_type] = prompt
#         return prompts

#     def create_dec_qury(self, arg, event_trigger):
#         dec_text = _PREDEFINED_QUERY_TEMPLATE.format(arg=arg, trigger=event_trigger)
#         dec = self.tokenizer(dec_text)
#         dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]
#         while len(dec_input_ids) < self.args.max_dec_seq_length:
#             dec_input_ids.append(self.tokenizer.pad_token_id)
#             dec_mask_ids.append(self.args.pad_mask_token)
#         matching_result = re.search(arg, dec_text)
#         char_idx_s, char_idx_e = matching_result.span()
#         char_idx_e -= 1
#         tok_prompt_s = dec.char_to_token(char_idx_s)
#         tok_prompt_e = dec.char_to_token(char_idx_e) + 1
#         return dec_input_ids, dec_mask_ids, tok_prompt_s, tok_prompt_e

#     def find_idx(self, target, list):
#         for i, item in enumerate(list):
#             if item == target:
#                 return i

#     # ── 新增：依存解析 + subword 对齐（专为本 processor 的数据格式定制）──
#     def _build_dep_fields(
#         self,
#         context: list,                    # 原始 token 列表（无 marker）
#         marked_context: list,             # 加了 <t-i> marker 的 token 列表
#         old_tok_to_new_tok_index: list,   # word_i → [new_tok_s, new_tok_e]
#         enc: object,                      # tokenizer 编码结果（带 offset_mapping）
#         seq_len: int,                     # all_ids 的有效长度（含 prompt 前的部分）
#     ):
#         """
#         对原始 context（无 marker）做依存解析，
#         再通过 old_tok_to_new_tok_index 映射到 marked_context 的 subword 粒度。

#         old_tok_to_new_tok_index[i] = [new_tok_s, new_tok_e]，对应原始第 i 个词
#         （已跳过 EXTERNAL_TOKENS，与 context 等长）。

#         Returns:
#             dep_heads : list[int]  长度 = seq_len
#             dep_rels  : list[str]  长度 = seq_len
#         """
#         # 默认：每个 subword 自指，关系为 'none'
#         dep_heads = list(range(seq_len))
#         dep_rels  = ['none'] * seq_len

#         # 用原始 context（无 marker）做解析，避免 <t-0> 等特殊符号干扰 spacy
#         plain_text = " ".join(context)
#         try:
#             doc = self.syntax_provider.nlp(plain_text)
#         except Exception as e:
#             import logging
#             logging.getLogger(__name__).warning(f"[dep_parse] spacy 解析失败: {e}")
#             return dep_heads, dep_rels

#         # 构建 char_start → word_idx 的反查表（基于原始 context）
#         char_start_to_word_idx = {}
#         char_pos = 0
#         for w_i, tok in enumerate(context):
#             char_start_to_word_idx[char_pos] = w_i
#             char_pos += len(tok) + 1   # +1 for space

#         # word_i → (head_word_i, deprel)
#         word_dep = {}
#         for spacy_tok in doc:
#             w_i = char_start_to_word_idx.get(spacy_tok.idx, None)
#             if w_i is None:
#                 continue
#             head_w_i = char_start_to_word_idx.get(spacy_tok.head.idx, w_i)
#             word_dep[w_i] = (head_w_i, spacy_tok.dep_)

#         # 映射到 subword 层
#         # old_tok_to_new_tok_index[w_i] = [new_tok_s, new_tok_e]
#         for w_i, (head_w_i, rel) in word_dep.items():
#             if w_i >= len(old_tok_to_new_tok_index):
#                 continue

#             sw_range   = old_tok_to_new_tok_index[w_i]       # [s, e)
#             sw_s, sw_e = sw_range[0], sw_range[1]

#             # head 的第一个 subword
#             if head_w_i < len(old_tok_to_new_tok_index):
#                 head_sw = old_tok_to_new_tok_index[head_w_i][0]
#             else:
#                 head_sw = sw_s   # fallback 自指

#             # 当前 word 的所有 subword 共享同一 head/rel
#             for sw in range(sw_s, min(sw_e, seq_len)):
#                 dep_heads[sw] = min(head_sw, seq_len - 1)
#                 dep_rels[sw]  = rel

#         return dep_heads, dep_rels

#     def convert_examples_to_features(self, examples, role_name_mapping=None):
#         features = []
#         if self.prompt_query:
#             prompts = self._read_prompt_group(self.args.prompt_path)

#         if os.environ.get("DEBUG", False): counter = [0, 0, 0]
#         over_nums = 0

#         for example in examples:
#             example_id         = example.doc_id
#             context            = example.context           # 原始 token 列表
#             event_type_2_events = example.event_type_2_events

#             list_event_type = []
#             triggers = []
#             for event_type, events in event_type_2_events.items():
#                 list_event_type += [e['event_type'] for e in events]
#                 triggers        += [tuple(e['trigger']) for e in events]

#             set_triggers = sorted(list(set(triggers)))

#             trigger_overlap = False
#             for t1 in set_triggers:
#                 for t2 in set_triggers:
#                     if t1[0] == t2[0] and t1[1] == t2[1]:
#                         continue
#                     if (t1[0] < t2[1] and t2[0] < t1[1]) or (t2[0] < t1[1] and t1[0] < t2[1]):
#                         trigger_overlap = True
#                         break
#             if trigger_overlap:
#                 print('[trigger_overlap]', event_type_2_events)
#                 exit(0)

#             # ── 构建 marked_context（与原逻辑完全一致）──────────────────
#             offset = 0
#             marked_context = deepcopy(context)
#             marker_indice  = list(range(len(triggers)))
#             for i, t in enumerate(set_triggers):
#                 t_start = t[0]; t_end = t[1]
#                 marked_context = (
#                     marked_context[:(t_start + offset)]
#                     + ['<t-%d>' % marker_indice[i]]
#                     + context[t_start: t_end]
#                     + ['</t-%d>' % marker_indice[i]]
#                     + context[t_end:]
#                 )
#                 offset += 2
#             enc_text = " ".join(marked_context)

#             # ── old_tok_to_new_tok_index（与原逻辑完全一致）──────────────
#             old_tok_to_char_index    = []
#             old_tok_to_new_tok_index = []

#             curr = 0
#             enc  = self.tokenizer(enc_text, add_special_tokens=True)
#             trigger_list = [[] for _ in range(len(triggers))]
#             for tok in marked_context:
#                 if tok not in EXTERNAL_TOKENS:
#                     old_tok_to_char_index.append([curr, curr + len(tok) - 1])
#                 curr += len(tok) + 1

#             enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
#             if len(enc_input_ids) > self.args.max_enc_seq_length:
#                 raise ValueError(f"Please increase max_enc_seq_length above {len(enc_input_ids)}")

#             all_ids      = enc_input_ids.copy()
#             all_mask_ids = enc_mask_ids.copy()
#             type_ids     = enc_mask_ids.copy()

#             offset_prompt = len(enc_input_ids)

#             while len(enc_input_ids) < self.args.max_enc_seq_length:
#                 enc_input_ids.append(self.tokenizer.pad_token_id)
#                 enc_mask_ids.append(self.args.pad_mask_token)

#             for old_tok_idx, (char_idx_s, char_idx_e) in enumerate(old_tok_to_char_index):
#                 new_tok_s = enc.char_to_token(char_idx_s)
#                 new_tok_e = enc.char_to_token(char_idx_e) + 1
#                 old_tok_to_new_tok_index.append([new_tok_s, new_tok_e])

#             trigger_enc_token_index = []
#             for t in triggers:
#                 new_t_start = old_tok_to_new_tok_index[t[0]][0]
#                 new_t_end   = old_tok_to_new_tok_index[t[1] - 1][1]
#                 trigger_enc_token_index.append([new_t_start, new_t_end])
#             for ii, it in enumerate(trigger_enc_token_index):
#                 type_ids[it[0] - 1] = ii + 2

#             dec_table_ids  = []
#             dec_table_mask = []

#             list_arg_2_prompt_slots    = []
#             list_num_prompt_slots      = []
#             list_dec_prompt_ids        = []
#             list_arg_2_prompt_slot_spans = []
#             offset_prompt_ = 0
#             kk = 0
#             enc_attention_mask = torch.zeros(
#                 (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
#                 dtype=torch.float32
#             )

#             for i, event_type in enumerate(event_type_2_events):
#                 events     = event_type_2_events[event_type]
#                 event_name = event_type.split('.')
#                 event_name = ['<e-%d>' % (i)] + event_name + ['</e-%d>' % (i)]
#                 for event in events:
#                     enc_trigger_start = trigger_enc_token_index[kk][0] - 1
#                     enc_trigger_end   = trigger_enc_token_index[kk][1] + 1
#                     kk += 1
#                     dec_prompt_text = prompts[event_type].strip()
#                     assert dec_prompt_text
#                     dec_prompt_text = ' '.join(event_name) + ' ' + dec_prompt_text
#                     dec_prompt      = self.tokenizer(dec_prompt_text, add_special_tokens=True)
#                     dec_prompt_ids_, dec_prompt_mask_ids_ = dec_prompt["input_ids"], dec_prompt["attention_mask"]

#                     arg_list = self.argument_dict[event_type.replace(':', '.')]
#                     arg_2_prompt_slots      = dict()
#                     arg_2_prompt_slot_spans = dict()
#                     num_prompt_slots = 0
#                     if os.environ.get("DEBUG", False): arg_set = set()
#                     for arg in arg_list:
#                         prompt_slots      = {"tok_s": [], "tok_e": [], "tok_s_off": [], "tok_e_off": []}
#                         prompt_slot_spans = []
#                         if role_name_mapping is not None:
#                             arg_ = role_name_mapping[event_type][arg]
#                         else:
#                             arg_ = arg
#                         for matching_result in re.finditer(
#                             r'\b' + re.escape(arg_) + r'\b',
#                             dec_prompt_text.split('.')[0]
#                         ):
#                             char_idx_s, char_idx_e = matching_result.span()
#                             char_idx_e -= 1
#                             tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
#                             tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
#                             prompt_slot_spans.append((tok_prompt_s, tok_prompt_e))
#                             prompt_slots["tok_s"].append(tok_prompt_s + offset_prompt_)
#                             prompt_slots["tok_e"].append(tok_prompt_e + offset_prompt_)
#                             prompt_slots["tok_s_off"].append(tok_prompt_s + offset_prompt + offset_prompt_)
#                             prompt_slots["tok_e_off"].append(tok_prompt_e + offset_prompt + offset_prompt_)
#                             num_prompt_slots += 1
#                         arg_2_prompt_slots[arg]      = prompt_slots
#                         arg_2_prompt_slot_spans[arg] = prompt_slot_spans

#                     list_arg_2_prompt_slots.append(arg_2_prompt_slots)
#                     list_num_prompt_slots.append(num_prompt_slots)
#                     list_dec_prompt_ids.append(dec_prompt_ids_)
#                     list_arg_2_prompt_slot_spans.append(arg_2_prompt_slot_spans)

#                     enc_attention_mask[0, enc_trigger_start:enc_trigger_end,
#                         offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_)] = 1
#                     enc_attention_mask[0,
#                         offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                         enc_trigger_start:enc_trigger_end] = 1
#                     enc_attention_mask[1, enc_trigger_start:enc_trigger_end,
#                         offset_prompt:offset_prompt + offset_prompt_] = 1
#                     enc_attention_mask[1, enc_trigger_start:enc_trigger_end,
#                         offset_prompt + offset_prompt_ + len(dec_prompt_ids_):] = 1
#                     enc_attention_mask[1, offset_prompt:offset_prompt + offset_prompt_,
#                         enc_trigger_start:enc_trigger_end] = 1
#                     enc_attention_mask[1, offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
#                         enc_trigger_start:enc_trigger_end] = 1

#                 enc_attention_mask[0,
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_)] = 1
#                 enc_attention_mask[1,
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                     offset_prompt:offset_prompt + offset_prompt_] = 1
#                 enc_attention_mask[1,
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                     offset_prompt + offset_prompt_ + len(dec_prompt_ids_):] = 1
#                 enc_attention_mask[1, offset_prompt:offset_prompt + offset_prompt_,
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_)] = 1
#                 enc_attention_mask[1, offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_)] = 1

#                 offset_prompt_ += len(dec_prompt_ids_)
#                 dec_table_ids  += dec_prompt_ids_
#                 dec_table_mask += dec_prompt_mask_ids_

#             all_ids.extend(dec_table_ids)
#             all_mask_ids.extend(dec_table_mask)
#             if len(all_ids) > self.args.max_enc_seq_length:
#                 over_nums += 1

#             while len(all_ids) < self.args.max_enc_seq_length:
#                 all_ids.append(self.tokenizer.pad_token_id)
#                 all_mask_ids.append(self.args.pad_mask_token)

#             # ════════════════════════════════════════════════════════════
#             # ▶▶ 新增①②：依存解析
#             #    此处 enc_text / old_tok_to_new_tok_index 已全部就绪
#             #    seq_len 取 all_ids padding 前的实际长度（offset_prompt + prompt长度）
#             #    以覆盖 context + prompt 的完整 subword 序列
#             # ════════════════════════════════════════════════════════════
#             actual_seq_len = offset_prompt + offset_prompt_   # context + 所有 prompt 的实际 token 数
#             actual_seq_len = min(actual_seq_len, self.args.max_enc_seq_length)

#             dep_heads, dep_rels = self._build_dep_fields(
#                 context=context,                            # 原始无 marker token 列表
#                 marked_context=marked_context,              # 含 marker 的列表（备用）
#                 old_tok_to_new_tok_index=old_tok_to_new_tok_index,  # word→[sw_s, sw_e]
#                 enc=enc,                                    # tokenizer 编码对象
#                 seq_len=actual_seq_len,
#             )

#             # ════════════════════════════════════════════════════════════
#             # ▶▶ 新增③：意群分组
#             #    遍历本样本所有事件，每个事件对应一个 group id
#             #    顺序与 list_event_type / trigger_enc_token_index 保持一致
#             # ════════════════════════════════════════════════════════════
#             # ── 新增③：意群 group id，直接从数据集的 token_chunk_ids 读取 ──
#             # token_chunk_ids[trigger_start] 即为该触发词所在的意群 id
#             # triggers 列表与 list_event_type 保持相同顺序
#             token_chunk_ids = example.token_chunk_ids
#             event_groups = []
#             for trig in triggers:
#                 trig_start = trig[0]   # 触发词在原始 context 中的起始 word 索引
#                 if token_chunk_ids and trig_start < len(token_chunk_ids):
#                     event_groups.append(token_chunk_ids[trig_start])
#                 else:
#                     event_groups.append(0)   # 无意群信息时默认同组

#             # ── 处理 target arguments（与原逻辑完全一致）────────────────
#             row_index       = 0
#             list_trigger_pos = []
#             list_arg_slots   = []
#             list_target_info = []
#             list_roles       = []
#             k = 0
#             for i, (event_type, events) in enumerate(event_type_2_events.items()):
#                 for event in events:
#                     arg_2_prompt_slots = list_arg_2_prompt_slots[k]
#                     num_prompt_slots   = list_num_prompt_slots[k]
#                     dec_prompt_ids_    = list_dec_prompt_ids[k]
#                     k += 1
#                     row_index += 1

#                     list_trigger_pos.append(len(dec_table_ids))
#                     arg_slots = []
#                     cursor    = len(dec_table_ids) + 1
#                     event_args      = event['args']
#                     event_args_name = [arg[-1] for arg in event_args]
#                     target_info     = dict()

#                     for arg, prompt_slots in arg_2_prompt_slots.items():
#                         num_slots = len(prompt_slots['tok_s'])
#                         arg_slots.append([cursor + x for x in range(num_slots)])
#                         cursor += num_slots

#                         arg_target = {"text": [], "span_s": [], "span_e": []}
#                         if arg in event_args_name:
#                             if os.environ.get("DEBUG", False): counter[0] += 1
#                             arg_idxs = [j for j, x in enumerate(event_args_name) if x == arg]
#                             if os.environ.get("DEBUG", False): counter[1] += len(arg_idxs)
#                             for arg_idx in arg_idxs:
#                                 event_arg_info = event_args[arg_idx]
#                                 answer_text    = event_arg_info[2]
#                                 start_old, end_old = event_arg_info[0], event_arg_info[1]
#                                 start_position = old_tok_to_new_tok_index[start_old][0]
#                                 end_position   = old_tok_to_new_tok_index[end_old - 1][1]
#                                 arg_target["text"].append(answer_text)
#                                 arg_target["span_s"].append(start_position)
#                                 arg_target["span_e"].append(end_position)

#                         target_info[arg] = arg_target

#                     assert sum([len(slots) for slots in arg_slots]) == num_prompt_slots
#                     list_arg_slots.append(arg_slots)
#                     list_target_info.append(target_info)
#                     roles = self.argument_dict[event_type.replace(':', '.')]
#                     assert len(roles) == len(arg_slots)
#                     list_roles.append(roles)

#             max_dec_seq_len = self.args.max_prompt_seq_length
#             while len(dec_table_ids) < max_dec_seq_len:
#                 dec_table_ids.append(self.tokenizer.pad_token_id)
#                 dec_table_mask.append(self.args.pad_mask_token)

#             if len(all_ids) > self.args.max_enc_seq_length:
#                 enc_attention_mask = torch.zeros(
#                     (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
#                     dtype=torch.float32
#                 )
#             if len(all_ids) > self.args.max_enc_seq_length:
#                 all_ids      = all_ids[:self.args.max_enc_seq_length]
#                 all_mask_ids = all_mask_ids[:self.args.max_enc_seq_length]
#             if len(list_arg_2_prompt_slots) == 1:
#                 enc_attention_mask = torch.zeros(
#                     (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
#                     dtype=torch.float32
#                 )

#             feature_idx = len(features)
#             features.append(
#                 InputFeatures(
#                     example_id, feature_idx,
#                     list_event_type, trigger_enc_token_index,
#                     enc_text, enc_input_ids, enc_mask_ids, all_ids, all_mask_ids,
#                     dec_prompt_text, dec_table_ids, dec_table_mask, None,
#                     list_arg_2_prompt_slots, list_target_info, enc_attention_mask,
#                     old_tok_to_new_tok_index=old_tok_to_new_tok_index,
#                     full_text=example.context,
#                     arg_list=list_roles,
#                     # ── 新增三个字段 ──────────────────────────────────
#                     dep_heads=dep_heads,
#                     dep_rels=dep_rels,
#                     event_groups=event_groups,
#                 )
#             )

#         print(over_nums)
#         if os.environ.get("DEBUG", False):
#             print('\033[91m' + f"distinct/tot arg_role: {counter[0]}/{counter[1]} ({counter[2]})" + '\033[0m')
#         return features

#     def convert_features_to_dataset(self, features):
#         dataset = ArgumentExtractionDataset(features)
#         return dataset

# ##版本2：采用触发词相似度来构建事件之间的关联
# import os
# import re
# import sys

# sys.path.append("../")
# import torch
# import numpy as np
# from copy import deepcopy
# from torch.utils.data import Dataset
# from processors.processor_base import DSET_processor, SyntaxProvider
# from utils import EXTERNAL_TOKENS, _PREDEFINED_QUERY_TEMPLATE


# class InputFeatures(object):
#     """A single set of features of data."""

#     def __init__(self, example_id, feature_id,
#                  event_type, event_trigger,
#                  enc_text, enc_input_ids, enc_mask_ids, all_ids, all_mask_ids,
#                  dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
#                  arg_quries, arg_joint_prompt, target_info, enc_attention_mask,
#                  old_tok_to_new_tok_index=None, full_text=None, arg_list=None,
#                  # ── 新增三个字段 ─────────────────────────────────────────
#                  dep_heads=None,
#                  dep_rels=None,
#                  event_groups=None,
#                  ):

#         self.example_id = example_id
#         self.feature_id = feature_id
#         self.event_type = event_type
#         self.event_trigger = event_trigger
#         self.num_events = len(event_trigger)

#         self.enc_text = enc_text
#         self.enc_input_ids = enc_input_ids
#         self.enc_mask_ids = enc_mask_ids

#         self.all_ids = all_ids
#         self.all_mask_ids = all_mask_ids
#         self.enc_attention_mask = enc_attention_mask

#         self.dec_prompt_texts = dec_prompt_text
#         self.dec_prompt_ids = dec_prompt_ids
#         self.dec_prompt_mask_ids = dec_prompt_mask_ids

#         if arg_quries is not None:
#             self.dec_arg_query_ids = [v[0] for k, v in arg_quries.items()]
#             self.dec_arg_query_masks = [v[1] for k, v in arg_quries.items()]
#             self.dec_arg_start_positions = [v[2] for k, v in arg_quries.items()]
#             self.dec_arg_end_positions = [v[3] for k, v in arg_quries.items()]
#             self.start_position_ids = [v['span_s'] for k, v in target_info.items()]
#             self.end_position_ids = [v['span_e'] for k, v in target_info.items()]
#         else:
#             self.dec_arg_query_ids = None
#             self.dec_arg_query_masks = None

#         self.arg_joint_prompt = arg_joint_prompt
#         self.target_info = target_info
#         self.old_tok_to_new_tok_index = old_tok_to_new_tok_index
#         self.full_text = full_text
#         self.arg_list = arg_list

#         # ── 新增字段赋值 ──────────────────────────────────────────────
#         # dep_heads / dep_rels：subword 粒度，长度 = len(all_ids)（含 prompt）
#         # event_groups：每个事件对应的意群 group id，长度 = num_events
#         self.dep_heads   = dep_heads   if dep_heads   is not None else []
#         self.dep_rels    = dep_rels    if dep_rels    is not None else []
#         self.event_groups = event_groups if event_groups is not None else []

#     def find_idx(self, target, list):
#         for i, item in enumerate(list):
#             if item == target:
#                 return i

#     def init_pred(self):
#         self.pred_dict_tok = [dict() for _ in range(self.num_events)]
#         self.pred_dict_word = [dict() for _ in range(self.num_events)]

#     def add_pred(self, role, span, event_index):
#         pred_dict_tok = self.pred_dict_tok[event_index]
#         pred_dict_word = self.pred_dict_word[event_index]
#         if role not in pred_dict_tok:
#             pred_dict_tok[role] = list()
#         if span not in pred_dict_tok[role]:
#             pred_dict_tok[role].append(span)
#             if span != (0, 0):
#                 if role not in pred_dict_word:
#                     pred_dict_word[role] = list()
#                 word_span = self.get_word_span(span)
#                 if word_span not in pred_dict_word[role]:
#                     pred_dict_word[role].append(word_span)

#     def set_gt(self):
#         self.gt_dict_tok = [dict() for _ in range(self.num_events)]
#         for i, target_info in enumerate(self.target_info):
#             for k, v in target_info.items():
#                 self.gt_dict_tok[i][k] = [(s, e) for (s, e) in zip(v["span_s"], v["span_e"])]

#         self.gt_dict_word = [dict() for _ in range(self.num_events)]
#         for i, gt_dict_tok in enumerate(self.gt_dict_tok):
#             gt_dict_word = self.gt_dict_word[i]
#             for role, spans in gt_dict_tok.items():
#                 for span in spans:
#                     if span != (0, 0):
#                         if role not in gt_dict_word:
#                             gt_dict_word[role] = list()
#                         word_span = self.get_word_span(span)
#                         gt_dict_word[role].append(word_span)

#     @property
#     def old_tok_index(self):
#         new_tok_index_to_old_tok_index = dict()
#         for old_tok_id, (new_tok_id_s, new_tok_id_e) in enumerate(self.old_tok_to_new_tok_index):
#             for j in range(new_tok_id_s, new_tok_id_e):
#                 new_tok_index_to_old_tok_index[j] = old_tok_id
#         return new_tok_index_to_old_tok_index

#     def get_word_span(self, span):
#         if span == (0, 0):
#             raise AssertionError()
#         offset = 0
#         span = list(span)
#         span[0] = min(span[0], max(self.old_tok_index.keys()))
#         span[1] = max(span[1] - 1, min(self.old_tok_index.keys()))
#         while span[0] not in self.old_tok_index:
#             span[0] += 1
#         span_s = self.old_tok_index[span[0]] + offset
#         while span[1] not in self.old_tok_index:
#             span[1] -= 1
#         span_e = self.old_tok_index[span[1]] + offset
#         while span_e < span_s:
#             span_e += 1
#         return (span_s, span_e)

#     def __repr__(self):
#         s = ""
#         s += "example_id: {}\n".format(self.example_id)
#         s += "event_type: {}\n".format(self.event_type)
#         s += "trigger_word: {}\n".format(self.event_trigger)
#         s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
#         s += "enc_input_ids: {}\n".format(self.enc_input_ids)
#         s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
#         s += "dec_prompt_ids: {}\n".format(self.dec_prompt_ids)
#         s += "dec_prompt_mask_ids: {}\n".format(self.dec_prompt_mask_ids)
#         s += "event_groups: {}\n".format(self.event_groups)
#         s += "dep_heads[:10]: {}\n".format(self.dep_heads[:10])
#         s += "dep_rels[:10]: {}\n".format(self.dep_rels[:10])
#         return s


# class ArgumentExtractionDataset(Dataset):
#     def __init__(self, features):
#         self.features = features

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx]

#     @staticmethod
#     def collate_fn(batch):
#         enc_input_ids = torch.tensor([f.enc_input_ids for f in batch])
#         enc_mask_ids = torch.tensor([f.enc_mask_ids for f in batch])

#         all_ids = torch.tensor([f.all_ids for f in batch])
#         all_mask_ids = torch.tensor([f.all_mask_ids for f in batch])

#         if batch[0].dec_prompt_ids is not None:
#             dec_prompt_ids = torch.tensor([f.dec_prompt_ids for f in batch])
#             dec_prompt_mask_ids = torch.tensor([f.dec_prompt_mask_ids for f in batch])
#         else:
#             dec_prompt_ids = None
#             dec_prompt_mask_ids = None

#         example_idx = [f.example_id for f in batch]
#         feature_idx = torch.tensor([f.feature_id for f in batch])

#         if batch[0].dec_arg_query_ids is not None:
#             dec_arg_query_ids = [torch.LongTensor(f.dec_arg_query_ids) for f in batch]
#             dec_arg_query_mask_ids = [torch.LongTensor(f.dec_arg_query_masks) for f in batch]
#             dec_arg_start_positions = [torch.LongTensor(f.dec_arg_start_positions) for f in batch]
#             dec_arg_end_positions = [torch.LongTensor(f.dec_arg_end_positions) for f in batch]
#             start_position_ids = [torch.FloatTensor(f.start_position_ids) for f in batch]
#             end_position_ids = [torch.FloatTensor(f.end_position_ids) for f in batch]
#         else:
#             dec_arg_query_ids = None
#             dec_arg_query_mask_ids = None
#             dec_arg_start_positions = None
#             dec_arg_end_positions = None
#             start_position_ids = None
#             end_position_ids = None

#         target_info = [f.target_info for f in batch]
#         old_tok_to_new_tok_index = [f.old_tok_to_new_tok_index for f in batch]
#         arg_joint_prompt = [f.arg_joint_prompt for f in batch]
#         arg_lists = [f.arg_list for f in batch]
#         event_trigger = [f.event_trigger for f in batch]
#         enc_attention_mask = [f.enc_attention_mask for f in batch]
#         event_type_batch = [f.event_type for f in batch]   # list[list[str]] 每个样本的事件类型列表

#         # ── 新增：三个图字段直接以 list 传出，无需 padding ──────────────
#         dep_heads_batch    = [f.dep_heads    for f in batch]   # list[list[int]]
#         dep_rels_batch     = [f.dep_rels     for f in batch]   # list[list[str]]
#         event_groups_batch = [f.event_groups for f in batch]   # list[list[int]]

#         return enc_input_ids, enc_mask_ids, all_ids, all_mask_ids, \
#                dec_arg_query_ids, dec_arg_query_mask_ids, \
#                dec_prompt_ids, dec_prompt_mask_ids, \
#                target_info, old_tok_to_new_tok_index, arg_joint_prompt, arg_lists, \
#                example_idx, feature_idx, \
#                dec_arg_start_positions, dec_arg_end_positions, \
#                start_position_ids, end_position_ids, event_trigger, enc_attention_mask, \
#                dep_heads_batch, dep_rels_batch, event_groups_batch, \
#                event_type_batch   # ← 事件类型，用于触发词图相似度计算


# class MultiargProcessor(DSET_processor):
#     def __init__(self, args, tokenizer):
#         super().__init__(args, tokenizer)
#         self.set_dec_input()
#         self.collate_fn = ArgumentExtractionDataset.collate_fn

#         # ── 新增：句法解析器（全局单例，避免重复加载）────────────────────
#         # SyntaxProvider 已在 processor_base 中定义，此处直接复用父类实例
#         # 如果父类 __init__ 中没有初始化，在这里初始化一次即可
#         if not hasattr(self, 'syntax_provider'):
#             self.syntax_provider = SyntaxProvider()

#     def set_dec_input(self):
#         self.arg_query = False
#         self.prompt_query = False
#         if self.args.model_type == "base":
#             self.arg_query = True
#         elif "DyGMA" in self.args.model_type:
#             self.prompt_query = True
#         else:
#             raise NotImplementedError(f"Unexpected setting {self.args.model_type}")

#     @staticmethod
#     def _read_prompt_group(prompt_path):
#         with open(prompt_path) as f:
#             lines = f.readlines()
#         prompts = dict()
#         for line in lines:
#             if not line:
#                 continue
#             event_type, prompt = line.split(":")
#             prompts[event_type] = prompt
#         return prompts

#     def create_dec_qury(self, arg, event_trigger):
#         dec_text = _PREDEFINED_QUERY_TEMPLATE.format(arg=arg, trigger=event_trigger)
#         dec = self.tokenizer(dec_text)
#         dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]
#         while len(dec_input_ids) < self.args.max_dec_seq_length:
#             dec_input_ids.append(self.tokenizer.pad_token_id)
#             dec_mask_ids.append(self.args.pad_mask_token)
#         matching_result = re.search(arg, dec_text)
#         char_idx_s, char_idx_e = matching_result.span()
#         char_idx_e -= 1
#         tok_prompt_s = dec.char_to_token(char_idx_s)
#         tok_prompt_e = dec.char_to_token(char_idx_e) + 1
#         return dec_input_ids, dec_mask_ids, tok_prompt_s, tok_prompt_e

#     def find_idx(self, target, list):
#         for i, item in enumerate(list):
#             if item == target:
#                 return i

#     # ── 新增：依存解析 + subword 对齐（专为本 processor 的数据格式定制）──
#     def _build_dep_fields(
#         self,
#         context: list,                    # 原始 token 列表（无 marker）
#         marked_context: list,             # 加了 <t-i> marker 的 token 列表
#         old_tok_to_new_tok_index: list,   # word_i → [new_tok_s, new_tok_e]
#         enc: object,                      # tokenizer 编码结果（带 offset_mapping）
#         seq_len: int,                     # all_ids 的有效长度（含 prompt 前的部分）
#     ):
#         """
#         对原始 context（无 marker）做依存解析，
#         再通过 old_tok_to_new_tok_index 映射到 marked_context 的 subword 粒度。

#         old_tok_to_new_tok_index[i] = [new_tok_s, new_tok_e]，对应原始第 i 个词
#         （已跳过 EXTERNAL_TOKENS，与 context 等长）。

#         Returns:
#             dep_heads : list[int]  长度 = seq_len
#             dep_rels  : list[str]  长度 = seq_len
#         """
#         # 默认：每个 subword 自指，关系为 'none'
#         dep_heads = list(range(seq_len))
#         dep_rels  = ['none'] * seq_len

#         # 用原始 context（无 marker）做解析，避免 <t-0> 等特殊符号干扰 spacy
#         plain_text = " ".join(context)
#         try:
#             doc = self.syntax_provider.nlp(plain_text)
#         except Exception as e:
#             import logging
#             logging.getLogger(__name__).warning(f"[dep_parse] spacy 解析失败: {e}")
#             return dep_heads, dep_rels

#         # 构建 char_start → word_idx 的反查表（基于原始 context）
#         char_start_to_word_idx = {}
#         char_pos = 0
#         for w_i, tok in enumerate(context):
#             char_start_to_word_idx[char_pos] = w_i
#             char_pos += len(tok) + 1   # +1 for space

#         # word_i → (head_word_i, deprel)
#         word_dep = {}
#         for spacy_tok in doc:
#             w_i = char_start_to_word_idx.get(spacy_tok.idx, None)
#             if w_i is None:
#                 continue
#             head_w_i = char_start_to_word_idx.get(spacy_tok.head.idx, w_i)
#             word_dep[w_i] = (head_w_i, spacy_tok.dep_)

#         # 映射到 subword 层
#         # old_tok_to_new_tok_index[w_i] = [new_tok_s, new_tok_e]
#         for w_i, (head_w_i, rel) in word_dep.items():
#             if w_i >= len(old_tok_to_new_tok_index):
#                 continue

#             sw_range   = old_tok_to_new_tok_index[w_i]       # [s, e)
#             sw_s, sw_e = sw_range[0], sw_range[1]

#             # head 的第一个 subword
#             if head_w_i < len(old_tok_to_new_tok_index):
#                 head_sw = old_tok_to_new_tok_index[head_w_i][0]
#             else:
#                 head_sw = sw_s   # fallback 自指

#             # 当前 word 的所有 subword 共享同一 head/rel
#             for sw in range(sw_s, min(sw_e, seq_len)):
#                 dep_heads[sw] = min(head_sw, seq_len - 1)
#                 dep_rels[sw]  = rel

#         return dep_heads, dep_rels

#     def convert_examples_to_features(self, examples, role_name_mapping=None):
#         features = []
#         if self.prompt_query:
#             prompts = self._read_prompt_group(self.args.prompt_path)

#         if os.environ.get("DEBUG", False): counter = [0, 0, 0]
#         over_nums = 0

#         for example in examples:
#             example_id         = example.doc_id
#             context            = example.context           # 原始 token 列表
#             event_type_2_events = example.event_type_2_events

#             list_event_type = []
#             triggers = []
#             for event_type, events in event_type_2_events.items():
#                 list_event_type += [e['event_type'] for e in events]
#                 triggers        += [tuple(e['trigger']) for e in events]

#             set_triggers = sorted(list(set(triggers)))

#             trigger_overlap = False
#             for t1 in set_triggers:
#                 for t2 in set_triggers:
#                     if t1[0] == t2[0] and t1[1] == t2[1]:
#                         continue
#                     if (t1[0] < t2[1] and t2[0] < t1[1]) or (t2[0] < t1[1] and t1[0] < t2[1]):
#                         trigger_overlap = True
#                         break
#             if trigger_overlap:
#                 print('[trigger_overlap]', event_type_2_events)
#                 exit(0)

#             # ── 构建 marked_context（与原逻辑完全一致）──────────────────
#             offset = 0
#             marked_context = deepcopy(context)
#             marker_indice  = list(range(len(triggers)))
#             for i, t in enumerate(set_triggers):
#                 t_start = t[0]; t_end = t[1]
#                 marked_context = (
#                     marked_context[:(t_start + offset)]
#                     + ['<t-%d>' % marker_indice[i]]
#                     + context[t_start: t_end]
#                     + ['</t-%d>' % marker_indice[i]]
#                     + context[t_end:]
#                 )
#                 offset += 2
#             enc_text = " ".join(marked_context)

#             # ── old_tok_to_new_tok_index（与原逻辑完全一致）──────────────
#             old_tok_to_char_index    = []
#             old_tok_to_new_tok_index = []

#             curr = 0
#             enc  = self.tokenizer(enc_text, add_special_tokens=True)
#             trigger_list = [[] for _ in range(len(triggers))]
#             for tok in marked_context:
#                 if tok not in EXTERNAL_TOKENS:
#                     old_tok_to_char_index.append([curr, curr + len(tok) - 1])
#                 curr += len(tok) + 1

#             enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
#             if len(enc_input_ids) > self.args.max_enc_seq_length:
#                 raise ValueError(f"Please increase max_enc_seq_length above {len(enc_input_ids)}")

#             all_ids      = enc_input_ids.copy()
#             all_mask_ids = enc_mask_ids.copy()
#             type_ids     = enc_mask_ids.copy()

#             offset_prompt = len(enc_input_ids)

#             while len(enc_input_ids) < self.args.max_enc_seq_length:
#                 enc_input_ids.append(self.tokenizer.pad_token_id)
#                 enc_mask_ids.append(self.args.pad_mask_token)

#             for old_tok_idx, (char_idx_s, char_idx_e) in enumerate(old_tok_to_char_index):
#                 new_tok_s = enc.char_to_token(char_idx_s)
#                 new_tok_e = enc.char_to_token(char_idx_e) + 1
#                 old_tok_to_new_tok_index.append([new_tok_s, new_tok_e])

#             trigger_enc_token_index = []
#             for t in triggers:
#                 new_t_start = old_tok_to_new_tok_index[t[0]][0]
#                 new_t_end   = old_tok_to_new_tok_index[t[1] - 1][1]
#                 trigger_enc_token_index.append([new_t_start, new_t_end])
#             for ii, it in enumerate(trigger_enc_token_index):
#                 type_ids[it[0] - 1] = ii + 2

#             dec_table_ids  = []
#             dec_table_mask = []

#             list_arg_2_prompt_slots    = []
#             list_num_prompt_slots      = []
#             list_dec_prompt_ids        = []
#             list_arg_2_prompt_slot_spans = []
#             offset_prompt_ = 0
#             kk = 0
#             enc_attention_mask = torch.zeros(
#                 (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
#                 dtype=torch.float32
#             )

#             for i, event_type in enumerate(event_type_2_events):
#                 events     = event_type_2_events[event_type]
#                 event_name = event_type.split('.')
#                 event_name = ['<e-%d>' % (i)] + event_name + ['</e-%d>' % (i)]
#                 for event in events:
#                     enc_trigger_start = trigger_enc_token_index[kk][0] - 1
#                     enc_trigger_end   = trigger_enc_token_index[kk][1] + 1
#                     kk += 1
#                     dec_prompt_text = prompts[event_type].strip()
#                     assert dec_prompt_text
#                     dec_prompt_text = ' '.join(event_name) + ' ' + dec_prompt_text
#                     dec_prompt      = self.tokenizer(dec_prompt_text, add_special_tokens=True)
#                     dec_prompt_ids_, dec_prompt_mask_ids_ = dec_prompt["input_ids"], dec_prompt["attention_mask"]

#                     arg_list = self.argument_dict[event_type.replace(':', '.')]
#                     arg_2_prompt_slots      = dict()
#                     arg_2_prompt_slot_spans = dict()
#                     num_prompt_slots = 0
#                     if os.environ.get("DEBUG", False): arg_set = set()
#                     for arg in arg_list:
#                         prompt_slots      = {"tok_s": [], "tok_e": [], "tok_s_off": [], "tok_e_off": []}
#                         prompt_slot_spans = []
#                         if role_name_mapping is not None:
#                             arg_ = role_name_mapping[event_type][arg]
#                         else:
#                             arg_ = arg
#                         for matching_result in re.finditer(
#                             r'\b' + re.escape(arg_) + r'\b',
#                             dec_prompt_text.split('.')[0]
#                         ):
#                             char_idx_s, char_idx_e = matching_result.span()
#                             char_idx_e -= 1
#                             tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
#                             tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
#                             prompt_slot_spans.append((tok_prompt_s, tok_prompt_e))
#                             prompt_slots["tok_s"].append(tok_prompt_s + offset_prompt_)
#                             prompt_slots["tok_e"].append(tok_prompt_e + offset_prompt_)
#                             prompt_slots["tok_s_off"].append(tok_prompt_s + offset_prompt + offset_prompt_)
#                             prompt_slots["tok_e_off"].append(tok_prompt_e + offset_prompt + offset_prompt_)
#                             num_prompt_slots += 1
#                         arg_2_prompt_slots[arg]      = prompt_slots
#                         arg_2_prompt_slot_spans[arg] = prompt_slot_spans

#                     list_arg_2_prompt_slots.append(arg_2_prompt_slots)
#                     list_num_prompt_slots.append(num_prompt_slots)
#                     list_dec_prompt_ids.append(dec_prompt_ids_)
#                     list_arg_2_prompt_slot_spans.append(arg_2_prompt_slot_spans)

#                     enc_attention_mask[0, enc_trigger_start:enc_trigger_end,
#                         offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_)] = 1
#                     enc_attention_mask[0,
#                         offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                         enc_trigger_start:enc_trigger_end] = 1
#                     enc_attention_mask[1, enc_trigger_start:enc_trigger_end,
#                         offset_prompt:offset_prompt + offset_prompt_] = 1
#                     enc_attention_mask[1, enc_trigger_start:enc_trigger_end,
#                         offset_prompt + offset_prompt_ + len(dec_prompt_ids_):] = 1
#                     enc_attention_mask[1, offset_prompt:offset_prompt + offset_prompt_,
#                         enc_trigger_start:enc_trigger_end] = 1
#                     enc_attention_mask[1, offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
#                         enc_trigger_start:enc_trigger_end] = 1

#                 enc_attention_mask[0,
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_)] = 1
#                 enc_attention_mask[1,
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                     offset_prompt:offset_prompt + offset_prompt_] = 1
#                 enc_attention_mask[1,
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                     offset_prompt + offset_prompt_ + len(dec_prompt_ids_):] = 1
#                 enc_attention_mask[1, offset_prompt:offset_prompt + offset_prompt_,
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_)] = 1
#                 enc_attention_mask[1, offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
#                     offset_prompt + offset_prompt_:offset_prompt + offset_prompt_ + len(dec_prompt_ids_)] = 1

#                 offset_prompt_ += len(dec_prompt_ids_)
#                 dec_table_ids  += dec_prompt_ids_
#                 dec_table_mask += dec_prompt_mask_ids_

#             all_ids.extend(dec_table_ids)
#             all_mask_ids.extend(dec_table_mask)
#             if len(all_ids) > self.args.max_enc_seq_length:
#                 over_nums += 1

#             while len(all_ids) < self.args.max_enc_seq_length:
#                 all_ids.append(self.tokenizer.pad_token_id)
#                 all_mask_ids.append(self.args.pad_mask_token)

#             # ════════════════════════════════════════════════════════════
#             # ▶▶ 新增①②：依存解析
#             #    此处 enc_text / old_tok_to_new_tok_index 已全部就绪
#             #    seq_len 取 all_ids padding 前的实际长度（offset_prompt + prompt长度）
#             #    以覆盖 context + prompt 的完整 subword 序列
#             # ════════════════════════════════════════════════════════════
#             actual_seq_len = offset_prompt + offset_prompt_   # context + 所有 prompt 的实际 token 数
#             actual_seq_len = min(actual_seq_len, self.args.max_enc_seq_length)

#             dep_heads, dep_rels = self._build_dep_fields(
#                 context=context,                            # 原始无 marker token 列表
#                 marked_context=marked_context,              # 含 marker 的列表（备用）
#                 old_tok_to_new_tok_index=old_tok_to_new_tok_index,  # word→[sw_s, sw_e]
#                 enc=enc,                                    # tokenizer 编码对象
#                 seq_len=actual_seq_len,
#             )

#             # ════════════════════════════════════════════════════════════
#             # ▶▶ 新增③：意群分组
#             #    遍历本样本所有事件，每个事件对应一个 group id
#             #    顺序与 list_event_type / trigger_enc_token_index 保持一致
#             # ════════════════════════════════════════════════════════════
#             # ── 新增③：意群 group id，直接从数据集的 token_chunk_ids 读取 ──
#             # token_chunk_ids[trigger_start] 即为该触发词所在的意群 id
#             # triggers 列表与 list_event_type 保持相同顺序
#             token_chunk_ids = example.token_chunk_ids
#             event_groups = []
#             for trig in triggers:
#                 trig_start = trig[0]   # 触发词在原始 context 中的起始 word 索引
#                 if token_chunk_ids and trig_start < len(token_chunk_ids):
#                     event_groups.append(token_chunk_ids[trig_start])
#                 else:
#                     event_groups.append(0)   # 无意群信息时默认同组

#             # ── 处理 target arguments（与原逻辑完全一致）────────────────
#             row_index       = 0
#             list_trigger_pos = []
#             list_arg_slots   = []
#             list_target_info = []
#             list_roles       = []
#             k = 0
#             for i, (event_type, events) in enumerate(event_type_2_events.items()):
#                 for event in events:
#                     arg_2_prompt_slots = list_arg_2_prompt_slots[k]
#                     num_prompt_slots   = list_num_prompt_slots[k]
#                     dec_prompt_ids_    = list_dec_prompt_ids[k]
#                     k += 1
#                     row_index += 1

#                     list_trigger_pos.append(len(dec_table_ids))
#                     arg_slots = []
#                     cursor    = len(dec_table_ids) + 1
#                     event_args      = event['args']
#                     event_args_name = [arg[-1] for arg in event_args]
#                     target_info     = dict()

#                     for arg, prompt_slots in arg_2_prompt_slots.items():
#                         num_slots = len(prompt_slots['tok_s'])
#                         arg_slots.append([cursor + x for x in range(num_slots)])
#                         cursor += num_slots

#                         arg_target = {"text": [], "span_s": [], "span_e": []}
#                         if arg in event_args_name:
#                             if os.environ.get("DEBUG", False): counter[0] += 1
#                             arg_idxs = [j for j, x in enumerate(event_args_name) if x == arg]
#                             if os.environ.get("DEBUG", False): counter[1] += len(arg_idxs)
#                             for arg_idx in arg_idxs:
#                                 event_arg_info = event_args[arg_idx]
#                                 answer_text    = event_arg_info[2]
#                                 start_old, end_old = event_arg_info[0], event_arg_info[1]
#                                 start_position = old_tok_to_new_tok_index[start_old][0]
#                                 end_position   = old_tok_to_new_tok_index[end_old - 1][1]
#                                 arg_target["text"].append(answer_text)
#                                 arg_target["span_s"].append(start_position)
#                                 arg_target["span_e"].append(end_position)

#                         target_info[arg] = arg_target

#                     assert sum([len(slots) for slots in arg_slots]) == num_prompt_slots
#                     list_arg_slots.append(arg_slots)
#                     list_target_info.append(target_info)
#                     roles = self.argument_dict[event_type.replace(':', '.')]
#                     assert len(roles) == len(arg_slots)
#                     list_roles.append(roles)

#             max_dec_seq_len = self.args.max_prompt_seq_length
#             while len(dec_table_ids) < max_dec_seq_len:
#                 dec_table_ids.append(self.tokenizer.pad_token_id)
#                 dec_table_mask.append(self.args.pad_mask_token)

#             if len(all_ids) > self.args.max_enc_seq_length:
#                 enc_attention_mask = torch.zeros(
#                     (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
#                     dtype=torch.float32
#                 )
#             if len(all_ids) > self.args.max_enc_seq_length:
#                 all_ids      = all_ids[:self.args.max_enc_seq_length]
#                 all_mask_ids = all_mask_ids[:self.args.max_enc_seq_length]
#             if len(list_arg_2_prompt_slots) == 1:
#                 enc_attention_mask = torch.zeros(
#                     (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
#                     dtype=torch.float32
#                 )

#             feature_idx = len(features)
#             features.append(
#                 InputFeatures(
#                     example_id, feature_idx,
#                     list_event_type, trigger_enc_token_index,
#                     enc_text, enc_input_ids, enc_mask_ids, all_ids, all_mask_ids,
#                     dec_prompt_text, dec_table_ids, dec_table_mask, None,
#                     list_arg_2_prompt_slots, list_target_info, enc_attention_mask,
#                     old_tok_to_new_tok_index=old_tok_to_new_tok_index,
#                     full_text=example.context,
#                     arg_list=list_roles,
#                     # ── 新增三个字段 ──────────────────────────────────
#                     dep_heads=dep_heads,
#                     dep_rels=dep_rels,
#                     event_groups=event_groups,
#                 )
#             )

#         print(over_nums)
#         if os.environ.get("DEBUG", False):
#             print('\033[91m' + f"distinct/tot arg_role: {counter[0]}/{counter[1]} ({counter[2]})" + '\033[0m')
#         return features

#     def convert_features_to_dataset(self, features):
#         dataset = ArgumentExtractionDataset(features)
#         return dataset

# #版本3：共现图+句法依存图
# import os
# import re
# import sys

# sys.path.append("../")
# import torch
# import numpy as np
# from copy import deepcopy
# from torch.utils.data import Dataset
# from processors.processor_base import DSET_processor, SyntaxProvider
# from utils import EXTERNAL_TOKENS, _PREDEFINED_QUERY_TEMPLATE


# class InputFeatures(object):
#     """A single set of features of data."""

#     def __init__(self, example_id, feature_id,
#                  event_type, event_trigger,
#                  enc_text, enc_input_ids, enc_mask_ids, all_ids, all_mask_ids,
#                  dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
#                  arg_quries, arg_joint_prompt, target_info, enc_attention_mask,
#                  old_tok_to_new_tok_index=None, full_text=None, arg_list=None,
#                  # ── 新增三个字段 ─────────────────────────────────────────
#                  dep_heads=None,
#                  dep_rels=None,
#                  event_groups=None,
#                  ):

#         self.example_id = example_id
#         self.feature_id = feature_id
#         self.event_type = event_type       # list[str]，本样本所有事件的类型字符串
#         self.event_trigger = event_trigger
#         self.num_events = len(event_trigger)

#         self.enc_text = enc_text
#         self.enc_input_ids = enc_input_ids
#         self.enc_mask_ids = enc_mask_ids

#         self.all_ids = all_ids
#         self.all_mask_ids = all_mask_ids
#         self.enc_attention_mask = enc_attention_mask

#         self.dec_prompt_texts = dec_prompt_text
#         self.dec_prompt_ids = dec_prompt_ids
#         self.dec_prompt_mask_ids = dec_prompt_mask_ids

#         if arg_quries is not None:
#             self.dec_arg_query_ids = [v[0] for k, v in arg_quries.items()]
#             self.dec_arg_query_masks = [v[1] for k, v in arg_quries.items()]
#             self.dec_arg_start_positions = [v[2] for k, v in arg_quries.items()]
#             self.dec_arg_end_positions = [v[3] for k, v in arg_quries.items()]
#             self.start_position_ids = [v['span_s'] for k, v in target_info.items()]
#             self.end_position_ids = [v['span_e'] for k, v in target_info.items()]
#         else:
#             self.dec_arg_query_ids = None
#             self.dec_arg_query_masks = None

#         self.arg_joint_prompt = arg_joint_prompt
#         self.target_info = target_info
#         self.old_tok_to_new_tok_index = old_tok_to_new_tok_index
#         self.full_text = full_text
#         self.arg_list = arg_list

#         # ── 新增字段赋值 ──────────────────────────────────────────────
#         # dep_heads / dep_rels：subword 粒度，长度 = actual_seq_len（context + prompt）
#         # event_groups：每个事件对应的意群 group id，长度 = num_events
#         self.dep_heads    = dep_heads    if dep_heads    is not None else []
#         self.dep_rels     = dep_rels     if dep_rels     is not None else []
#         self.event_groups = event_groups if event_groups is not None else []

#     def find_idx(self, target, list):
#         for i, item in enumerate(list):
#             if item == target:
#                 return i

#     def init_pred(self):
#         self.pred_dict_tok  = [dict() for _ in range(self.num_events)]
#         self.pred_dict_word = [dict() for _ in range(self.num_events)]

#     def add_pred(self, role, span, event_index):
#         pred_dict_tok  = self.pred_dict_tok[event_index]
#         pred_dict_word = self.pred_dict_word[event_index]
#         if role not in pred_dict_tok:
#             pred_dict_tok[role] = list()
#         if span not in pred_dict_tok[role]:
#             pred_dict_tok[role].append(span)
#             if span != (0, 0):
#                 if role not in pred_dict_word:
#                     pred_dict_word[role] = list()
#                 word_span = self.get_word_span(span)
#                 if word_span not in pred_dict_word[role]:
#                     pred_dict_word[role].append(word_span)

#     def set_gt(self):
#         self.gt_dict_tok = [dict() for _ in range(self.num_events)]
#         for i, target_info in enumerate(self.target_info):
#             for k, v in target_info.items():
#                 self.gt_dict_tok[i][k] = [
#                     (s, e) for (s, e) in zip(v["span_s"], v["span_e"])
#                 ]

#         self.gt_dict_word = [dict() for _ in range(self.num_events)]
#         for i, gt_dict_tok in enumerate(self.gt_dict_tok):
#             gt_dict_word = self.gt_dict_word[i]
#             for role, spans in gt_dict_tok.items():
#                 for span in spans:
#                     if span != (0, 0):
#                         if role not in gt_dict_word:
#                             gt_dict_word[role] = list()
#                         word_span = self.get_word_span(span)
#                         gt_dict_word[role].append(word_span)

#     @property
#     def old_tok_index(self):
#         new_tok_index_to_old_tok_index = dict()
#         for old_tok_id, (new_tok_id_s, new_tok_id_e) in enumerate(
#                 self.old_tok_to_new_tok_index):
#             for j in range(new_tok_id_s, new_tok_id_e):
#                 new_tok_index_to_old_tok_index[j] = old_tok_id
#         return new_tok_index_to_old_tok_index

#     def get_word_span(self, span):
#         if span == (0, 0):
#             raise AssertionError()
#         offset = 0
#         span   = list(span)
#         span[0] = min(span[0], max(self.old_tok_index.keys()))
#         span[1] = max(span[1] - 1, min(self.old_tok_index.keys()))
#         while span[0] not in self.old_tok_index:
#             span[0] += 1
#         span_s = self.old_tok_index[span[0]] + offset
#         while span[1] not in self.old_tok_index:
#             span[1] -= 1
#         span_e = self.old_tok_index[span[1]] + offset
#         while span_e < span_s:
#             span_e += 1
#         return (span_s, span_e)

#     def __repr__(self):
#         s  = ""
#         s += "example_id: {}\n".format(self.example_id)
#         s += "event_type: {}\n".format(self.event_type)
#         s += "trigger_word: {}\n".format(self.event_trigger)
#         s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
#         s += "enc_input_ids: {}\n".format(self.enc_input_ids)
#         s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
#         s += "dec_prompt_ids: {}\n".format(self.dec_prompt_ids)
#         s += "dec_prompt_mask_ids: {}\n".format(self.dec_prompt_mask_ids)
#         s += "event_groups: {}\n".format(self.event_groups)
#         s += "dep_heads[:10]: {}\n".format(self.dep_heads[:10])
#         s += "dep_rels[:10]: {}\n".format(self.dep_rels[:10])
#         return s


# # ===========================================================================
# # Dataset
# # ===========================================================================

# class ArgumentExtractionDataset(Dataset):
#     def __init__(self, features):
#         self.features = features

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx]

#     @staticmethod
#     def collate_fn(batch):
#         enc_input_ids  = torch.tensor([f.enc_input_ids  for f in batch])
#         enc_mask_ids   = torch.tensor([f.enc_mask_ids   for f in batch])
#         all_ids        = torch.tensor([f.all_ids        for f in batch])
#         all_mask_ids   = torch.tensor([f.all_mask_ids   for f in batch])

#         if batch[0].dec_prompt_ids is not None:
#             dec_prompt_ids      = torch.tensor([f.dec_prompt_ids      for f in batch])
#             dec_prompt_mask_ids = torch.tensor([f.dec_prompt_mask_ids for f in batch])
#         else:
#             dec_prompt_ids      = None
#             dec_prompt_mask_ids = None

#         example_idx = [f.example_id for f in batch]
#         feature_idx = torch.tensor([f.feature_id for f in batch])

#         if batch[0].dec_arg_query_ids is not None:
#             dec_arg_query_ids       = [torch.LongTensor(f.dec_arg_query_ids)       for f in batch]
#             dec_arg_query_mask_ids  = [torch.LongTensor(f.dec_arg_query_masks)     for f in batch]
#             dec_arg_start_positions = [torch.LongTensor(f.dec_arg_start_positions) for f in batch]
#             dec_arg_end_positions   = [torch.LongTensor(f.dec_arg_end_positions)   for f in batch]
#             start_position_ids      = [torch.FloatTensor(f.start_position_ids)     for f in batch]
#             end_position_ids        = [torch.FloatTensor(f.end_position_ids)       for f in batch]
#         else:
#             dec_arg_query_ids       = None
#             dec_arg_query_mask_ids  = None
#             dec_arg_start_positions = None
#             dec_arg_end_positions   = None
#             start_position_ids      = None
#             end_position_ids        = None

#         target_info              = [f.target_info              for f in batch]
#         old_tok_to_new_tok_index = [f.old_tok_to_new_tok_index for f in batch]
#         arg_joint_prompt         = [f.arg_joint_prompt         for f in batch]
#         arg_lists                = [f.arg_list                 for f in batch]
#         event_trigger            = [f.event_trigger            for f in batch]
#         enc_attention_mask       = [f.enc_attention_mask       for f in batch]

#         # ── 图1 字段：直接以 list 传出，无需 padding ──────────────────
#         dep_heads_batch    = [f.dep_heads    for f in batch]   # list[list[int]]
#         dep_rels_batch     = [f.dep_rels     for f in batch]   # list[list[str]]
#         event_groups_batch = [f.event_groups for f in batch]   # list[list[int]]

#         # ── 图2 字段：事件类型列表，用于共现查询 ──────────────────────
#         # f.event_type 是 list_event_type，即本样本所有事件的类型字符串列表
#         event_types_batch  = [f.event_type  for f in batch]   # list[list[str]]

#         return (
#             enc_input_ids, enc_mask_ids,
#             all_ids, all_mask_ids,
#             dec_arg_query_ids, dec_arg_query_mask_ids,
#             dec_prompt_ids, dec_prompt_mask_ids,
#             target_info, old_tok_to_new_tok_index,
#             arg_joint_prompt, arg_lists,
#             example_idx, feature_idx,
#             dec_arg_start_positions, dec_arg_end_positions,
#             start_position_ids, end_position_ids,
#             event_trigger, enc_attention_mask,
#             dep_heads_batch, dep_rels_batch, event_groups_batch,  # 图1
#             event_types_batch,                                     # 图2 ← 新增
#         )


# # ===========================================================================
# # Processor
# # ===========================================================================

# class MultiargProcessor(DSET_processor):
#     def __init__(self, args, tokenizer):
#         super().__init__(args, tokenizer)
#         self.set_dec_input()
#         self.collate_fn = ArgumentExtractionDataset.collate_fn

#         # 句法解析器（全局单例，避免重复加载）
#         # SyntaxProvider 已在 processor_base.__init__ 中初始化，此处直接复用
#         if not hasattr(self, 'syntax_provider'):
#             self.syntax_provider = SyntaxProvider()

#     def set_dec_input(self):
#         self.arg_query    = False
#         self.prompt_query = False
#         if self.args.model_type == "base":
#             self.arg_query = True
#         elif "DyGMA" in self.args.model_type:
#             self.prompt_query = True
#         else:
#             raise NotImplementedError(f"Unexpected setting {self.args.model_type}")

#     @staticmethod
#     def _read_prompt_group(prompt_path):
#         with open(prompt_path) as f:
#             lines = f.readlines()
#         prompts = dict()
#         for line in lines:
#             if not line:
#                 continue
#             event_type, prompt = line.split(":")
#             prompts[event_type] = prompt
#         return prompts

#     def create_dec_qury(self, arg, event_trigger):
#         dec_text = _PREDEFINED_QUERY_TEMPLATE.format(
#             arg=arg, trigger=event_trigger
#         )
#         dec = self.tokenizer(dec_text)
#         dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]
#         while len(dec_input_ids) < self.args.max_dec_seq_length:
#             dec_input_ids.append(self.tokenizer.pad_token_id)
#             dec_mask_ids.append(self.args.pad_mask_token)
    #     matching_result = re.search(arg, dec_text)
    #     char_idx_s, char_idx_e = matching_result.span()
    #     char_idx_e -= 1
    #     tok_prompt_s = dec.char_to_token(char_idx_s)
    #     tok_prompt_e = dec.char_to_token(char_idx_e) + 1
    #     return dec_input_ids, dec_mask_ids, tok_prompt_s, tok_prompt_e

    # def find_idx(self, target, list):
    #     for i, item in enumerate(list):
    #         if item == target:
    #             return i

    # # ── 新增：依存解析 + subword 对齐 ─────────────────────────────────
    # def _build_dep_fields(
    #     self,
    #     context:                  list,    # 原始 token 列表（无 marker）
    #     marked_context:           list,    # 含 <t-i> marker 的 token 列表（备用）
    #     old_tok_to_new_tok_index: list,    # word_i → [new_tok_s, new_tok_e]
    #     enc:                      object,  # tokenizer 编码结果
    #     seq_len:                  int,     # all_ids 的实际有效长度
    # ):
    #     """
    #     对原始 context（无 marker）做依存解析，
    #     再通过 old_tok_to_new_tok_index 映射到 subword 粒度。

    #     Returns:
    #         dep_heads : list[int]  长度 = seq_len，每个 subword 的 head subword 索引
    #         dep_rels  : list[str]  长度 = seq_len，每个 subword 的依存关系标签
    #     """
    #     # 默认：每个 subword 自指，关系为 'none'
    #     dep_heads = list(range(seq_len))
    #     dep_rels  = ['none'] * seq_len

    #     # 用原始 context（无 marker）做解析，避免 <t-0> 等特殊符号干扰 spaCy
    #     plain_text = " ".join(context)
    #     try:
    #         doc = self.syntax_provider.nlp(plain_text)
    #     except Exception as e:
    #         import logging
    #         logging.getLogger(__name__).warning(
    #             f"[dep_parse] spaCy 解析失败: {e}"
    #         )
    #         return dep_heads, dep_rels

    #     # 构建 char_start → word_idx 的反查表（基于原始 context）
    #     char_start_to_word_idx = {}
    #     char_pos = 0
    #     for w_i, tok in enumerate(context):
    #         char_start_to_word_idx[char_pos] = w_i
    #         char_pos += len(tok) + 1    # +1 for space

    #     # word_i → (head_word_i, deprel)
    #     word_dep = {}
    #     for spacy_tok in doc:
    #         w_i = char_start_to_word_idx.get(spacy_tok.idx, None)
    #         if w_i is None:
    #             continue
    #         head_w_i = char_start_to_word_idx.get(spacy_tok.head.idx, w_i)
    #         word_dep[w_i] = (head_w_i, spacy_tok.dep_)

    #     # 映射到 subword 层
    #     # old_tok_to_new_tok_index[w_i] = [new_tok_s, new_tok_e]
    #     for w_i, (head_w_i, rel) in word_dep.items():
    #         if w_i >= len(old_tok_to_new_tok_index):
    #             continue

    #         sw_range   = old_tok_to_new_tok_index[w_i]
    #         sw_s, sw_e = sw_range[0], sw_range[1]

    #         # head 的第一个 subword
    #         if head_w_i < len(old_tok_to_new_tok_index):
    #             head_sw = old_tok_to_new_tok_index[head_w_i][0]
    #         else:
    #             head_sw = sw_s    # fallback 自指

    #         # 当前 word 的所有 subword 共享同一 head/rel
    #         for sw in range(sw_s, min(sw_e, seq_len)):
    #             dep_heads[sw] = min(head_sw, seq_len - 1)
    #             dep_rels[sw]  = rel

    #     return dep_heads, dep_rels

    # # ── 主转换函数 ────────────────────────────────────────────────────
    # def convert_examples_to_features(self, examples, role_name_mapping=None):
    #     features  = []
    #     over_nums = 0

    #     if self.prompt_query:
    #         prompts = self._read_prompt_group(self.args.prompt_path)

    #     if os.environ.get("DEBUG", False):
    #         counter = [0, 0, 0]

    #     for example in examples:
    #         example_id          = example.doc_id
    #         context             = example.context           # 原始 token 列表
    #         event_type_2_events = example.event_type_2_events

    #         list_event_type = []
    #         triggers        = []
    #         for event_type, events in event_type_2_events.items():
    #             list_event_type += [e['event_type'] for e in events]
    #             triggers        += [tuple(e['trigger']) for e in events]

    #         set_triggers = sorted(list(set(triggers)))

    #         # 触发词重叠检测
    #         trigger_overlap = False
    #         for t1 in set_triggers:
    #             for t2 in set_triggers:
    #                 if t1[0] == t2[0] and t1[1] == t2[1]:
    #                     continue
    #                 if ((t1[0] < t2[1] and t2[0] < t1[1]) or
    #                         (t2[0] < t1[1] and t1[0] < t2[1])):
    #                     trigger_overlap = True
    #                     break
    #         if trigger_overlap:
    #             print('[trigger_overlap]', event_type_2_events)
    #             exit(0)

    #         # ── 构建 marked_context（与原逻辑完全一致）────────────────
    #         offset         = 0
    #         marked_context = deepcopy(context)
    #         marker_indice  = list(range(len(triggers)))
    #         for i, t in enumerate(set_triggers):
    #             t_start = t[0]
    #             t_end   = t[1]
    #             marked_context = (
    #                 marked_context[:(t_start + offset)]
    #                 + ['<t-%d>' % marker_indice[i]]
    #                 + context[t_start: t_end]
    #                 + ['</t-%d>' % marker_indice[i]]
    #                 + context[t_end:]
    #             )
    #             offset += 2
    #         enc_text = " ".join(marked_context)

    #         # ── old_tok_to_new_tok_index（与原逻辑完全一致）──────────
    #         old_tok_to_char_index    = []
    #         old_tok_to_new_tok_index = []

    #         curr         = 0
    #         enc          = self.tokenizer(enc_text, add_special_tokens=True)
    #         trigger_list = [[] for _ in range(len(triggers))]
    #         for tok in marked_context:
    #             if tok not in EXTERNAL_TOKENS:
    #                 old_tok_to_char_index.append([curr, curr + len(tok) - 1])
    #             curr += len(tok) + 1

    #         enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
    #         if len(enc_input_ids) > self.args.max_enc_seq_length:
    #             raise ValueError(
    #                 f"Please increase max_enc_seq_length above {len(enc_input_ids)}"
    #             )

    #         all_ids      = enc_input_ids.copy()
    #         all_mask_ids = enc_mask_ids.copy()
    #         type_ids     = enc_mask_ids.copy()

    #         offset_prompt = len(enc_input_ids)

    #         while len(enc_input_ids) < self.args.max_enc_seq_length:
    #             enc_input_ids.append(self.tokenizer.pad_token_id)
    #             enc_mask_ids.append(self.args.pad_mask_token)

    #         for old_tok_idx, (char_idx_s, char_idx_e) in enumerate(
    #                 old_tok_to_char_index):
    #             new_tok_s = enc.char_to_token(char_idx_s)
    #             new_tok_e = enc.char_to_token(char_idx_e) + 1
    #             old_tok_to_new_tok_index.append([new_tok_s, new_tok_e])

    #         trigger_enc_token_index = []
    #         for t in triggers:
    #             new_t_start = old_tok_to_new_tok_index[t[0]][0]
    #             new_t_end   = old_tok_to_new_tok_index[t[1] - 1][1]
    #             trigger_enc_token_index.append([new_t_start, new_t_end])
    #         for ii, it in enumerate(trigger_enc_token_index):
    #             type_ids[it[0] - 1] = ii + 2

    #         dec_table_ids  = []
    #         dec_table_mask = []

    #         list_arg_2_prompt_slots      = []
    #         list_num_prompt_slots        = []
    #         list_dec_prompt_ids          = []
    #         list_arg_2_prompt_slot_spans = []
    #         offset_prompt_               = 0
    #         kk                           = 0
    #         enc_attention_mask           = torch.zeros(
    #             (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
    #             dtype=torch.float32,
    #         )

    #         for i, event_type in enumerate(event_type_2_events):
    #             events     = event_type_2_events[event_type]
    #             event_name = event_type.split('.')
    #             event_name = (
    #                 ['<e-%d>' % i] + event_name + ['</e-%d>' % i]
    #             )
    #             for event in events:
    #                 enc_trigger_start = trigger_enc_token_index[kk][0] - 1
    #                 enc_trigger_end   = trigger_enc_token_index[kk][1] + 1
    #                 kk += 1
    #                 dec_prompt_text = prompts[event_type].strip()
    #                 assert dec_prompt_text
    #                 dec_prompt_text = ' '.join(event_name) + ' ' + dec_prompt_text
    #                 dec_prompt      = self.tokenizer(
    #                     dec_prompt_text, add_special_tokens=True
    #                 )
    #                 dec_prompt_ids_, dec_prompt_mask_ids_ = (
    #                     dec_prompt["input_ids"],
    #                     dec_prompt["attention_mask"],
    #                 )

    #                 arg_list = self.argument_dict[
    #                     event_type.replace(':', '.')
    #                 ]
    #                 arg_2_prompt_slots      = dict()
    #                 arg_2_prompt_slot_spans = dict()
    #                 num_prompt_slots        = 0
    #                 if os.environ.get("DEBUG", False):
    #                     arg_set = set()
    #                 for arg in arg_list:
    #                     prompt_slots      = {
    #                         "tok_s": [], "tok_e": [],
    #                         "tok_s_off": [], "tok_e_off": [],
    #                     }
    #                     prompt_slot_spans = []
    #                     if role_name_mapping is not None:
    #                         arg_ = role_name_mapping[event_type][arg]
    #                     else:
    #                         arg_ = arg
    #                     for matching_result in re.finditer(
    #                         r'\b' + re.escape(arg_) + r'\b',
    #                         dec_prompt_text.split('.')[0],
    #                     ):
    #                         char_idx_s, char_idx_e = matching_result.span()
    #                         char_idx_e -= 1
    #                         tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
    #                         tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
    #                         prompt_slot_spans.append((tok_prompt_s, tok_prompt_e))
    #                         prompt_slots["tok_s"].append(
    #                             tok_prompt_s + offset_prompt_
    #                         )
    #                         prompt_slots["tok_e"].append(
    #                             tok_prompt_e + offset_prompt_
    #                         )
    #                         prompt_slots["tok_s_off"].append(
    #                             tok_prompt_s + offset_prompt + offset_prompt_
    #                         )
    #                         prompt_slots["tok_e_off"].append(
    #                             tok_prompt_e + offset_prompt + offset_prompt_
    #                         )
    #                         num_prompt_slots += 1
    #                     arg_2_prompt_slots[arg]      = prompt_slots
    #                     arg_2_prompt_slot_spans[arg] = prompt_slot_spans

    #                 list_arg_2_prompt_slots.append(arg_2_prompt_slots)
    #                 list_num_prompt_slots.append(num_prompt_slots)
    #                 list_dec_prompt_ids.append(dec_prompt_ids_)
    #                 list_arg_2_prompt_slot_spans.append(arg_2_prompt_slot_spans)

    #                 enc_attention_mask[
    #                     0,
    #                     enc_trigger_start:enc_trigger_end,
    #                     offset_prompt + offset_prompt_:
    #                     offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
    #                 ] = 1
    #                 enc_attention_mask[
    #                     0,
    #                     offset_prompt + offset_prompt_:
    #                     offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
    #                     enc_trigger_start:enc_trigger_end,
    #                 ] = 1
    #                 enc_attention_mask[
    #                     1,
    #                     enc_trigger_start:enc_trigger_end,
    #                     offset_prompt:offset_prompt + offset_prompt_,
    #                 ] = 1
    #                 enc_attention_mask[
    #                     1,
    #                     enc_trigger_start:enc_trigger_end,
    #                     offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
    #                 ] = 1
    #                 enc_attention_mask[
    #                     1,
    #                     offset_prompt:offset_prompt + offset_prompt_,
    #                     enc_trigger_start:enc_trigger_end,
    #                 ] = 1
    #                 enc_attention_mask[
    #                     1,
    #                     offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
    #                     enc_trigger_start:enc_trigger_end,
    #                 ] = 1

    #             enc_attention_mask[
    #                 0,
    #                 offset_prompt + offset_prompt_:
    #                 offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
    #                 offset_prompt + offset_prompt_:
    #                 offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
    #             ] = 1
    #             enc_attention_mask[
    #                 1,
    #                 offset_prompt + offset_prompt_:
    #                 offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
    #                 offset_prompt:offset_prompt + offset_prompt_,
    #             ] = 1
    #             enc_attention_mask[
    #                 1,
    #                 offset_prompt + offset_prompt_:
    #                 offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
    #                 offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
    #             ] = 1
    #             enc_attention_mask[
    #                 1,
    #                 offset_prompt:offset_prompt + offset_prompt_,
    #                 offset_prompt + offset_prompt_:
    #                 offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
    #             ] = 1
    #             enc_attention_mask[
    #                 1,
    #                 offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
    #                 offset_prompt + offset_prompt_:
    #                 offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
    #             ] = 1

    #             offset_prompt_ += len(dec_prompt_ids_)
    #             dec_table_ids   += dec_prompt_ids_
    #             dec_table_mask  += dec_prompt_mask_ids_

    #         all_ids.extend(dec_table_ids)
    #         all_mask_ids.extend(dec_table_mask)
    #         if len(all_ids) > self.args.max_enc_seq_length:
    #             over_nums += 1

    #         while len(all_ids) < self.args.max_enc_seq_length:
    #             all_ids.append(self.tokenizer.pad_token_id)
    #             all_mask_ids.append(self.args.pad_mask_token)

    #         # ── 新增①②：依存解析 ──────────────────────────────────────
    #         # actual_seq_len：context + 所有 prompt 的实际 token 数（不含 padding）
    #         actual_seq_len = min(
    #             offset_prompt + offset_prompt_,
    #             self.args.max_enc_seq_length,
    #         )

    #         dep_heads, dep_rels = self._build_dep_fields(
    #             context=context,
    #             marked_context=marked_context,
    #             old_tok_to_new_tok_index=old_tok_to_new_tok_index,
    #             enc=enc,
    #             seq_len=actual_seq_len,
    #         )

    #         # ── 新增③：意群 group id，从数据集的 token_chunk_ids 读取 ──
    #         # token_chunk_ids[trigger_start] 即为该触发词所在意群的 id
    #         # triggers 列表顺序与 list_event_type 保持一致
    #         token_chunk_ids = example.token_chunk_ids
    #         event_groups    = []
    #         for trig in triggers:
    #             trig_start = trig[0]    # 触发词在原始 context 中的起始 word 索引
    #             if token_chunk_ids and trig_start < len(token_chunk_ids):
    #                 event_groups.append(token_chunk_ids[trig_start])
    #             else:
    #                 event_groups.append(0)    # 无意群信息时默认同组

    #         # ── 处理 target arguments（与原逻辑完全一致）────────────────
    #         row_index        = 0
    #         list_trigger_pos = []
    #         list_arg_slots   = []
    #         list_target_info = []
    #         list_roles       = []
    #         k                = 0

    #         for i, (event_type, events) in enumerate(
    #                 event_type_2_events.items()):
    #             for event in events:
    #                 arg_2_prompt_slots = list_arg_2_prompt_slots[k]
    #                 num_prompt_slots   = list_num_prompt_slots[k]
    #                 dec_prompt_ids_    = list_dec_prompt_ids[k]
    #                 k         += 1
    #                 row_index += 1

    #                 list_trigger_pos.append(len(dec_table_ids))
    #                 arg_slots = []
    #                 cursor    = len(dec_table_ids) + 1
    #                 event_args      = event['args']
    #                 event_args_name = [arg[-1] for arg in event_args]
    #                 target_info     = dict()

    #                 for arg, prompt_slots in arg_2_prompt_slots.items():
    #                     num_slots = len(prompt_slots['tok_s'])
    #                     arg_slots.append(
    #                         [cursor + x for x in range(num_slots)]
    #                     )
    #                     cursor += num_slots

    #                     arg_target = {"text": [], "span_s": [], "span_e": []}
    #                     if arg in event_args_name:
    #                         if os.environ.get("DEBUG", False):
    #                             counter[0] += 1
    #                         arg_idxs = [
    #                             j for j, x in enumerate(event_args_name)
    #                             if x == arg
    #                         ]
    #                         if os.environ.get("DEBUG", False):
    #                             counter[1] += len(arg_idxs)
    #                         for arg_idx in arg_idxs:
    #                             event_arg_info = event_args[arg_idx]
    #                             answer_text    = event_arg_info[2]
    #                             start_old, end_old = (
    #                                 event_arg_info[0], event_arg_info[1]
    #                             )
    #                             start_position = old_tok_to_new_tok_index[
    #                                 start_old
    #                             ][0]
    #                             end_position   = old_tok_to_new_tok_index[
    #                                 end_old - 1
    #                             ][1]
    #                             arg_target["text"].append(answer_text)
    #                             arg_target["span_s"].append(start_position)
    #                             arg_target["span_e"].append(end_position)

    #                     target_info[arg] = arg_target

    #                 assert sum(
    #                     [len(slots) for slots in arg_slots]
    #                 ) == num_prompt_slots
    #                 list_arg_slots.append(arg_slots)
    #                 list_target_info.append(target_info)
    #                 roles = self.argument_dict[
    #                     event_type.replace(':', '.')
    #                 ]
    #                 assert len(roles) == len(arg_slots)
    #                 list_roles.append(roles)

    #         max_dec_seq_len = self.args.max_prompt_seq_length
    #         while len(dec_table_ids) < max_dec_seq_len:
    #             dec_table_ids.append(self.tokenizer.pad_token_id)
    #             dec_table_mask.append(self.args.pad_mask_token)

    #         if len(all_ids) > self.args.max_enc_seq_length:
    #             enc_attention_mask = torch.zeros(
    #                 (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
    #                 dtype=torch.float32,
    #             )
    #         if len(all_ids) > self.args.max_enc_seq_length:
    #             all_ids      = all_ids[:self.args.max_enc_seq_length]
    #             all_mask_ids = all_mask_ids[:self.args.max_enc_seq_length]
    #         if len(list_arg_2_prompt_slots) == 1:
    #             enc_attention_mask = torch.zeros(
    #                 (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
    #                 dtype=torch.float32,
    #             )

    #         feature_idx = len(features)
    #         features.append(
    #             InputFeatures(
    #                 example_id, feature_idx,
    #                 list_event_type,           # ← 保存事件类型列表，供图2使用
    #                 trigger_enc_token_index,
    #                 enc_text, enc_input_ids, enc_mask_ids,
    #                 all_ids, all_mask_ids,
    #                 dec_prompt_text, dec_table_ids, dec_table_mask,
    #                 None,
    #                 list_arg_2_prompt_slots, list_target_info,
    #                 enc_attention_mask,
    #                 old_tok_to_new_tok_index=old_tok_to_new_tok_index,
    #                 full_text=example.context,
    #                 arg_list=list_roles,
    #                 # ── 新增三个字段 ──────────────────────────────────
    #                 dep_heads=dep_heads,
    #                 dep_rels=dep_rels,
    #                 event_groups=event_groups,
    #             )
    #         )

    #     print(over_nums)
    #     if os.environ.get("DEBUG", False):
    #         print(
    #             '\033[91m'
    #             + f"distinct/tot arg_role: {counter[0]}/{counter[1]} ({counter[2]})"
    #             + '\033[0m'
    #         )
    #     return features

    # def convert_features_to_dataset(self, features):
    #     dataset = ArgumentExtractionDataset(features)
    #     return dataset

#版本4：句法依赖+共指
import os
import re
import sys

sys.path.append("../")
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from processors.processor_base import DSET_processor, SyntaxProvider
from utils import EXTERNAL_TOKENS, _PREDEFINED_QUERY_TEMPLATE


# ─────────────────────────────────────────────────────────────────────────────
# InputFeatures
# ─────────────────────────────────────────────────────────────────────────────
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, feature_id,
                 event_type, event_trigger,
                 enc_text, enc_input_ids, enc_mask_ids, all_ids, all_mask_ids,
                 dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
                 arg_quries, arg_joint_prompt, target_info, enc_attention_mask,
                 old_tok_to_new_tok_index=None, full_text=None, arg_list=None,
                 dep_heads=None,
                 dep_rels=None,
                 event_groups=None,
                 coref_clusters=None,   # ← 新增：list[list[int]] subword级共指簇
                 coref_logits=None
                 ):

        self.example_id    = example_id
        self.feature_id    = feature_id
        self.event_type    = event_type
        self.event_trigger = event_trigger
        self.num_events    = len(event_trigger)

        self.enc_text           = enc_text
        self.enc_input_ids      = enc_input_ids
        self.enc_mask_ids       = enc_mask_ids
        self.all_ids            = all_ids
        self.all_mask_ids       = all_mask_ids
        self.enc_attention_mask = enc_attention_mask

        self.dec_prompt_texts    = dec_prompt_text
        self.dec_prompt_ids      = dec_prompt_ids
        self.dec_prompt_mask_ids = dec_prompt_mask_ids

        if arg_quries is not None:
            self.dec_arg_query_ids       = [v[0] for k, v in arg_quries.items()]
            self.dec_arg_query_masks     = [v[1] for k, v in arg_quries.items()]
            self.dec_arg_start_positions = [v[2] for k, v in arg_quries.items()]
            self.dec_arg_end_positions   = [v[3] for k, v in arg_quries.items()]
            self.start_position_ids      = [v['span_s'] for k, v in target_info.items()]
            self.end_position_ids        = [v['span_e'] for k, v in target_info.items()]
        else:
            self.dec_arg_query_ids       = None
            self.dec_arg_query_masks     = None

        self.arg_joint_prompt         = arg_joint_prompt
        self.target_info              = target_info
        self.old_tok_to_new_tok_index  = old_tok_to_new_tok_index
        self.full_text                = full_text
        self.arg_list                 = arg_list

        # ── 图相关字段 ────────────────────────────────────────────────
        self.dep_heads      = dep_heads      if dep_heads      is not None else []
        self.dep_rels       = dep_rels       if dep_rels       is not None else []
        self.event_groups   = event_groups   if event_groups   is not None else []
        # coref_clusters：subword 粒度共指簇
        # 格式：[[sw_rep_0, sw_rep_1, ...], [sw_rep_0, ...], ...]
        # 每个子列表为同一实体的各 mention 代表 subword，第一个为超级节点
        # self.coref_clusters = coref_clusters if coref_clusters is not None else []
        self.coref_clusters  = coref_clusters
        self.coref_logits    = coref_logits if coref_logits is not None else []

    def find_idx(self, target, lst):
        for i, item in enumerate(lst):
            if item == target:
                return i

    def init_pred(self):
        self.pred_dict_tok  = [dict() for _ in range(self.num_events)]
        self.pred_dict_word = [dict() for _ in range(self.num_events)]

    def add_pred(self, role, span, event_index):
        pred_dict_tok  = self.pred_dict_tok[event_index]
        pred_dict_word = self.pred_dict_word[event_index]
        if role not in pred_dict_tok:
            pred_dict_tok[role] = list()
        if span not in pred_dict_tok[role]:
            pred_dict_tok[role].append(span)
            if span != (0, 0):
                if role not in pred_dict_word:
                    pred_dict_word[role] = list()
                word_span = self.get_word_span(span)
                if word_span not in pred_dict_word[role]:
                    pred_dict_word[role].append(word_span)

    def set_gt(self):
        self.gt_dict_tok = [dict() for _ in range(self.num_events)]
        for i, target_info in enumerate(self.target_info):
            for k, v in target_info.items():
                self.gt_dict_tok[i][k] = [
                    (s, e) for (s, e) in zip(v["span_s"], v["span_e"])
                ]

        self.gt_dict_word = [dict() for _ in range(self.num_events)]
        for i, gt_dict_tok in enumerate(self.gt_dict_tok):
            gt_dict_word = self.gt_dict_word[i]
            for role, spans in gt_dict_tok.items():
                for span in spans:
                    if span != (0, 0):
                        if role not in gt_dict_word:
                            gt_dict_word[role] = list()
                        word_span = self.get_word_span(span)
                        gt_dict_word[role].append(word_span)

    @property
    def old_tok_index(self):
        new_tok_index_to_old_tok_index = dict()
        for old_tok_id, (new_tok_id_s, new_tok_id_e) in enumerate(
                self.old_tok_to_new_tok_index):
            for j in range(new_tok_id_s, new_tok_id_e):
                new_tok_index_to_old_tok_index[j] = old_tok_id
        return new_tok_index_to_old_tok_index

    def get_word_span(self, span):
        if span == (0, 0):
            raise AssertionError()
        offset  = 0
        span    = list(span)
        span[0] = min(span[0], max(self.old_tok_index.keys()))
        span[1] = max(span[1] - 1, min(self.old_tok_index.keys()))
        while span[0] not in self.old_tok_index:
            span[0] += 1
        span_s = self.old_tok_index[span[0]] + offset
        while span[1] not in self.old_tok_index:
            span[1] -= 1
        span_e = self.old_tok_index[span[1]] + offset
        while span_e < span_s:
            span_e += 1
        return (span_s, span_e)

    def __repr__(self):
        s  = ""
        s += "example_id: {}\n".format(self.example_id)
        s += "event_type: {}\n".format(self.event_type)
        s += "trigger_word: {}\n".format(self.event_trigger)
        s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
        s += "enc_input_ids: {}\n".format(self.enc_input_ids)
        s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
        s += "dec_prompt_ids: {}\n".format(self.dec_prompt_ids)
        s += "dec_prompt_mask_ids: {}\n".format(self.dec_prompt_mask_ids)
        s += "event_groups: {}\n".format(self.event_groups)
        s += "dep_heads[:10]: {}\n".format(self.dep_heads[:10])
        s += "dep_rels[:10]: {}\n".format(self.dep_rels[:10])
        s += "coref_clusters[:3]: {}\n".format(self.coref_clusters[:3])
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class ArgumentExtractionDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    @staticmethod
    def _pad_prompt_batch(batch):
        max_len = max(len(f.dec_prompt_ids) for f in batch)

        pad_token_id = 1
        for f in batch:
            for tok_id, mask_id in zip(f.dec_prompt_ids, f.dec_prompt_mask_ids):
                if mask_id == 0:
                    pad_token_id = tok_id
                    break
            else:
                continue
            break

        padded_ids, padded_masks = [], []
        for f in batch:
            pad_len = max_len - len(f.dec_prompt_ids)
            padded_ids.append(f.dec_prompt_ids + [pad_token_id] * pad_len)
            padded_masks.append(f.dec_prompt_mask_ids + [0] * pad_len)
        return torch.tensor(padded_ids), torch.tensor(padded_masks)

    @staticmethod
    def collate_fn(batch):
        enc_input_ids  = torch.tensor([f.enc_input_ids  for f in batch])
        enc_mask_ids   = torch.tensor([f.enc_mask_ids   for f in batch])
        all_ids        = torch.tensor([f.all_ids        for f in batch])
        all_mask_ids   = torch.tensor([f.all_mask_ids   for f in batch])

        if batch[0].dec_prompt_ids is not None:
            dec_prompt_ids, dec_prompt_mask_ids = ArgumentExtractionDataset._pad_prompt_batch(batch)
        else:
            dec_prompt_ids      = None
            dec_prompt_mask_ids = None

        example_idx = [f.example_id for f in batch]
        feature_idx = torch.tensor([f.feature_id for f in batch])

        if batch[0].dec_arg_query_ids is not None:
            dec_arg_query_ids       = [torch.LongTensor(f.dec_arg_query_ids)       for f in batch]
            dec_arg_query_mask_ids  = [torch.LongTensor(f.dec_arg_query_masks)     for f in batch]
            dec_arg_start_positions = [torch.LongTensor(f.dec_arg_start_positions) for f in batch]
            dec_arg_end_positions   = [torch.LongTensor(f.dec_arg_end_positions)   for f in batch]
            start_position_ids      = [torch.FloatTensor(f.start_position_ids)     for f in batch]
            end_position_ids        = [torch.FloatTensor(f.end_position_ids)       for f in batch]
        else:
            dec_arg_query_ids       = None
            dec_arg_query_mask_ids  = None
            dec_arg_start_positions = None
            dec_arg_end_positions   = None
            start_position_ids      = None
            end_position_ids        = None

        target_info              = [f.target_info              for f in batch]
        old_tok_to_new_tok_index = [f.old_tok_to_new_tok_index for f in batch]
        arg_joint_prompt         = [f.arg_joint_prompt         for f in batch]
        arg_lists                = [f.arg_list                 for f in batch]
        event_trigger            = [f.event_trigger            for f in batch]
        enc_attention_mask       = [f.enc_attention_mask       for f in batch]

        # ── 图1 字段 ──────────────────────────────────────────────────
        dep_heads_batch    = [f.dep_heads    for f in batch]   # list[list[int]]
        dep_rels_batch     = [f.dep_rels     for f in batch]   # list[list[str]]
        event_groups_batch = [f.event_groups for f in batch]   # list[list[int]]

        # ── 共指簇字段（新增）────────────────────────────────────────
        # coref_clusters 是嵌套列表（每个样本的簇数量和每簇大小均不同）
        # 直接以 list 传出，不 padding
        coref_clusters_batch = [f.coref_clusters for f in batch]  # list[list[list[int]]]
        coref_logits_batch   = [f.coref_logits   for f in batch]

        return (
            enc_input_ids, enc_mask_ids,
            all_ids, all_mask_ids,
            dec_arg_query_ids, dec_arg_query_mask_ids,
            dec_prompt_ids, dec_prompt_mask_ids,
            target_info, old_tok_to_new_tok_index,
            arg_joint_prompt, arg_lists,
            example_idx, feature_idx,
            dec_arg_start_positions, dec_arg_end_positions,
            start_position_ids, end_position_ids,
            event_trigger, enc_attention_mask,
            dep_heads_batch, dep_rels_batch, event_groups_batch,  # [20][21][22]
            coref_clusters_batch,                                  # [23] ← 新增
            coref_logits_batch,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Processor
# ─────────────────────────────────────────────────────────────────────────────
class MultiargProcessor(DSET_processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.set_dec_input()
        self._coref_sample_count=0
        self.collate_fn = ArgumentExtractionDataset.collate_fn

        if not hasattr(self, 'syntax_provider'):
            self.syntax_provider = SyntaxProvider()
        # coref_provider 已在 DSET_processor.__init__ 中初始化

    def set_dec_input(self):
        self.arg_query    = False
        self.prompt_query = False
        if self.args.model_type == "base":
            self.arg_query = True
        elif "DyGMA" in self.args.model_type:
            self.prompt_query = True
        else:
            raise NotImplementedError(f"Unexpected setting {self.args.model_type}")

    @staticmethod
    def _read_prompt_group(prompt_path):
        with open(prompt_path) as f:
            lines = f.readlines()
        prompts = dict()
        for line in lines:
            if not line:
                continue
            event_type, prompt = line.split(":")
            prompts[event_type] = prompt
        return prompts

    def create_dec_qury(self, arg, event_trigger):
        dec_text = _PREDEFINED_QUERY_TEMPLATE.format(
            arg=arg, trigger=event_trigger
        )
        dec = self.tokenizer(dec_text)
        dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]
        while len(dec_input_ids) < self.args.max_dec_seq_length:
            dec_input_ids.append(self.tokenizer.pad_token_id)
            dec_mask_ids.append(self.args.pad_mask_token)
        matching_result = re.search(arg, dec_text)
        char_idx_s, char_idx_e = matching_result.span()
        char_idx_e -= 1
        tok_prompt_s = dec.char_to_token(char_idx_s)
        tok_prompt_e = dec.char_to_token(char_idx_e) + 1
        return dec_input_ids, dec_mask_ids, tok_prompt_s, tok_prompt_e

    def find_idx(self, target, lst):
        for i, item in enumerate(lst):
            if item == target:
                return i

    # ── 依存解析 + subword 对齐 ──────────────────────────────────────
    def _build_dep_fields(
        self,
        context:                  list,
        marked_context:           list,
        old_tok_to_new_tok_index: list,
        enc:                      object,
        seq_len:                  int,
    ):
        dep_heads = list(range(seq_len))
        dep_rels  = ['none'] * seq_len

        plain_text = " ".join(context)
        try:
            doc = self.syntax_provider.nlp(plain_text)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[dep_parse] spaCy 解析失败: {e}")
            return dep_heads, dep_rels

        char_start_to_word_idx = {}
        char_pos = 0
        for w_i, tok in enumerate(context):
            char_start_to_word_idx[char_pos] = w_i
            char_pos += len(tok) + 1

        word_dep = {}
        for spacy_tok in doc:
            w_i = char_start_to_word_idx.get(spacy_tok.idx, None)
            if w_i is None:
                continue
            head_w_i = char_start_to_word_idx.get(spacy_tok.head.idx, w_i)
            word_dep[w_i] = (head_w_i, spacy_tok.dep_)

        for w_i, (head_w_i, rel) in word_dep.items():
            if w_i >= len(old_tok_to_new_tok_index):
                continue
            sw_range   = old_tok_to_new_tok_index[w_i]
            sw_s, sw_e = sw_range[0], sw_range[1]
            if head_w_i < len(old_tok_to_new_tok_index):
                head_sw = old_tok_to_new_tok_index[head_w_i][0]
            else:
                head_sw = sw_s
            for sw in range(sw_s, min(sw_e, seq_len)):
                dep_heads[sw] = min(head_sw, seq_len - 1)
                dep_rels[sw]  = rel

        return dep_heads, dep_rels

    # ── 共指解析 + subword 对齐（新增）──────────────────────────────
    def _build_coref_clusters(self, context, old_tok_to_new_tok_index, seq_len):
        if not hasattr(self, 'coref_provider') or not self.coref_provider.enabled:
            return [], []
        plain_text = ' '.join(context)
        try:
            clusters, logits = self.coref_provider.get_coref_clusters(
                text=plain_text,
                context=context,
                old_tok_to_new_tok_index=old_tok_to_new_tok_index,
                seq_len=seq_len,
            )
            return clusters, logits
        except Exception as e:
            logger.warning(f"[_build_coref_clusters] 失败: {e}")
            return [], []

    # ── 主转换函数 ────────────────────────────────────────────────────
    def convert_examples_to_features(self, examples, role_name_mapping=None):
        features  = []
        over_nums = 0

        if self.prompt_query:
            prompts = self._read_prompt_group(self.args.prompt_path)

        if os.environ.get("DEBUG", False):
            counter = [0, 0, 0]

        for example in examples:
            example_id           = example.doc_id
            context              = example.context
            event_type_2_events  = example.event_type_2_events

            list_event_type = []
            triggers        = []
            for event_type, events in event_type_2_events.items():
                list_event_type += [e['event_type'] for e in events]
                triggers        += [tuple(e['trigger']) for e in events]

            set_triggers = sorted(list(set(triggers)))

            trigger_overlap = False
            for t1 in set_triggers:
                for t2 in set_triggers:
                    if t1[0] == t2[0] and t1[1] == t2[1]:
                        continue
                    if ((t1[0] < t2[1] and t2[0] < t1[1]) or
                            (t2[0] < t1[1] and t1[0] < t2[1])):
                        trigger_overlap = True
                        break
            if trigger_overlap:
                print('[trigger_overlap]', event_type_2_events)
                exit(0)

            # ── marked_context ────────────────────────────────────────
            offset         = 0
            marked_context = deepcopy(context)
            marker_indice  = list(range(len(triggers)))
            for i, t in enumerate(set_triggers):
                t_start = t[0]
                t_end   = t[1]
                marked_context = (
                    marked_context[:(t_start + offset)]
                    + ['<t-%d>' % marker_indice[i]]
                    + context[t_start: t_end]
                    + ['</t-%d>' % marker_indice[i]]
                    + context[t_end:]
                )
                offset += 2
            enc_text = " ".join(marked_context)

            # ── old_tok_to_new_tok_index ──────────────────────────────
            old_tok_to_char_index    = []
            old_tok_to_new_tok_index = []

            curr         = 0
            enc          = self.tokenizer(enc_text, add_special_tokens=True)
            trigger_list = [[] for _ in range(len(triggers))]
            for tok in marked_context:
                if tok not in EXTERNAL_TOKENS:
                    old_tok_to_char_index.append([curr, curr + len(tok) - 1])
                curr += len(tok) + 1

            enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
            if len(enc_input_ids) > self.args.max_enc_seq_length:
                raise ValueError(
                    f"Please increase max_enc_seq_length above {len(enc_input_ids)}"
                )

            all_ids      = enc_input_ids.copy()
            all_mask_ids = enc_mask_ids.copy()
            type_ids     = enc_mask_ids.copy()

            offset_prompt = len(enc_input_ids)

            while len(enc_input_ids) < self.args.max_enc_seq_length:
                enc_input_ids.append(self.tokenizer.pad_token_id)
                enc_mask_ids.append(self.args.pad_mask_token)

            for old_tok_idx, (char_idx_s, char_idx_e) in enumerate(
                    old_tok_to_char_index):
                new_tok_s = enc.char_to_token(char_idx_s)
                new_tok_e = enc.char_to_token(char_idx_e) + 1
                old_tok_to_new_tok_index.append([new_tok_s, new_tok_e])

            trigger_enc_token_index = []
            for t in triggers:
                new_t_start = old_tok_to_new_tok_index[t[0]][0]
                new_t_end   = old_tok_to_new_tok_index[t[1] - 1][1]
                trigger_enc_token_index.append([new_t_start, new_t_end])
            for ii, it in enumerate(trigger_enc_token_index):
                type_ids[it[0] - 1] = ii + 2

            dec_table_ids  = []
            dec_table_mask = []

            list_arg_2_prompt_slots      = []
            list_num_prompt_slots        = []
            list_dec_prompt_ids          = []
            list_arg_2_prompt_slot_spans = []
            offset_prompt_               = 0
            kk                           = 0
            enc_attention_mask           = torch.zeros(
                (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
                dtype=torch.float32,
            )

            for i, event_type in enumerate(event_type_2_events):
                events     = event_type_2_events[event_type]
                event_name = event_type.split('.')
                event_name = ['<e-%d>' % i] + event_name + ['</e-%d>' % i]
                for event in events:
                    enc_trigger_start = trigger_enc_token_index[kk][0] - 1
                    enc_trigger_end   = trigger_enc_token_index[kk][1] + 1
                    kk += 1
                    dec_prompt_text = prompts[event_type].strip()
                    assert dec_prompt_text
                    dec_prompt_text = ' '.join(event_name) + ' ' + dec_prompt_text
                    dec_prompt      = self.tokenizer(
                        dec_prompt_text, add_special_tokens=True
                    )
                    dec_prompt_ids_, dec_prompt_mask_ids_ = (
                        dec_prompt["input_ids"],
                        dec_prompt["attention_mask"],
                    )

                    arg_list = self.argument_dict[
                        event_type.replace(':', '.')
                    ]
                    arg_2_prompt_slots      = dict()
                    arg_2_prompt_slot_spans = dict()
                    num_prompt_slots        = 0
                    for arg in arg_list:
                        prompt_slots      = {
                            "tok_s": [], "tok_e": [],
                            "tok_s_off": [], "tok_e_off": [],
                        }
                        prompt_slot_spans = []
                        if role_name_mapping is not None:
                            arg_ = role_name_mapping[event_type][arg]
                        else:
                            arg_ = arg
                        for matching_result in re.finditer(
                            r'\b' + re.escape(arg_) + r'\b',
                            dec_prompt_text.split('.')[0],
                        ):
                            char_idx_s, char_idx_e = matching_result.span()
                            char_idx_e -= 1
                            tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
                            tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
                            prompt_slot_spans.append((tok_prompt_s, tok_prompt_e))
                            prompt_slots["tok_s"].append(
                                tok_prompt_s + offset_prompt_
                            )
                            prompt_slots["tok_e"].append(
                                tok_prompt_e + offset_prompt_
                            )
                            prompt_slots["tok_s_off"].append(
                                tok_prompt_s + offset_prompt + offset_prompt_
                            )
                            prompt_slots["tok_e_off"].append(
                                tok_prompt_e + offset_prompt + offset_prompt_
                            )
                            num_prompt_slots += 1
                        arg_2_prompt_slots[arg]      = prompt_slots
                        arg_2_prompt_slot_spans[arg] = prompt_slot_spans

                    list_arg_2_prompt_slots.append(arg_2_prompt_slots)
                    list_num_prompt_slots.append(num_prompt_slots)
                    list_dec_prompt_ids.append(dec_prompt_ids_)
                    list_arg_2_prompt_slot_spans.append(arg_2_prompt_slot_spans)

                    enc_attention_mask[
                        0,
                        enc_trigger_start:enc_trigger_end,
                        offset_prompt + offset_prompt_:
                        offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                    ] = 1
                    enc_attention_mask[
                        0,
                        offset_prompt + offset_prompt_:
                        offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                        enc_trigger_start:enc_trigger_end,
                    ] = 1
                    enc_attention_mask[
                        1,
                        enc_trigger_start:enc_trigger_end,
                        offset_prompt:offset_prompt + offset_prompt_,
                    ] = 1
                    enc_attention_mask[
                        1,
                        enc_trigger_start:enc_trigger_end,
                        offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
                    ] = 1
                    enc_attention_mask[
                        1,
                        offset_prompt:offset_prompt + offset_prompt_,
                        enc_trigger_start:enc_trigger_end,
                    ] = 1
                    enc_attention_mask[
                        1,
                        offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
                        enc_trigger_start:enc_trigger_end,
                    ] = 1

                enc_attention_mask[
                    0,
                    offset_prompt + offset_prompt_:
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                    offset_prompt + offset_prompt_:
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                ] = 1
                enc_attention_mask[
                    1,
                    offset_prompt + offset_prompt_:
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                    offset_prompt:offset_prompt + offset_prompt_,
                ] = 1
                enc_attention_mask[
                    1,
                    offset_prompt + offset_prompt_:
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
                ] = 1
                enc_attention_mask[
                    1,
                    offset_prompt:offset_prompt + offset_prompt_,
                    offset_prompt + offset_prompt_:
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                ] = 1
                enc_attention_mask[
                    1,
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
                    offset_prompt + offset_prompt_:
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                ] = 1

                offset_prompt_ += len(dec_prompt_ids_)
                dec_table_ids   += dec_prompt_ids_
                dec_table_mask  += dec_prompt_mask_ids_

            all_ids.extend(dec_table_ids)
            all_mask_ids.extend(dec_table_mask)
            if len(all_ids) > self.args.max_enc_seq_length:
                over_nums += 1

            while len(all_ids) < self.args.max_enc_seq_length:
                all_ids.append(self.tokenizer.pad_token_id)
                all_mask_ids.append(self.args.pad_mask_token)

            # ── 依存解析 ──────────────────────────────────────────────
            actual_seq_len = min(
                offset_prompt + offset_prompt_,
                self.args.max_enc_seq_length,
            )
            dep_heads, dep_rels = self._build_dep_fields(
                context=context,
                marked_context=marked_context,
                old_tok_to_new_tok_index=old_tok_to_new_tok_index,
                enc=enc,
                seq_len=actual_seq_len,
            )

            # ── 意群 group id ─────────────────────────────────────────
            token_chunk_ids = example.token_chunk_ids
            event_groups    = []
            for trig in triggers:
                trig_start = trig[0]
                if token_chunk_ids and trig_start < len(token_chunk_ids):
                    event_groups.append(token_chunk_ids[trig_start])
                else:
                    event_groups.append(0)

            # ── 共指解析（新增）───────────────────────────────────────
            # 使用原始 context（无 marker）进行 AllenNLP 共指消解
            # 得到 subword 粒度的共指簇，直接对应 DyGMA 图构建的输入格式
            if self._coref_sample_count==0:
                if not hasattr(self, '_coref_sample_count'):
                    self._coref_sample_count = 0
                self._coref_sample_count += 1
                print(f"[Coref] 第 {self._coref_sample_count} 条样本: {example_id}")
            coref_clusters, coref_logits = self._build_coref_clusters(
                    context=context,
                    old_tok_to_new_tok_index=old_tok_to_new_tok_index,
                    seq_len=actual_seq_len,
                )
            # print(f"[Coref] 第 {self._coref_sample_count} 条样本共指簇数: {len(coref_clusters)}")
            # for ci, cluster in enumerate(coref_clusters):
            #     print(f"[Coref]   簇{ci}: {cluster}")

            # ── 处理 target arguments（与原逻辑完全一致）────────────────
            row_index        = 0
            list_trigger_pos = []
            list_arg_slots   = []
            list_target_info = []
            list_roles       = []
            k                = 0

            for i, (event_type, events) in enumerate(
                    event_type_2_events.items()):
                for event in events:
                    arg_2_prompt_slots = list_arg_2_prompt_slots[k]
                    num_prompt_slots   = list_num_prompt_slots[k]
                    dec_prompt_ids_    = list_dec_prompt_ids[k]
                    k         += 1
                    row_index += 1

                    list_trigger_pos.append(len(dec_table_ids))
                    arg_slots = []
                    cursor    = len(dec_table_ids) + 1
                    event_args      = event['args']
                    event_args_name = [arg[-1] for arg in event_args]
                    target_info     = dict()

                    for arg, prompt_slots in arg_2_prompt_slots.items():
                        num_slots = len(prompt_slots['tok_s'])
                        arg_slots.append(
                            [cursor + x for x in range(num_slots)]
                        )
                        cursor += num_slots

                        arg_target = {"text": [], "span_s": [], "span_e": []}
                        if arg in event_args_name:
                            if os.environ.get("DEBUG", False):
                                counter[0] += 1
                            arg_idxs = [
                                j for j, x in enumerate(event_args_name)
                                if x == arg
                            ]
                            if os.environ.get("DEBUG", False):
                                counter[1] += len(arg_idxs)
                            for arg_idx in arg_idxs:
                                event_arg_info  = event_args[arg_idx]
                                answer_text     = event_arg_info[2]
                                start_old, end_old = (
                                    event_arg_info[0], event_arg_info[1]
                                )
                                start_position = old_tok_to_new_tok_index[
                                    start_old
                                ][0]
                                end_position   = old_tok_to_new_tok_index[
                                    end_old - 1
                                ][1]
                                arg_target["text"].append(answer_text)
                                arg_target["span_s"].append(start_position)
                                arg_target["span_e"].append(end_position)

                        target_info[arg] = arg_target

                    assert sum(
                        [len(slots) for slots in arg_slots]
                    ) == num_prompt_slots
                    list_arg_slots.append(arg_slots)
                    list_target_info.append(target_info)
                    roles = self.argument_dict[
                        event_type.replace(':', '.')
                    ]
                    assert len(roles) == len(arg_slots)
                    list_roles.append(roles)

            max_dec_seq_len = self.args.max_prompt_seq_length
            while len(dec_table_ids) < max_dec_seq_len:
                dec_table_ids.append(self.tokenizer.pad_token_id)
                dec_table_mask.append(self.args.pad_mask_token)

            if len(all_ids) > self.args.max_enc_seq_length:
                enc_attention_mask = torch.zeros(
                    (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
                    dtype=torch.float32,
                )
            if len(all_ids) > self.args.max_enc_seq_length:
                all_ids      = all_ids[:self.args.max_enc_seq_length]
                all_mask_ids = all_mask_ids[:self.args.max_enc_seq_length]
            if len(list_arg_2_prompt_slots) == 1:
                enc_attention_mask = torch.zeros(
                    (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
                    dtype=torch.float32,
                )

            feature_idx = len(features)
            features.append(
                InputFeatures(
                    example_id, feature_idx,
                    list_event_type,
                    trigger_enc_token_index,
                    enc_text, enc_input_ids, enc_mask_ids,
                    all_ids, all_mask_ids,
                    dec_prompt_text, dec_table_ids, dec_table_mask,
                    None,
                    list_arg_2_prompt_slots, list_target_info,
                    enc_attention_mask,
                    old_tok_to_new_tok_index=old_tok_to_new_tok_index,
                    full_text=example.context,
                    arg_list=list_roles,
                    dep_heads=dep_heads,
                    dep_rels=dep_rels,
                    event_groups=event_groups,
                    coref_clusters=coref_clusters,   # ← 新增
                    coref_logits=coref_logits,
                )
            )

        print(over_nums)
        if os.environ.get("DEBUG", False):
            print(
                '\033[91m'
                + f"distinct/tot arg_role: {counter[0]}/{counter[1]} ({counter[2]})"
                + '\033[0m'
            )
        return features

    def convert_features_to_dataset(self, features):
        return ArgumentExtractionDataset(features)

# #版本5：句法依赖+共指+事件模式
# import os
# import re
# import sys

# sys.path.append("../")
# import torch
# from copy import deepcopy
# from torch.utils.data import Dataset
# from processors.processor_base import DSET_processor, SyntaxProvider
# from utils import EXTERNAL_TOKENS, _PREDEFINED_QUERY_TEMPLATE


# # ─────────────────────────────────────────────────────────────────────────────
# # InputFeatures
# # ─────────────────────────────────────────────────────────────────────────────
# class InputFeatures(object):
#     """A single set of features of data."""

#     def __init__(self, example_id, feature_id,
#                  event_type, event_trigger,
#                  enc_text, enc_input_ids, enc_mask_ids, all_ids, all_mask_ids,
#                  dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
#                  arg_quries, arg_joint_prompt, target_info, enc_attention_mask,
#                  old_tok_to_new_tok_index=None, full_text=None, arg_list=None,
#                  dep_heads=None,
#                  dep_rels=None,
#                  event_groups=None,
#                  coref_clusters=None,   # ← 新增：list[list[int]] subword级共指簇
#                  coref_logits=None,
#                  entity_type_map=None, 
#                  event_type_map=None
#                  ):

#         self.entity_type_map = entity_type_map or {}  # {sw -> entity_type}
#         self.event_type_map  = event_type_map  or {}  # {trig_sw -> event_type}
#         self.example_id    = example_id
#         self.feature_id    = feature_id
#         self.event_type    = event_type
#         self.event_trigger = event_trigger
#         self.num_events    = len(event_trigger)

#         self.enc_text           = enc_text
#         self.enc_input_ids      = enc_input_ids
#         self.enc_mask_ids       = enc_mask_ids
#         self.all_ids            = all_ids
#         self.all_mask_ids       = all_mask_ids
#         self.enc_attention_mask = enc_attention_mask

#         self.dec_prompt_texts    = dec_prompt_text
#         self.dec_prompt_ids      = dec_prompt_ids
#         self.dec_prompt_mask_ids = dec_prompt_mask_ids

#         if arg_quries is not None:
#             self.dec_arg_query_ids       = [v[0] for k, v in arg_quries.items()]
#             self.dec_arg_query_masks     = [v[1] for k, v in arg_quries.items()]
#             self.dec_arg_start_positions = [v[2] for k, v in arg_quries.items()]
#             self.dec_arg_end_positions   = [v[3] for k, v in arg_quries.items()]
#             self.start_position_ids      = [v['span_s'] for k, v in target_info.items()]
#             self.end_position_ids        = [v['span_e'] for k, v in target_info.items()]
#         else:
#             self.dec_arg_query_ids       = None
#             self.dec_arg_query_masks     = None

#         self.arg_joint_prompt         = arg_joint_prompt
#         self.target_info              = target_info
#         self.old_tok_to_new_tok_index  = old_tok_to_new_tok_index
#         self.full_text                = full_text
#         self.arg_list                 = arg_list

#         # ── 图相关字段 ────────────────────────────────────────────────
#         self.dep_heads      = dep_heads      if dep_heads      is not None else []
#         self.dep_rels       = dep_rels       if dep_rels       is not None else []
#         self.event_groups   = event_groups   if event_groups   is not None else []
#         # coref_clusters：subword 粒度共指簇
#         # 格式：[[sw_rep_0, sw_rep_1, ...], [sw_rep_0, ...], ...]
#         # 每个子列表为同一实体的各 mention 代表 subword，第一个为超级节点
#         # self.coref_clusters = coref_clusters if coref_clusters is not None else []
#         self.coref_clusters  = coref_clusters
#         self.coref_logits    = coref_logits if coref_logits is not None else []

#     def find_idx(self, target, lst):
#         for i, item in enumerate(lst):
#             if item == target:
#                 return i

#     def init_pred(self):
#         self.pred_dict_tok  = [dict() for _ in range(self.num_events)]
#         self.pred_dict_word = [dict() for _ in range(self.num_events)]

#     def add_pred(self, role, span, event_index):
#         pred_dict_tok  = self.pred_dict_tok[event_index]
#         pred_dict_word = self.pred_dict_word[event_index]
#         if role not in pred_dict_tok:
#             pred_dict_tok[role] = list()
#         if span not in pred_dict_tok[role]:
#             pred_dict_tok[role].append(span)
#             if span != (0, 0):
#                 if role not in pred_dict_word:
#                     pred_dict_word[role] = list()
#                 word_span = self.get_word_span(span)
#                 if word_span not in pred_dict_word[role]:
#                     pred_dict_word[role].append(word_span)

#     def set_gt(self):
#         self.gt_dict_tok = [dict() for _ in range(self.num_events)]
#         for i, target_info in enumerate(self.target_info):
#             for k, v in target_info.items():
#                 self.gt_dict_tok[i][k] = [
#                     (s, e) for (s, e) in zip(v["span_s"], v["span_e"])
#                 ]

#         self.gt_dict_word = [dict() for _ in range(self.num_events)]
#         for i, gt_dict_tok in enumerate(self.gt_dict_tok):
#             gt_dict_word = self.gt_dict_word[i]
#             for role, spans in gt_dict_tok.items():
#                 for span in spans:
#                     if span != (0, 0):
#                         if role not in gt_dict_word:
#                             gt_dict_word[role] = list()
#                         word_span = self.get_word_span(span)
#                         gt_dict_word[role].append(word_span)

#     @property
#     def old_tok_index(self):
#         new_tok_index_to_old_tok_index = dict()
#         for old_tok_id, (new_tok_id_s, new_tok_id_e) in enumerate(
#                 self.old_tok_to_new_tok_index):
#             for j in range(new_tok_id_s, new_tok_id_e):
#                 new_tok_index_to_old_tok_index[j] = old_tok_id
#         return new_tok_index_to_old_tok_index

#     def get_word_span(self, span):
#         if span == (0, 0):
#             raise AssertionError()
#         offset  = 0
#         span    = list(span)
#         span[0] = min(span[0], max(self.old_tok_index.keys()))
#         span[1] = max(span[1] - 1, min(self.old_tok_index.keys()))
#         while span[0] not in self.old_tok_index:
#             span[0] += 1
#         span_s = self.old_tok_index[span[0]] + offset
#         while span[1] not in self.old_tok_index:
#             span[1] -= 1
#         span_e = self.old_tok_index[span[1]] + offset
#         while span_e < span_s:
#             span_e += 1
#         return (span_s, span_e)

#     def __repr__(self):
#         s  = ""
#         s += "example_id: {}\n".format(self.example_id)
#         s += "event_type: {}\n".format(self.event_type)
#         s += "trigger_word: {}\n".format(self.event_trigger)
#         s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
#         s += "enc_input_ids: {}\n".format(self.enc_input_ids)
#         s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
#         s += "dec_prompt_ids: {}\n".format(self.dec_prompt_ids)
#         s += "dec_prompt_mask_ids: {}\n".format(self.dec_prompt_mask_ids)
#         s += "event_groups: {}\n".format(self.event_groups)
#         s += "dep_heads[:10]: {}\n".format(self.dep_heads[:10])
#         s += "dep_rels[:10]: {}\n".format(self.dep_rels[:10])
#         s += "coref_clusters[:3]: {}\n".format(self.coref_clusters[:3])
#         return s


# # ─────────────────────────────────────────────────────────────────────────────
# # Dataset
# # ─────────────────────────────────────────────────────────────────────────────
# class ArgumentExtractionDataset(Dataset):
#     def __init__(self, features):
#         self.features = features

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx]

#     @staticmethod
#     def collate_fn(batch):
#         enc_input_ids  = torch.tensor([f.enc_input_ids  for f in batch])
#         enc_mask_ids   = torch.tensor([f.enc_mask_ids   for f in batch])
#         all_ids        = torch.tensor([f.all_ids        for f in batch])
#         all_mask_ids   = torch.tensor([f.all_mask_ids   for f in batch])

#         entity_type_map_batch = [f.entity_type_map for f in batch]
#         event_type_map_batch  = [f.event_type_map  for f in batch]

#         if batch[0].dec_prompt_ids is not None:
#             dec_prompt_ids      = torch.tensor([f.dec_prompt_ids      for f in batch])
#             dec_prompt_mask_ids = torch.tensor([f.dec_prompt_mask_ids for f in batch])
#         else:
#             dec_prompt_ids      = None
#             dec_prompt_mask_ids = None

#         example_idx = [f.example_id for f in batch]
#         feature_idx = torch.tensor([f.feature_id for f in batch])

#         if batch[0].dec_arg_query_ids is not None:
#             dec_arg_query_ids       = [torch.LongTensor(f.dec_arg_query_ids)       for f in batch]
#             dec_arg_query_mask_ids  = [torch.LongTensor(f.dec_arg_query_masks)     for f in batch]
#             dec_arg_start_positions = [torch.LongTensor(f.dec_arg_start_positions) for f in batch]
#             dec_arg_end_positions   = [torch.LongTensor(f.dec_arg_end_positions)   for f in batch]
#             start_position_ids      = [torch.FloatTensor(f.start_position_ids)     for f in batch]
#             end_position_ids        = [torch.FloatTensor(f.end_position_ids)       for f in batch]
#         else:
#             dec_arg_query_ids       = None
#             dec_arg_query_mask_ids  = None
#             dec_arg_start_positions = None
#             dec_arg_end_positions   = None
#             start_position_ids      = None
#             end_position_ids        = None

#         target_info              = [f.target_info              for f in batch]
#         old_tok_to_new_tok_index = [f.old_tok_to_new_tok_index for f in batch]
#         arg_joint_prompt         = [f.arg_joint_prompt         for f in batch]
#         arg_lists                = [f.arg_list                 for f in batch]
#         event_trigger            = [f.event_trigger            for f in batch]
#         enc_attention_mask       = [f.enc_attention_mask       for f in batch]

#         # ── 图1 字段 ──────────────────────────────────────────────────
#         dep_heads_batch    = [f.dep_heads    for f in batch]   # list[list[int]]
#         dep_rels_batch     = [f.dep_rels     for f in batch]   # list[list[str]]
#         event_groups_batch = [f.event_groups for f in batch]   # list[list[int]]

#         # ── 共指簇字段（新增）────────────────────────────────────────
#         # coref_clusters 是嵌套列表（每个样本的簇数量和每簇大小均不同）
#         # 直接以 list 传出，不 padding
#         coref_clusters_batch = [f.coref_clusters for f in batch]  # list[list[list[int]]]
#         coref_logits_batch   = [f.coref_logits   for f in batch]

#         return (
#             enc_input_ids, enc_mask_ids,
#             all_ids, all_mask_ids,
#             dec_arg_query_ids, dec_arg_query_mask_ids,
#             dec_prompt_ids, dec_prompt_mask_ids,
#             target_info, old_tok_to_new_tok_index,
#             arg_joint_prompt, arg_lists,
#             example_idx, feature_idx,
#             dec_arg_start_positions, dec_arg_end_positions,
#             start_position_ids, end_position_ids,
#             event_trigger, enc_attention_mask,
#             dep_heads_batch, dep_rels_batch, event_groups_batch,  # [20][21][22]
#             coref_clusters_batch,                                  # [23] ← 新增
#             coref_logits_batch,
#             entity_type_map_batch,   # 索引 [25]
#             event_type_map_batch,    # 索引 [26]
#         )


# # ─────────────────────────────────────────────────────────────────────────────
# # Processor
# # ─────────────────────────────────────────────────────────────────────────────
# class MultiargProcessor(DSET_processor):
#     def __init__(self, args, tokenizer):
#         super().__init__(args, tokenizer)
#         self.set_dec_input()
#         self._coref_sample_count=0
#         self.collate_fn = ArgumentExtractionDataset.collate_fn

#         if not hasattr(self, 'syntax_provider'):
#             self.syntax_provider = SyntaxProvider()
#         # coref_provider 已在 DSET_processor.__init__ 中初始化

#     def set_dec_input(self):
#         self.arg_query    = False
#         self.prompt_query = False
#         if self.args.model_type == "base":
#             self.arg_query = True
#         elif "DyGMA" in self.args.model_type:
#             self.prompt_query = True
#         else:
#             raise NotImplementedError(f"Unexpected setting {self.args.model_type}")

#     @staticmethod
#     def _read_prompt_group(prompt_path):
#         with open(prompt_path) as f:
#             lines = f.readlines()
#         prompts = dict()
#         for line in lines:
#             if not line:
#                 continue
#             event_type, prompt = line.split(":")
#             prompts[event_type] = prompt
#         return prompts

#     def create_dec_qury(self, arg, event_trigger):
#         dec_text = _PREDEFINED_QUERY_TEMPLATE.format(
#             arg=arg, trigger=event_trigger
#         )
#         dec = self.tokenizer(dec_text)
#         dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]
#         while len(dec_input_ids) < self.args.max_dec_seq_length:
#             dec_input_ids.append(self.tokenizer.pad_token_id)
#             dec_mask_ids.append(self.args.pad_mask_token)
#         matching_result = re.search(arg, dec_text)
#         char_idx_s, char_idx_e = matching_result.span()
#         char_idx_e -= 1
#         tok_prompt_s = dec.char_to_token(char_idx_s)
#         tok_prompt_e = dec.char_to_token(char_idx_e) + 1
#         return dec_input_ids, dec_mask_ids, tok_prompt_s, tok_prompt_e

#     def find_idx(self, target, lst):
#         for i, item in enumerate(lst):
#             if item == target:
#                 return i

#     # ── 依存解析 + subword 对齐 ──────────────────────────────────────
#     def _build_dep_fields(
#         self,
#         context:                  list,
#         marked_context:           list,
#         old_tok_to_new_tok_index: list,
#         enc:                      object,
#         seq_len:                  int,
#     ):
#         dep_heads = list(range(seq_len))
#         dep_rels  = ['none'] * seq_len

#         plain_text = " ".join(context)
#         try:
#             doc = self.syntax_provider.nlp(plain_text)
#         except Exception as e:
#             import logging
#             logging.getLogger(__name__).warning(f"[dep_parse] spaCy 解析失败: {e}")
#             return dep_heads, dep_rels

#         char_start_to_word_idx = {}
#         char_pos = 0
#         for w_i, tok in enumerate(context):
#             char_start_to_word_idx[char_pos] = w_i
#             char_pos += len(tok) + 1

#         word_dep = {}
#         for spacy_tok in doc:
#             w_i = char_start_to_word_idx.get(spacy_tok.idx, None)
#             if w_i is None:
#                 continue
#             head_w_i = char_start_to_word_idx.get(spacy_tok.head.idx, w_i)
#             word_dep[w_i] = (head_w_i, spacy_tok.dep_)

#         for w_i, (head_w_i, rel) in word_dep.items():
#             if w_i >= len(old_tok_to_new_tok_index):
#                 continue
#             sw_range   = old_tok_to_new_tok_index[w_i]
#             sw_s, sw_e = sw_range[0], sw_range[1]
#             if head_w_i < len(old_tok_to_new_tok_index):
#                 head_sw = old_tok_to_new_tok_index[head_w_i][0]
#             else:
#                 head_sw = sw_s
#             for sw in range(sw_s, min(sw_e, seq_len)):
#                 dep_heads[sw] = min(head_sw, seq_len - 1)
#                 dep_rels[sw]  = rel

#         return dep_heads, dep_rels

#     # ── 共指解析 + subword 对齐（新增）──────────────────────────────
#     def _build_coref_clusters(self, context, old_tok_to_new_tok_index, seq_len):
#         if not hasattr(self, 'coref_provider') or not self.coref_provider.enabled:
#             return [], []
#         plain_text = ' '.join(context)
#         try:
#             clusters, logits = self.coref_provider.get_coref_clusters(
#                 text=plain_text,
#                 context=context,
#                 old_tok_to_new_tok_index=old_tok_to_new_tok_index,
#                 seq_len=seq_len,
#             )
#             return clusters, logits
#         except Exception as e:
#             logger.warning(f"[_build_coref_clusters] 失败: {e}")
#             return [], []

#     # ── 主转换函数 ────────────────────────────────────────────────────
#     def convert_examples_to_features(self, examples, role_name_mapping=None):
#         features  = []
#         over_nums = 0

#         if self.prompt_query:
#             prompts = self._read_prompt_group(self.args.prompt_path)

#         if os.environ.get("DEBUG", False):
#             counter = [0, 0, 0]

#         for example in examples:
#             example_id           = example.doc_id
#             context              = example.context
#             event_type_2_events  = example.event_type_2_events

#             # ======================================================
#             if example_id == 0:  # 只打印第一条数据，防止刷屏
#                 print("\n" + "="*40)
#                 print("[DEBUG] example 对象包含以下属性 (Keys):")
#                 print(example.__dict__.keys())
#                 print("="*40 + "\n")
#                 # 如果你想看里面的具体内容，可以取消下面这行的注释：
#                 # print(example.__dict__) 
#             # ======================================================

            

#             list_event_type = []
#             triggers        = []
#             for event_type, events in event_type_2_events.items():
#                 list_event_type += [e['event_type'] for e in events]
#                 triggers        += [tuple(e['trigger']) for e in events]

#             set_triggers = sorted(list(set(triggers)))

#             trigger_overlap = False
#             for t1 in set_triggers:
#                 for t2 in set_triggers:
#                     if t1[0] == t2[0] and t1[1] == t2[1]:
#                         continue
#                     if ((t1[0] < t2[1] and t2[0] < t1[1]) or
#                             (t2[0] < t1[1] and t1[0] < t2[1])):
#                         trigger_overlap = True
#                         break
#             if trigger_overlap:
#                 print('[trigger_overlap]', event_type_2_events)
#                 exit(0)

#             # ── marked_context ────────────────────────────────────────
#             offset         = 0
#             marked_context = deepcopy(context)
#             marker_indice  = list(range(len(triggers)))
#             for i, t in enumerate(set_triggers):
#                 t_start = t[0]
#                 t_end   = t[1]
#                 marked_context = (
#                     marked_context[:(t_start + offset)]
#                     + ['<t-%d>' % marker_indice[i]]
#                     + context[t_start: t_end]
#                     + ['</t-%d>' % marker_indice[i]]
#                     + context[t_end:]
#                 )
#                 offset += 2
#             enc_text = " ".join(marked_context)

#             # ── old_tok_to_new_tok_index ──────────────────────────────
#             old_tok_to_char_index    = []
#             old_tok_to_new_tok_index = []

#             curr         = 0
#             enc          = self.tokenizer(enc_text, add_special_tokens=True)
#             trigger_list = [[] for _ in range(len(triggers))]
#             for tok in marked_context:
#                 if tok not in EXTERNAL_TOKENS:
#                     old_tok_to_char_index.append([curr, curr + len(tok) - 1])
#                 curr += len(tok) + 1

#             enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
#             if len(enc_input_ids) > self.args.max_enc_seq_length:
#                 raise ValueError(
#                     f"Please increase max_enc_seq_length above {len(enc_input_ids)}"
#                 )

#             all_ids      = enc_input_ids.copy()
#             all_mask_ids = enc_mask_ids.copy()
#             type_ids     = enc_mask_ids.copy()

#             offset_prompt = len(enc_input_ids)

#             while len(enc_input_ids) < self.args.max_enc_seq_length:
#                 enc_input_ids.append(self.tokenizer.pad_token_id)
#                 enc_mask_ids.append(self.args.pad_mask_token)

#             for old_tok_idx, (char_idx_s, char_idx_e) in enumerate(
#                     old_tok_to_char_index):
#                 new_tok_s = enc.char_to_token(char_idx_s)
#                 new_tok_e = enc.char_to_token(char_idx_e) + 1
#                 old_tok_to_new_tok_index.append([new_tok_s, new_tok_e])

#             trigger_enc_token_index = []
#             for t in triggers:
#                 new_t_start = old_tok_to_new_tok_index[t[0]][0]
#                 new_t_end   = old_tok_to_new_tok_index[t[1] - 1][1]
#                 trigger_enc_token_index.append([new_t_start, new_t_end])
#             for ii, it in enumerate(trigger_enc_token_index):
#                 type_ids[it[0] - 1] = ii + 2

#             dec_table_ids  = []
#             dec_table_mask = []

#             list_arg_2_prompt_slots      = []
#             list_num_prompt_slots        = []
#             list_dec_prompt_ids          = []
#             list_arg_2_prompt_slot_spans = []
#             offset_prompt_               = 0
#             kk                           = 0
#             enc_attention_mask           = torch.zeros(
#                 (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
#                 dtype=torch.float32,
#             )

#             for i, event_type in enumerate(event_type_2_events):
#                 events     = event_type_2_events[event_type]
#                 event_name = event_type.split('.')
#                 event_name = ['<e-%d>' % i] + event_name + ['</e-%d>' % i]
#                 for event in events:
#                     enc_trigger_start = trigger_enc_token_index[kk][0] - 1
#                     enc_trigger_end   = trigger_enc_token_index[kk][1] + 1
#                     kk += 1
#                     dec_prompt_text = prompts[event_type].strip()
#                     assert dec_prompt_text
#                     dec_prompt_text = ' '.join(event_name) + ' ' + dec_prompt_text
#                     dec_prompt      = self.tokenizer(
#                         dec_prompt_text, add_special_tokens=True
#                     )
#                     dec_prompt_ids_, dec_prompt_mask_ids_ = (
#                         dec_prompt["input_ids"],
#                         dec_prompt["attention_mask"],
#                     )

#                     arg_list = self.argument_dict[
#                         event_type.replace(':', '.')
#                     ]
#                     arg_2_prompt_slots      = dict()
#                     arg_2_prompt_slot_spans = dict()
#                     num_prompt_slots        = 0
#                     for arg in arg_list:
#                         prompt_slots      = {
#                             "tok_s": [], "tok_e": [],
#                             "tok_s_off": [], "tok_e_off": [],
#                         }
#                         prompt_slot_spans = []
#                         if role_name_mapping is not None:
#                             arg_ = role_name_mapping[event_type][arg]
#                         else:
#                             arg_ = arg
#                         for matching_result in re.finditer(
#                             r'\b' + re.escape(arg_) + r'\b',
#                             dec_prompt_text.split('.')[0],
#                         ):
#                             char_idx_s, char_idx_e = matching_result.span()
#                             char_idx_e -= 1
#                             tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
#                             tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
#                             prompt_slot_spans.append((tok_prompt_s, tok_prompt_e))
#                             prompt_slots["tok_s"].append(
#                                 tok_prompt_s + offset_prompt_
#                             )
#                             prompt_slots["tok_e"].append(
#                                 tok_prompt_e + offset_prompt_
#                             )
#                             prompt_slots["tok_s_off"].append(
#                                 tok_prompt_s + offset_prompt + offset_prompt_
#                             )
#                             prompt_slots["tok_e_off"].append(
#                                 tok_prompt_e + offset_prompt + offset_prompt_
#                             )
#                             num_prompt_slots += 1
#                         arg_2_prompt_slots[arg]      = prompt_slots
#                         arg_2_prompt_slot_spans[arg] = prompt_slot_spans

#                     list_arg_2_prompt_slots.append(arg_2_prompt_slots)
#                     list_num_prompt_slots.append(num_prompt_slots)
#                     list_dec_prompt_ids.append(dec_prompt_ids_)
#                     list_arg_2_prompt_slot_spans.append(arg_2_prompt_slot_spans)

#                     enc_attention_mask[
#                         0,
#                         enc_trigger_start:enc_trigger_end,
#                         offset_prompt + offset_prompt_:
#                         offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                     ] = 1
#                     enc_attention_mask[
#                         0,
#                         offset_prompt + offset_prompt_:
#                         offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                         enc_trigger_start:enc_trigger_end,
#                     ] = 1
#                     enc_attention_mask[
#                         1,
#                         enc_trigger_start:enc_trigger_end,
#                         offset_prompt:offset_prompt + offset_prompt_,
#                     ] = 1
#                     enc_attention_mask[
#                         1,
#                         enc_trigger_start:enc_trigger_end,
#                         offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
#                     ] = 1
#                     enc_attention_mask[
#                         1,
#                         offset_prompt:offset_prompt + offset_prompt_,
#                         enc_trigger_start:enc_trigger_end,
#                     ] = 1
#                     enc_attention_mask[
#                         1,
#                         offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
#                         enc_trigger_start:enc_trigger_end,
#                     ] = 1

#                 enc_attention_mask[
#                     0,
#                     offset_prompt + offset_prompt_:
#                     offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                     offset_prompt + offset_prompt_:
#                     offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                 ] = 1
#                 enc_attention_mask[
#                     1,
#                     offset_prompt + offset_prompt_:
#                     offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                     offset_prompt:offset_prompt + offset_prompt_,
#                 ] = 1
#                 enc_attention_mask[
#                     1,
#                     offset_prompt + offset_prompt_:
#                     offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                     offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
#                 ] = 1
#                 enc_attention_mask[
#                     1,
#                     offset_prompt:offset_prompt + offset_prompt_,
#                     offset_prompt + offset_prompt_:
#                     offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                 ] = 1
#                 enc_attention_mask[
#                     1,
#                     offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
#                     offset_prompt + offset_prompt_:
#                     offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
#                 ] = 1

#                 offset_prompt_ += len(dec_prompt_ids_)
#                 dec_table_ids   += dec_prompt_ids_
#                 dec_table_mask  += dec_prompt_mask_ids_

#             all_ids.extend(dec_table_ids)
#             all_mask_ids.extend(dec_table_mask)
#             if len(all_ids) > self.args.max_enc_seq_length:
#                 over_nums += 1

#             while len(all_ids) < self.args.max_enc_seq_length:
#                 all_ids.append(self.tokenizer.pad_token_id)
#                 all_mask_ids.append(self.args.pad_mask_token)

#             # ── 依存解析 ──────────────────────────────────────────────
#             actual_seq_len = min(
#                 offset_prompt + offset_prompt_,
#                 self.args.max_enc_seq_length,
#             )
#             dep_heads, dep_rels = self._build_dep_fields(
#                 context=context,
#                 marked_context=marked_context,
#                 old_tok_to_new_tok_index=old_tok_to_new_tok_index,
#                 enc=enc,
#                 seq_len=actual_seq_len,
#             )

#             entity_type_map = {}
#             for ent in getattr(example, 'entity_mentions', []):
#                 ent_type = ent['entity_type']
#                 w_s = ent['start']
#                 w_e = ent.get('end', w_s + 1) - 1

#                 if w_s < len(old_tok_to_new_tok_index):
#                     sw_s = old_tok_to_new_tok_index[w_s]
#                     if isinstance(sw_s, list) and len(sw_s) > 0:
#                         sw_s = sw_s[0]
#                     entity_type_map[sw_s] = {
#                         'type': ent_type,
#                         'is_start': True,
#                         'is_end': w_s == w_e,
#                     }

#                 if w_e >= 0 and w_e < len(old_tok_to_new_tok_index):
#                     sw_e = old_tok_to_new_tok_index[w_e]
#                     if isinstance(sw_e, list) and len(sw_e) > 0:
#                         sw_e = sw_e[-1]
#                     if sw_e in entity_type_map and isinstance(
#                         entity_type_map[sw_e], dict
#                     ):
#                         entity_type_map[sw_e]['is_end'] = True
#                     else:
#                         entity_type_map[sw_e] = {
#                             'type': ent_type,
#                             'is_start': False,
#                             'is_end': True,
#                         }

#             event_type_map = {}
#             for evt in getattr(example, 'event_mentions', []):
#                 trig_start = evt['trigger']['start']
#                 if trig_start < len(old_tok_to_new_tok_index):
#                     sw = old_tok_to_new_tok_index[trig_start]
#                     if isinstance(sw, list) and len(sw) > 0:
#                         sw = sw[0]
#                     event_type_map[sw] = evt['event_type']

#             # ── 意群 group id ─────────────────────────────────────────
#             token_chunk_ids = example.token_chunk_ids
#             event_groups    = []
#             for trig in triggers:
#                 trig_start = trig[0]
#                 if token_chunk_ids and trig_start < len(token_chunk_ids):
#                     event_groups.append(token_chunk_ids[trig_start])
#                 else:
#                     event_groups.append(0)

#             # ── 共指解析（新增）───────────────────────────────────────
#             # 使用原始 context（无 marker）进行 AllenNLP 共指消解
#             # 得到 subword 粒度的共指簇，直接对应 DyGMA 图构建的输入格式
#             if self._coref_sample_count==0:
#                 if not hasattr(self, '_coref_sample_count'):
#                     self._coref_sample_count = 0
#                 self._coref_sample_count += 1
#                 print(f"[Coref] 第 {self._coref_sample_count} 条样本: {example_id}")
#             coref_clusters, coref_logits = self._build_coref_clusters(
#                     context=context,
#                     old_tok_to_new_tok_index=old_tok_to_new_tok_index,
#                     seq_len=actual_seq_len,
#                 )
#             # print(f"[Coref] 第 {self._coref_sample_count} 条样本共指簇数: {len(coref_clusters)}")
#             # for ci, cluster in enumerate(coref_clusters):
#             #     print(f"[Coref]   簇{ci}: {cluster}")

#             # ── 处理 target arguments（与原逻辑完全一致）────────────────
#             row_index        = 0
#             list_trigger_pos = []
#             list_arg_slots   = []
#             list_target_info = []
#             list_roles       = []
#             k                = 0

#             for i, (event_type, events) in enumerate(
#                     event_type_2_events.items()):
#                 for event in events:
#                     arg_2_prompt_slots = list_arg_2_prompt_slots[k]
#                     num_prompt_slots   = list_num_prompt_slots[k]
#                     dec_prompt_ids_    = list_dec_prompt_ids[k]
#                     k         += 1
#                     row_index += 1

#                     list_trigger_pos.append(len(dec_table_ids))
#                     arg_slots = []
#                     cursor    = len(dec_table_ids) + 1
#                     event_args      = event['args']
#                     event_args_name = [arg[-1] for arg in event_args]
#                     target_info     = dict()

#                     for arg, prompt_slots in arg_2_prompt_slots.items():
#                         num_slots = len(prompt_slots['tok_s'])
#                         arg_slots.append(
#                             [cursor + x for x in range(num_slots)]
#                         )
#                         cursor += num_slots

#                         arg_target = {"text": [], "span_s": [], "span_e": []}
#                         if arg in event_args_name:
#                             if os.environ.get("DEBUG", False):
#                                 counter[0] += 1
#                             arg_idxs = [
#                                 j for j, x in enumerate(event_args_name)
#                                 if x == arg
#                             ]
#                             if os.environ.get("DEBUG", False):
#                                 counter[1] += len(arg_idxs)
#                             for arg_idx in arg_idxs:
#                                 event_arg_info  = event_args[arg_idx]
#                                 answer_text     = event_arg_info[2]
#                                 start_old, end_old = (
#                                     event_arg_info[0], event_arg_info[1]
#                                 )
#                                 start_position = old_tok_to_new_tok_index[
#                                     start_old
#                                 ][0]
#                                 end_position   = old_tok_to_new_tok_index[
#                                     end_old - 1
#                                 ][1]
#                                 arg_target["text"].append(answer_text)
#                                 arg_target["span_s"].append(start_position)
#                                 arg_target["span_e"].append(end_position)

#                         target_info[arg] = arg_target

#                     assert sum(
#                         [len(slots) for slots in arg_slots]
#                     ) == num_prompt_slots
#                     list_arg_slots.append(arg_slots)
#                     list_target_info.append(target_info)
#                     roles = self.argument_dict[
#                         event_type.replace(':', '.')
#                     ]
#                     assert len(roles) == len(arg_slots)
#                     list_roles.append(roles)

#             max_dec_seq_len = self.args.max_prompt_seq_length
#             while len(dec_table_ids) < max_dec_seq_len:
#                 dec_table_ids.append(self.tokenizer.pad_token_id)
#                 dec_table_mask.append(self.args.pad_mask_token)

#             if len(all_ids) > self.args.max_enc_seq_length:
#                 enc_attention_mask = torch.zeros(
#                     (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
#                     dtype=torch.float32,
#                 )
#             if len(all_ids) > self.args.max_enc_seq_length:
#                 all_ids      = all_ids[:self.args.max_enc_seq_length]
#                 all_mask_ids = all_mask_ids[:self.args.max_enc_seq_length]
#             if len(list_arg_2_prompt_slots) == 1:
#                 enc_attention_mask = torch.zeros(
#                     (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
#                     dtype=torch.float32,
#                 )

#             feature_idx = len(features)
#             features.append(
#                 InputFeatures(
#                     example_id, feature_idx,
#                     list_event_type,
#                     trigger_enc_token_index,
#                     enc_text, enc_input_ids, enc_mask_ids,
#                     all_ids, all_mask_ids,
#                     dec_prompt_text, dec_table_ids, dec_table_mask,
#                     None,
#                     list_arg_2_prompt_slots, list_target_info,
#                     enc_attention_mask,
#                     old_tok_to_new_tok_index=old_tok_to_new_tok_index,
#                     full_text=example.context,
#                     arg_list=list_roles,
#                     dep_heads=dep_heads,
#                     dep_rels=dep_rels,
#                     event_groups=event_groups,
#                     coref_clusters=coref_clusters,   # ← 新增
#                     coref_logits=coref_logits,
#                     entity_type_map=entity_type_map,
#                     event_type_map=event_type_map
#                 )
#             )

#         print(over_nums)
#         if os.environ.get("DEBUG", False):
#             print(
#                 '\033[91m'
#                 + f"distinct/tot arg_role: {counter[0]}/{counter[1]} ({counter[2]})"
#                 + '\033[0m'
#             )
#         return features

#     def convert_features_to_dataset(self, features):
#         return ArgumentExtractionDataset(features)
