# import csv
# import json
# import ipdb
# import jsonlines
# import torch
# import spacy
# import hashlib
# import pickle
# import os

# from random import sample
# from itertools import chain
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
# from utils import MAX_NUM_EVENTS
# import copy
# import logging

# logger = logging.getLogger(__name__)


# # ─────────────────────────────────────────────────────────────────────────────
# # 意群分组字典
# # ─────────────────────────────────────────────────────────────────────────────
# EVENT_TYPE_TO_GROUP = {
#     "Conflict:Attack": 0,        "Conflict:Demonstrate": 0,
#     "conflict.attack": 0,        "conflict.demonstrate": 0,
#     "Life:Be-Born": 1,           "Life:Marry": 1,
#     "Life:Divorce": 1,           "Life:Injure": 1,
#     "Life:Die": 1,
#     "life.born": 1,              "life.marry": 1,
#     "life.divorce": 1,           "life.injure": 1,
#     "life.die": 1,
#     "Movement:Transport": 2,     "movement.transport": 2,
#     "Transaction:Transfer-Ownership": 3,
#     "Transaction:Transfer-Money": 3,
#     "transaction.transferownership": 3,
#     "transaction.transfermoney": 3,
#     "Justice:Arrest-Jail": 4,    "Justice:Release-Parole": 4,
#     "Justice:Trial-Hearing": 4,  "Justice:Charge-Indict": 4,
#     "Justice:Sue": 4,            "Justice:Convict": 4,
#     "Justice:Sentence": 4,       "Justice:Fine": 4,
#     "Justice:Execute": 4,        "Justice:Extradite": 4,
#     "Justice:Acquit": 4,         "Justice:Appeal": 4,
#     "Justice:Pardon": 4,
#     "Personnel:Start-Position": 5, "Personnel:End-Position": 5,
#     "Personnel:Nominate": 5,       "Personnel:Elect": 5,
#     "personnel.startposition": 5,  "personnel.endposition": 5,
#     "Business:Start-Org": 6,     "Business:End-Org": 6,
#     "Business:Declare-Bankruptcy": 6, "Business:Merge-Org": 6,
#     "Contact:Meet": 7,           "Contact:Phone-Write": 7,
#     "contact.meet": 7,           "contact.phonewrite": 7,
# }
# _UNKNOWN_GROUP_START = 100
# _unknown_group_counter: dict = {}


# def _get_event_group(event_type: str) -> int:
#     key = event_type.replace(" ", "").lower()
#     if event_type in EVENT_TYPE_TO_GROUP:
#         return EVENT_TYPE_TO_GROUP[event_type]
#     for k, v in EVENT_TYPE_TO_GROUP.items():
#         if k.replace(" ", "").lower() == key:
#             return v
#     if event_type not in _unknown_group_counter:
#         _unknown_group_counter[event_type] = (
#             _UNKNOWN_GROUP_START + len(_unknown_group_counter)
#         )
#     return _unknown_group_counter[event_type]


# # ─────────────────────────────────────────────────────────────────────────────
# # 缓存 key 生成
# # ─────────────────────────────────────────────────────────────────────────────
# def _make_cache_key(args, set_type: str) -> str:
#     """
#     生成与 seed 无关的缓存 key。
#     key 由：数据文件修改时间+大小、tokenizer路径、seq长度、模型类型、数据集类型 决定。
#     seed 不影响 features 内容，不纳入 key，使 6 个 seed 共享同一份缓存。
#     """
#     if set_type == 'train':
#         data_file = args.train_file
#     elif set_type == 'dev':
#         data_file = args.dev_file
#     else:
#         data_file = args.test_file

#     try:
#         stat = os.stat(data_file)
#         file_sig = f"{stat.st_mtime}_{stat.st_size}"
#     except OSError:
#         file_sig = data_file

#     key_str = "_".join([
#         file_sig,
#         str(getattr(args, 'model_name_or_path', '')),
#         str(getattr(args, 'max_enc_seq_length', '')),
#         str(getattr(args, 'max_prompt_seq_length', '')),
#         str(getattr(args, 'model_type', '')),
#         str(getattr(args, 'dataset_type', '')),
#         str(getattr(args, 'role_path', '')),
#         str(getattr(args, 'prompt_path', '')),
#         set_type,
#     ])
#     return hashlib.md5(key_str.encode()).hexdigest()[:16]


# def _get_cache_path(args, set_type: str) -> str:
#     """
#     缓存文件存放位置：output_dir 的上两级目录下的 feature_cache/。
    
#     原因：output_dir 是 exps/wikievent/<seed>/<lr>/，不同 seed 目录不同，
#     若缓存存在 output_dir 内则每个 seed 各存一份，失去共享意义。
#     存到上级公共目录 exps/wikievent/feature_cache/ 则所有 seed 共享。
#     """
#     # exps/wikievent/<seed>/<lr>  →  上溯两级  →  exps/wikievent/
#     seed_dir    = os.path.dirname(args.output_dir)   # exps/wikievent/<seed>
#     dataset_dir = os.path.dirname(seed_dir)          # exps/wikievent/
#     cache_dir   = os.path.join(dataset_dir, "feature_cache")
#     os.makedirs(cache_dir, exist_ok=True)
#     key = _make_cache_key(args, set_type)
#     return os.path.join(cache_dir, f"features_{set_type}_{key}.pkl")


# # ─────────────────────────────────────────────────────────────────────────────
# # SyntaxProvider
# # ─────────────────────────────────────────────────────────────────────────────
# class SyntaxProvider:
#     DEPREL_WEIGHTS = {
#         "nsubj": 2.5,  "nsubjpass": 2.5,
#         "obj":   2.5,  "dobj":      2.5,  "iobj": 2.0,
#         "csubj": 2.0,  "ccomp":     1.8,  "xcomp": 1.8,
#         "root":  2.5,  "aux":       1.5,  "auxpass": 1.5,
#         "nmod":  1.2,  "amod":      1.0,  "advmod": 1.0,
#         "nummod":1.0,  "appos":     1.2,
#         "advcl": 0.8,  "acl":       0.8,  "relcl":  0.8,
#         "prep":  0.8,  "pobj":      1.0,
#     }
#     DEFAULT_WEIGHT = 0.5

#     def __init__(self):
#         print("[SyntaxProvider] 加载 en_core_web_trf ...")
#         self.nlp = spacy.load("en_core_web_trf")

#     def get_weighted_mask(self, text, trigger_span, tokenizer, max_len):
#         doc = self.nlp(text)
#         mask = torch.zeros(max_len)
#         target_node = None
#         for token in doc:
#             if (token.idx >= trigger_span[0]
#                     and (token.idx + len(token.text)) <= trigger_span[1]):
#                 target_node = token
#                 break
#         if target_node:
#             encoding = tokenizer(
#                 text, return_offsets_mapping=True,
#                 max_length=max_len, truncation=True
#             )
#             offsets = encoding['offset_mapping']
#             for child in target_node.children:
#                 weight = self.DEPREL_WEIGHTS.get(child.dep_, 0.0)
#                 if weight > 0:
#                     c_start, c_end = child.idx, child.idx + len(child.text)
#                     for i, (tok_start, tok_end) in enumerate(offsets):
#                         if tok_start >= c_start and tok_end <= c_end:
#                             mask[i] = weight
#         return mask


# # ─────────────────────────────────────────────────────────────────────────────
# # 数据结构
# # ─────────────────────────────────────────────────────────────────────────────
# class Events:
#     def __init__(self, doc_id, context, event_type_2_events):
#         self.doc_id = doc_id
#         self.context = context
#         self.event_type_2_events = event_type_2_events


# class InputFeatures(object):
#     def __init__(self, example_id, feature_id,
#                  enc_text, dec_text,
#                  enc_tokens, dec_tokens,
#                  old_tok_to_new_tok_index,
#                  event_type, event_trigger, argument_type,
#                  enc_input_ids, enc_mask_ids,
#                  dec_input_ids, dec_mask_ids,
#                  answer_text, start_position=None, end_position=None):
#         self.example_id = example_id
#         self.feature_id = feature_id
#         self.enc_text = enc_text
#         self.dec_text = dec_text
#         self.enc_tokens = enc_tokens
#         self.dec_tokens = dec_tokens
#         self.old_tok_to_new_tok_index = old_tok_to_new_tok_index
#         self.event_type = event_type
#         self.event_trigger = event_trigger
#         self.argument_type = argument_type
#         self.enc_input_ids = enc_input_ids
#         self.enc_mask_ids = enc_mask_ids
#         self.dec_input_ids = dec_input_ids
#         self.dec_mask_ids = dec_mask_ids
#         self.answer_text = answer_text
#         self.start_position = start_position
#         self.end_position = end_position

#     def __repr__(self):
#         s = ""
#         s += "example_id: {}\n".format(self.example_id)
#         s += "event_type: {}\n".format(self.event_type)
#         s += "trigger_word: {}\n".format(self.event_trigger)
#         s += "argument_type: {}\n".format(self.argument_type)
#         s += "enc_tokens: {}\n".format(self.enc_tokens)
#         s += "dec_tokens: {}\n".format(self.dec_tokens)
#         s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
#         s += "enc_input_ids: {}\n".format(self.enc_input_ids)
#         s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
#         s += "dec_input_ids: {}\n".format(self.dec_input_ids)
#         s += "dec_mask_ids: {}\n".format(self.dec_mask_ids)
#         s += "answer_text: {}\n".format(self.answer_text)
#         s += "start_position: {}\n".format(self.start_position)
#         s += "end_position: {}\n".format(self.end_position)
#         return s


# # ─────────────────────────────────────────────────────────────────────────────
# # 主处理器
# # ─────────────────────────────────────────────────────────────────────────────
# class DSET_processor:
#     def __init__(self, args, tokenizer):
#         self.args = args
#         self.tokenizer = tokenizer
#         self.template_dict, self.argument_dict = self._read_roles(self.args.role_path)
#         self.collate_fn = None

#         # 句法解析器：全局单例，所有 set_type 共享同一个已加载的模型
#         # 注意：缓存命中时不会调用 nlp，但仍需初始化以备 miss 时使用
#         self.syntax_provider = SyntaxProvider()

#     def _read_jsonlines(self, input_file):
#         lines = []
#         with jsonlines.open(input_file) as reader:
#             for obj in reader:
#                 lines.append(obj)
#         return lines

#     def _read_json(self, input_file):
#         with open(input_file, "r", encoding='utf-8') as f:
#             return json.load(f)

#     def _read_roles(self, role_path):
#         template_dict = {}
#         role_dict = {}
#         if 'MLEE' in role_path:
#             with open(role_path) as f:
#                 role_name_mapping = json.load(f)
#                 for event_type, mapping in role_name_mapping.items():
#                     roles = list(mapping.keys())
#                     role_dict[event_type] = roles
#             return None, role_dict

#         with open(role_path, "r", encoding='utf-8') as f:
#             csv_reader = csv.reader(f)
#             for line in csv_reader:
#                 event_type_arg, template = line
#                 template_dict[event_type_arg] = template
#                 event_type, arg = event_type_arg.split('_')
#                 if event_type not in role_dict:
#                     role_dict[event_type] = []
#                 role_dict[event_type].append(arg)
#         return template_dict, role_dict

#     def _create_example(self, lines, over_sample=None, max_num_event=MAX_NUM_EVENTS):
#         W = self.args.window_size
#         examples = []
#         for line in lines:
#             doc_id = line["id"]
#             context = line['context']
#             events = line["events"]
#             num_events = len(events)
#             if num_events < 1:
#                 print('[num_events < 1]', doc_id)
#                 continue

#             events = sorted(events, key=lambda x: x['trigger'])
#             context_length = len(context)
#             if context_length > W:
#                 for event in events:
#                     self.invalid_arg_num += len(event['args'])
#                 print('[context_length > W] %s\t\t%d' % (doc_id, context_length))
#                 continue

#             if num_events > max_num_event:
#                 for event in events[max_num_event:]:
#                     self.invalid_arg_num += len(event['args'])
#                 events = events[:max_num_event]
#                 print('[num_events > max_num_event] %s\t\t%d' % (doc_id, num_events))

#             assert len(events) <= MAX_NUM_EVENTS

#             if self.args.single:
#                 for event in events:
#                     event_type = event['event_type']
#                     event_type_2_events = {event_type: [event]}
#                     examples.append(Events(doc_id, context, event_type_2_events))
#                 continue

#             event_type_2_events = dict()
#             for event in events:
#                 event_type = event['event_type']
#                 if event_type not in event_type_2_events:
#                     event_type_2_events[event_type] = [event]
#                 else:
#                     event_type_2_events[event_type].append(event)

#             examples.append(Events(doc_id, context, event_type_2_events))

#             if over_sample == 'double' and num_events > 1:
#                 examples.append(Events(doc_id, context, event_type_2_events))
#             elif over_sample == 'power' and num_events > 1:
#                 power_set = []

#                 def dfs(tmp, n):
#                     if len(tmp) > 1:
#                         power_set.append(tmp[:])
#                     for i in range(n, num_events):
#                         tmp.append(events[i])
#                         dfs(tmp, i + 1)
#                         tmp.pop()

#                 dfs([], 0)
#                 for i, events_ in enumerate(power_set):
#                     event_type_2_events_ = dict()
#                     for event in events_:
#                         etype = event['event_type']
#                         if etype not in event_type_2_events_:
#                             event_type_2_events_[etype] = [event]
#                         else:
#                             event_type_2_events_[etype].append(event)
#                     examples.append(Events('%d-%s' % (i, doc_id), context, event_type_2_events_))

#         logger.info("{} examples collected. {} arguments dropped.".format(
#             len(examples), self.invalid_arg_num))
#         print("{} examples collected. {} arguments dropped.".format(
#             len(examples), self.invalid_arg_num))
#         return examples

#     def create_example(self, file_path, set_type):
#         self.invalid_arg_num = 0
#         lines = self._read_jsonlines(file_path)
#         if self.args.dataset_type == 'MLEE':
#             return self._create_example(lines, over_sample=None)
#         elif self.args.dataset_type == 'rams':
#             return self._create_example(lines, over_sample=('power' if set_type == 'train' else None))
#         elif self.args.dataset_type == 'wikievent':
#             return self._create_example(lines, over_sample=('double' if set_type == 'train' else None))
#         else:
#             raise NotImplementedError()

#     def convert_examples_to_features(self, examples, role_name_mapping=None):
#         raise NotImplementedError("子类实现")

#     def convert_features_to_dataset(self, features):
#         raise NotImplementedError("子类实现")

#     def generate_dataloader(self, set_type):
#         assert set_type in ['train', 'dev', 'test']

#         if set_type == 'train':
#             file_path = self.args.train_file
#         elif set_type == 'dev':
#             file_path = self.args.dev_file
#         else:
#             file_path = self.args.test_file

#         # ════════════════════════════════════════════════════════════════
#         # 缓存逻辑
#         # 缓存存放在 exps/wikievent/feature_cache/（seed 无关的公共目录）
#         # 6 个 seed 第一次运行时各自检查，只有第一个会触发解析，其余命中缓存
#         # ════════════════════════════════════════════════════════════════
#         cache_path = _get_cache_path(self.args, set_type)

#         if os.path.exists(cache_path):
#             print(f"[Cache] ✓ 命中缓存，跳过解析，直接加载 {set_type}: {cache_path}")
#             logger.info(f"[Cache] Loading {set_type} features from cache: {cache_path}")
#             with open(cache_path, 'rb') as f:
#                 cached = pickle.load(f)
#             examples             = cached['examples']
#             features             = cached['features']
#             self.invalid_arg_num = cached.get('invalid_arg_num', 0)

#         else:
#             print(f"[Cache] ✗ 未命中缓存，开始解析 {set_type} 数据（含 spaCy 句法解析）...")
#             logger.info(f"[Cache] Building {set_type} features (spaCy parsing included) ...")

#             examples = self.create_example(file_path, set_type)

#             # few-shot 采样在解析前完成，保证缓存内容和采样后一致
#             if set_type == 'train' and self.args.keep_ratio < 1.0:
#                 sample_num = int(len(examples) * self.args.keep_ratio)
#                 examples = sample(examples, sample_num)
#                 logger.info(
#                     "Few shot: keep ratio {}. {} training samples kept.".format(
#                         self.args.keep_ratio, len(examples))
#                 )

#             features = self.convert_examples_to_features(
#                 examples, getattr(self.args, 'role_name_mapping', None)
#             )

#             with open(cache_path, 'wb') as f:
#                 pickle.dump({
#                     'examples':        examples,
#                     'features':        features,
#                     'invalid_arg_num': self.invalid_arg_num,
#                 }, f, protocol=pickle.HIGHEST_PROTOCOL)
#             print(f"[Cache] 已保存缓存 → {cache_path}")
#             logger.info(f"[Cache] Saved {set_type} features to {cache_path}")

#         # DataLoader 每次重新构建（sampler 需要新实例，且不应被缓存）
#         dataset = self.convert_features_to_dataset(features)

#         dataset_sampler = (
#             RandomSampler(dataset) if set_type == 'train'
#             else SequentialSampler(dataset)
#         )

#         if self.collate_fn:
#             dataloader = DataLoader(
#                 dataset,
#                 sampler=dataset_sampler,
#                 batch_size=self.args.batch_size,
#                 collate_fn=self.collate_fn,
#             )
#         else:
#             dataloader = DataLoader(
#                 dataset,
#                 sampler=dataset_sampler,
#                 batch_size=self.args.batch_size,
#             )

#         return examples, features, dataloader, self.invalid_arg_num

#  ##版本2：采用触发词相似度来构建事件之间的关联
# import csv
# import json
# import ipdb
# import jsonlines
# import torch
# import spacy
# import hashlib
# import pickle
# import os

# from random import sample
# from itertools import chain
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
# from utils import MAX_NUM_EVENTS
# import copy
# import logging

# logger = logging.getLogger(__name__)


# # ─────────────────────────────────────────────────────────────────────────────
# # 意群分组字典
# # ─────────────────────────────────────────────────────────────────────────────
# # 意群信息直接从数据集的 token_chunk_ids 字段读取，不再自动生成
# # token_chunk_ids[i] 是第 i 个 token 所属的 chunk id（即意群 id）
# # 每个事件触发词的 group id = token_chunk_ids[trigger_start]


# # ─────────────────────────────────────────────────────────────────────────────
# # 缓存 key 生成
# # ─────────────────────────────────────────────────────────────────────────────
# def _make_cache_key(args, set_type: str) -> str:
#     """
#     生成与 seed 无关的缓存 key。
#     key 由：数据文件修改时间+大小、tokenizer路径、seq长度、模型类型、数据集类型 决定。
#     seed 不影响 features 内容，不纳入 key，使 6 个 seed 共享同一份缓存。
#     """
#     if set_type == 'train':
#         data_file = args.train_file
#     elif set_type == 'dev':
#         data_file = args.dev_file
#     else:
#         data_file = args.test_file

#     try:
#         stat = os.stat(data_file)
#         file_sig = f"{stat.st_mtime}_{stat.st_size}"
#     except OSError:
#         file_sig = data_file

#     key_str = "_".join([
#         file_sig,
#         str(getattr(args, 'model_name_or_path', '')),
#         str(getattr(args, 'max_enc_seq_length', '')),
#         str(getattr(args, 'max_prompt_seq_length', '')),
#         str(getattr(args, 'model_type', '')),
#         str(getattr(args, 'dataset_type', '')),
#         str(getattr(args, 'role_path', '')),
#         str(getattr(args, 'prompt_path', '')),
#         set_type,
#     ])
#     return hashlib.md5(key_str.encode()).hexdigest()[:16]


# def _get_cache_path(args, set_type: str) -> str:
#     """
#     缓存文件存放位置：output_dir 的上两级目录下的 feature_cache/。
    
#     原因：output_dir 是 exps/wikievent/<seed>/<lr>/，不同 seed 目录不同，
#     若缓存存在 output_dir 内则每个 seed 各存一份，失去共享意义。
#     存到上级公共目录 exps/wikievent/feature_cache/ 则所有 seed 共享。
#     """
#     # exps/wikievent/<seed>/<lr>  →  上溯两级  →  exps/wikievent/
#     seed_dir    = os.path.dirname(args.output_dir)   # exps/wikievent/<seed>
#     dataset_dir = os.path.dirname(seed_dir)          # exps/wikievent/
#     cache_dir   = os.path.join(dataset_dir, "feature_cache")
#     os.makedirs(cache_dir, exist_ok=True)
#     key = _make_cache_key(args, set_type)
#     return os.path.join(cache_dir, f"features_{set_type}_{key}.pkl")


# # ─────────────────────────────────────────────────────────────────────────────
# # SyntaxProvider
# # ─────────────────────────────────────────────────────────────────────────────
# class SyntaxProvider:
#     DEPREL_WEIGHTS = {
#         "nsubj": 2.5,  "nsubjpass": 2.5,
#         "obj":   2.5,  "dobj":      2.5,  "iobj": 2.0,
#         "csubj": 2.0,  "ccomp":     1.8,  "xcomp": 1.8,
#         "root":  2.5,  "aux":       1.5,  "auxpass": 1.5,
#         "nmod":  1.2,  "amod":      1.0,  "advmod": 1.0,
#         "nummod":1.0,  "appos":     1.2,
#         "advcl": 0.8,  "acl":       0.8,  "relcl":  0.8,
#         "prep":  0.8,  "pobj":      1.0,
#     }
#     DEFAULT_WEIGHT = 0.5

#     def __init__(self):
#         print("[SyntaxProvider] 加载 en_core_web_trf ...")
#         self.nlp = spacy.load("en_core_web_trf")

#     def get_weighted_mask(self, text, trigger_span, tokenizer, max_len):
#         doc = self.nlp(text)
#         mask = torch.zeros(max_len)
#         target_node = None
#         for token in doc:
#             if (token.idx >= trigger_span[0]
#                     and (token.idx + len(token.text)) <= trigger_span[1]):
#                 target_node = token
#                 break
#         if target_node:
#             encoding = tokenizer(
#                 text, return_offsets_mapping=True,
#                 max_length=max_len, truncation=True
#             )
#             offsets = encoding['offset_mapping']
#             for child in target_node.children:
#                 weight = self.DEPREL_WEIGHTS.get(child.dep_, 0.0)
#                 if weight > 0:
#                     c_start, c_end = child.idx, child.idx + len(child.text)
#                     for i, (tok_start, tok_end) in enumerate(offsets):
#                         if tok_start >= c_start and tok_end <= c_end:
#                             mask[i] = weight
#         return mask


# # ─────────────────────────────────────────────────────────────────────────────
# # 数据结构
# # ─────────────────────────────────────────────────────────────────────────────
# class Events:
#     def __init__(self, doc_id, context, event_type_2_events, token_chunk_ids=None):
#         self.doc_id = doc_id
#         self.context = context
#         self.event_type_2_events = event_type_2_events
#         # 直接来自数据集的意群信息：token_chunk_ids[i] = 第 i 个 token 的意群 id
#         self.token_chunk_ids = token_chunk_ids if token_chunk_ids is not None else []


# class InputFeatures(object):
#     def __init__(self, example_id, feature_id,
#                  enc_text, dec_text,
#                  enc_tokens, dec_tokens,
#                  old_tok_to_new_tok_index,
#                  event_type, event_trigger, argument_type,
#                  enc_input_ids, enc_mask_ids,
#                  dec_input_ids, dec_mask_ids,
#                  answer_text, start_position=None, end_position=None):
#         self.example_id = example_id
#         self.feature_id = feature_id
#         self.enc_text = enc_text
#         self.dec_text = dec_text
#         self.enc_tokens = enc_tokens
#         self.dec_tokens = dec_tokens
#         self.old_tok_to_new_tok_index = old_tok_to_new_tok_index
#         self.event_type = event_type
#         self.event_trigger = event_trigger
#         self.argument_type = argument_type
#         self.enc_input_ids = enc_input_ids
#         self.enc_mask_ids = enc_mask_ids
#         self.dec_input_ids = dec_input_ids
#         self.dec_mask_ids = dec_mask_ids
#         self.answer_text = answer_text
#         self.start_position = start_position
#         self.end_position = end_position

#     def __repr__(self):
#         s = ""
#         s += "example_id: {}\n".format(self.example_id)
#         s += "event_type: {}\n".format(self.event_type)
#         s += "trigger_word: {}\n".format(self.event_trigger)
#         s += "argument_type: {}\n".format(self.argument_type)
#         s += "enc_tokens: {}\n".format(self.enc_tokens)
#         s += "dec_tokens: {}\n".format(self.dec_tokens)
#         s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
#         s += "enc_input_ids: {}\n".format(self.enc_input_ids)
#         s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
#         s += "dec_input_ids: {}\n".format(self.dec_input_ids)
#         s += "dec_mask_ids: {}\n".format(self.dec_mask_ids)
#         s += "answer_text: {}\n".format(self.answer_text)
#         s += "start_position: {}\n".format(self.start_position)
#         s += "end_position: {}\n".format(self.end_position)
#         return s


# # ─────────────────────────────────────────────────────────────────────────────
# # 主处理器
# # ─────────────────────────────────────────────────────────────────────────────
# class DSET_processor:
#     def __init__(self, args, tokenizer):
#         self.args = args
#         self.tokenizer = tokenizer
#         self.template_dict, self.argument_dict = self._read_roles(self.args.role_path)
#         self.collate_fn = None

#         # 句法解析器：全局单例，所有 set_type 共享同一个已加载的模型
#         # 注意：缓存命中时不会调用 nlp，但仍需初始化以备 miss 时使用
#         self.syntax_provider = SyntaxProvider()

#     def _read_jsonlines(self, input_file):
#         lines = []
#         with jsonlines.open(input_file) as reader:
#             for obj in reader:
#                 lines.append(obj)
#         return lines

#     def _read_json(self, input_file):
#         with open(input_file, "r", encoding='utf-8') as f:
#             return json.load(f)

#     def _read_roles(self, role_path):
#         template_dict = {}
#         role_dict = {}
#         if 'MLEE' in role_path:
#             with open(role_path) as f:
#                 role_name_mapping = json.load(f)
#                 for event_type, mapping in role_name_mapping.items():
#                     roles = list(mapping.keys())
#                     role_dict[event_type] = roles
#             return None, role_dict

#         with open(role_path, "r", encoding='utf-8') as f:
#             csv_reader = csv.reader(f)
#             for line in csv_reader:
#                 event_type_arg, template = line
#                 template_dict[event_type_arg] = template
#                 event_type, arg = event_type_arg.split('_')
#                 if event_type not in role_dict:
#                     role_dict[event_type] = []
#                 role_dict[event_type].append(arg)
#         return template_dict, role_dict

#     def _create_example(self, lines, over_sample=None, max_num_event=MAX_NUM_EVENTS):
#         W = self.args.window_size
#         examples = []
#         for line in lines:
#             doc_id = line["id"]
#             context = line['context']
#             events = line["events"]
#             token_chunk_ids = line.get('token_chunk_ids', [])
#             num_events = len(events)
#             if num_events < 1:
#                 print('[num_events < 1]', doc_id)
#                 continue

#             events = sorted(events, key=lambda x: x['trigger'])
#             context_length = len(context)
#             if context_length > W:
#                 for event in events:
#                     self.invalid_arg_num += len(event['args'])
#                 print('[context_length > W] %s\t\t%d' % (doc_id, context_length))
#                 continue

#             if num_events > max_num_event:
#                 for event in events[max_num_event:]:
#                     self.invalid_arg_num += len(event['args'])
#                 events = events[:max_num_event]
#                 print('[num_events > max_num_event] %s\t\t%d' % (doc_id, num_events))

#             assert len(events) <= MAX_NUM_EVENTS

#             if self.args.single:
#                 for event in events:
#                     event_type = event['event_type']
#                     event_type_2_events = {event_type: [event]}
#                     examples.append(Events(doc_id, context, event_type_2_events, token_chunk_ids, entity_mentions, event_mentions))
#                 continue

#             event_type_2_events = dict()
#             for event in events:
#                 event_type = event['event_type']
#                 if event_type not in event_type_2_events:
#                     event_type_2_events[event_type] = [event]
#                 else:
#                     event_type_2_events[event_type].append(event)

#             examples.append(Events(doc_id, context, event_type_2_events, token_chunk_ids, entity_mentions, event_mentions))

#             if over_sample == 'double' and num_events > 1:
#                 examples.append(Events(doc_id, context, event_type_2_events, token_chunk_ids, entity_mentions, event_mentions))
#             elif over_sample == 'power' and num_events > 1:
#                 power_set = []

#                 def dfs(tmp, n):
#                     if len(tmp) > 1:
#                         power_set.append(tmp[:])
#                     for i in range(n, num_events):
#                         tmp.append(events[i])
#                         dfs(tmp, i + 1)
#                         tmp.pop()

#                 dfs([], 0)
#                 for i, events_ in enumerate(power_set):
#                     event_type_2_events_ = dict()
#                     for event in events_:
#                         etype = event['event_type']
#                         if etype not in event_type_2_events_:
#                             event_type_2_events_[etype] = [event]
#                         else:
#                             event_type_2_events_[etype].append(event)
#                     examples.append(Events('%d-%s' % (i, doc_id), context, event_type_2_events_, token_chunk_ids))

#         logger.info("{} examples collected. {} arguments dropped.".format(
#             len(examples), self.invalid_arg_num))
#         print("{} examples collected. {} arguments dropped.".format(
#             len(examples), self.invalid_arg_num))
#         return examples

#     def create_example(self, file_path, set_type):
#         self.invalid_arg_num = 0
#         lines = self._read_jsonlines(file_path)
#         if self.args.dataset_type == 'MLEE':
#             return self._create_example(lines, over_sample=None)
#         elif self.args.dataset_type == 'rams':
#             return self._create_example(lines, over_sample=('power' if set_type == 'train' else None))
#         elif self.args.dataset_type == 'wikievent':
#             return self._create_example(lines, over_sample=('double' if set_type == 'train' else None))
#         else:
#             raise NotImplementedError()

#     def convert_examples_to_features(self, examples, role_name_mapping=None):
#         raise NotImplementedError("子类实现")

#     def convert_features_to_dataset(self, features):
#         raise NotImplementedError("子类实现")

#     def generate_dataloader(self, set_type):
#         assert set_type in ['train', 'dev', 'test']

#         if set_type == 'train':
#             file_path = self.args.train_file
#         elif set_type == 'dev':
#             file_path = self.args.dev_file
#         else:
#             file_path = self.args.test_file

#         # ════════════════════════════════════════════════════════════════
#         # 缓存逻辑
#         # 缓存存放在 exps/wikievent/feature_cache/（seed 无关的公共目录）
#         # 6 个 seed 第一次运行时各自检查，只有第一个会触发解析，其余命中缓存
#         # ════════════════════════════════════════════════════════════════
#         cache_path = _get_cache_path(self.args, set_type)

#         if os.path.exists(cache_path):
#             print(f"[Cache] ✓ 命中缓存，跳过解析，直接加载 {set_type}: {cache_path}")
#             logger.info(f"[Cache] Loading {set_type} features from cache: {cache_path}")
#             with open(cache_path, 'rb') as f:
#                 cached = pickle.load(f)
#             examples             = cached['examples']
#             features             = cached['features']
#             self.invalid_arg_num = cached.get('invalid_arg_num', 0)

#         else:
#             print(f"[Cache] ✗ 未命中缓存，开始解析 {set_type} 数据（含 spaCy 句法解析）...")
#             logger.info(f"[Cache] Building {set_type} features (spaCy parsing included) ...")

#             examples = self.create_example(file_path, set_type)

#             # few-shot 采样在解析前完成，保证缓存内容和采样后一致
#             if set_type == 'train' and self.args.keep_ratio < 1.0:
#                 sample_num = int(len(examples) * self.args.keep_ratio)
#                 examples = sample(examples, sample_num)
#                 logger.info(
#                     "Few shot: keep ratio {}. {} training samples kept.".format(
#                         self.args.keep_ratio, len(examples))
#                 )

#             features = self.convert_examples_to_features(
#                 examples, getattr(self.args, 'role_name_mapping', None)
#             )

#             with open(cache_path, 'wb') as f:
#                 pickle.dump({
#                     'examples':        examples,
#                     'features':        features,
#                     'invalid_arg_num': self.invalid_arg_num,
#                 }, f, protocol=pickle.HIGHEST_PROTOCOL)
#             print(f"[Cache] 已保存缓存 → {cache_path}")
#             logger.info(f"[Cache] Saved {set_type} features to {cache_path}")

#         # DataLoader 每次重新构建（sampler 需要新实例，且不应被缓存）
#         dataset = self.convert_features_to_dataset(features)

#         dataset_sampler = (
#             RandomSampler(dataset) if set_type == 'train'
#             else SequentialSampler(dataset)
#         )

#         if self.collate_fn:
#             dataloader = DataLoader(
#                 dataset,
#                 sampler=dataset_sampler,
#                 batch_size=self.args.batch_size,
#                 collate_fn=self.collate_fn,
#             )
#         else:
#             dataloader = DataLoader(
#                 dataset,
#                 sampler=dataset_sampler,
#                 batch_size=self.args.batch_size,
#             )

#         return examples, features, dataloader, self.invalid_arg_num

#版本4：句法依赖+共指
import csv
import json
import jsonlines
import torch
import spacy
import hashlib
import pickle
import os
from pathlib import Path

from random import sample
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils import MAX_NUM_EVENTS
import copy
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 缓存 key 生成
# ─────────────────────────────────────────────────────────────────────────────
def _make_cache_key(args, set_type: str) -> str:
    if set_type == 'train':
        data_file = args.train_file
    elif set_type == 'dev':
        data_file = args.dev_file
    else:
        data_file = args.test_file

    try:
        stat     = os.stat(data_file)
        file_sig = f"{stat.st_mtime}_{stat.st_size}"
    except OSError:
        file_sig = data_file

    key_str = "_".join([
        file_sig,
        str(getattr(args, 'model_name_or_path', '')),
        str(getattr(args, 'max_enc_seq_length', '')),
        str(getattr(args, 'max_prompt_seq_length', '')),
        str(getattr(args, 'model_type', '')),
        str(getattr(args, 'dataset_type', '')),
        str(getattr(args, 'role_path', '')),
        str(getattr(args, 'prompt_path', '')),
        set_type,
    ])
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def _get_cache_path(args, set_type: str) -> str:
    seed_dir    = os.path.dirname(args.output_dir)
    dataset_dir = os.path.dirname(seed_dir)
    cache_dir   = os.path.join(dataset_dir, "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    key = _make_cache_key(args, set_type)
    return os.path.join(cache_dir, f"features_{set_type}_{key}.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# SyntaxProvider  ── spaCy 句法解析（全局单例）
# ─────────────────────────────────────────────────────────────────────────────
class SyntaxProvider:
    DEPREL_WEIGHTS = {
        "nsubj": 2.5,  "nsubjpass": 2.5,
        "obj":   2.5,  "dobj":      2.5,  "iobj": 2.0,
        "csubj": 2.0,  "ccomp":     1.8,  "xcomp": 1.8,
        "root":  2.5,  "aux":       1.5,  "auxpass": 1.5,
        "nmod":  1.2,  "amod":      1.0,  "advmod": 1.0,
        "nummod":1.0,  "appos":     1.2,
        "advcl": 0.8,  "acl":       0.8,  "relcl":  0.8,
        "prep":  0.8,  "pobj":      1.0,
    }
    DEFAULT_WEIGHT = 0.5

    def __init__(self):
        print("[SyntaxProvider] 加载 en_core_web_trf ...")
        self.nlp = spacy.load("en_core_web_trf")

    def get_weighted_mask(self, text, trigger_span, tokenizer, max_len):
        doc  = self.nlp(text)
        mask = torch.zeros(max_len)
        target_node = None
        for token in doc:
            if (token.idx >= trigger_span[0] and
                    (token.idx + len(token.text)) <= trigger_span[1]):
                target_node = token
                break
        if target_node:
            encoding = tokenizer(
                text, return_offsets_mapping=True,
                max_length=max_len, truncation=True,
            )
            offsets = encoding['offset_mapping']
            for child in target_node.children:
                weight = self.DEPREL_WEIGHTS.get(child.dep_, 0.0)
                if weight > 0:
                    c_start, c_end = child.idx, child.idx + len(child.text)
                    for i, (tok_start, tok_end) in enumerate(offsets):
                        if tok_start >= c_start and tok_end <= c_end:
                            mask[i] = weight
        return mask

def _merge_overlapping_clusters(clusters, logits):
    """合并有公共节点的簇，取最大 logit。"""
    merged = True
    while merged:
        merged = False
        new_clusters = []
        new_logits   = []
        used = [False] * len(clusters)
        for i in range(len(clusters)):
            if used[i]:
                continue
            current       = set(clusters[i])
            current_logit = logits[i]
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                if current & set(clusters[j]):
                    current       |= set(clusters[j])
                    current_logit  = max(current_logit, logits[j])
                    used[j]        = True
                    merged         = True
            new_clusters.append(sorted(current))
            new_logits.append(current_logit)
            used[i] = True
        clusters = new_clusters
        logits   = new_logits
    return clusters, logits

class CorefProvider:
    """
    使用 fastcoref 进行共指消解，完全离线可用。
    模型从本地路径加载。

    安装：pip install fastcoref
    模型：biu-nlp/f-coref（下载后放到本地路径）
    """

    DEFAULT_FASTCOREF_MODEL_PATH = str(Path(__file__).resolve().parents[1] / "fcoref_model")

    def __init__(self, syntax_provider=None, model_path=None):
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)
        from datasets import disable_progress_bar
        disable_progress_bar()
        self.model_path = model_path or os.environ.get("DYGMA_COREF_MODEL") or self.DEFAULT_FASTCOREF_MODEL_PATH
        print(f"[CorefProvider] 加载 fastcoref 模型: {self.model_path} ...")
        try:
            from fastcoref import FCoref
            self.model = FCoref(
                model_name_or_path=self.model_path,
                device='cpu',
                nlp='en_core_web_sm',
                enable_progress_bar=False,
            )
            self.enabled = True
            print("[CorefProvider] fastcoref 加载成功。")
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.warning(f"[CorefProvider] 加载失败: {e}，共指功能已禁用。")
            self.enabled = False

    def get_coref_clusters(
        self,
        text:                     str,
        context:                  list,
        old_tok_to_new_tok_index: list,
        seq_len:                  int,
        logit_threshold:          float = 3.0,
    ) -> tuple:
        """
        返回 (clusters, cluster_logits)
        clusters:      list[list[int]]  subword 粒度共指簇
        cluster_logits: list[float]     每个簇的 max pairwise logit
        """
        FILTER_WORDS = {
            'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for',
            'with', 'by', 'from', 'and', 'or', 'but', 'as', 'that',
            'this', 'it', 'its', 'be', 'is', 'was', 'are', 'were',
            ',', '.', '!', '?', ';', ':', '"', "'", '(', ')',
        }

        if not self.enabled:
            return [], []

        if not hasattr(self, '_call_count'):
            self._call_count = 0
        self._call_count += 1
        is_first = (self._call_count == 1)

        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)

        try:
            preds         = self.model.predict(texts=[text])
            clusters_char = preds[0].get_clusters(as_strings=False)
            pred          = preds[0]
        except Exception as e:
            logger.warning(f"[CorefProvider] 推理失败: {e}")
            return [], []

        if not clusters_char:
            return [], []

        # 字符映射
        char_start_to_word_idx = {}
        char_pos = 0
        for w_i, tok in enumerate(context):
            char_start_to_word_idx[char_pos] = w_i
            char_pos += len(tok) + 1

        def char_to_sw(char_start: int) -> int:
            best_key, best_diff = None, float('inf')
            for k in char_start_to_word_idx:
                diff = char_start - k
                if 0 <= diff < best_diff:
                    best_diff, best_key = diff, k
            if best_key is None:
                return -1
            w_i = char_start_to_word_idx[best_key]
            if w_i >= len(old_tok_to_new_tok_index):
                return -1
            sw = old_tok_to_new_tok_index[w_i][0]
            return sw if sw < seq_len else -1

        if is_first:
            print("\n[CorefProvider] ── 第1条样本共指簇详情 ──")

        clusters      = []
        cluster_logits = []

        for cluster_char in clusters_char:
            # 计算簇内最大 pairwise logit
            max_logit = float('-inf')
            for i in range(len(cluster_char)):
                for j in range(len(cluster_char)):
                    if i == j:
                        continue
                    try:
                        logit = pred.get_logit(cluster_char[i], cluster_char[j])
                        if logit > max_logit:
                            max_logit = logit
                    except Exception:
                        continue

            if max_logit < logit_threshold:
                if is_first:
                    print(f"  ❌ 过滤 (logit={max_logit:.2f}): "
                        f"{[text[s:e] for s,e in cluster_char]}")
                continue

            if is_first:
                print(f"  ✓ 保留 (max_logit={max_logit:.2f}): "
                    f"{[text[s:e] for s,e in cluster_char]}")

            # 转换为 subword 索引
            sw_reps = []
            for (char_s, char_e) in cluster_char:
                sw = char_to_sw(char_s)
                if sw >= 0 and sw not in sw_reps:
                    sw_reps.append(sw)

            sw_reps_sorted = sorted(sw_reps)
            if len(sw_reps_sorted) < 2:
                continue

            # 过滤功能词超级节点
            rep_text = None
            rep_idx  = sw_reps_sorted[0]
            for (char_s, char_e) in cluster_char:
                if char_to_sw(char_s) == rep_idx:
                    rep_text = text[char_s:char_e].strip().lower()
                    break
            if rep_text and rep_text in FILTER_WORDS:
                if is_first:
                    print(f"  ⚠ 功能词超级节点过滤: '{rep_text}'")
                continue

            clusters.append(sw_reps_sorted)
            cluster_logits.append(max_logit)

        if is_first:
            print(f"[CorefProvider] 第1条样本最终簇数: {len(clusters)}\n")

        # 传递闭包合并重叠簇
        clusters, cluster_logits = _merge_overlapping_clusters(clusters, cluster_logits)
        return clusters, cluster_logits
            
  
    
    
# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────
class Events:
    def __init__(self, doc_id, context, event_type_2_events,
                 token_chunk_ids=None,):
        self.doc_id              = doc_id
        self.context             = context
        self.event_type_2_events = event_type_2_events
        self.token_chunk_ids     = token_chunk_ids if token_chunk_ids is not None else []


class InputFeatures(object):
    def __init__(self, example_id, feature_id,
                 enc_text, dec_text,
                 enc_tokens, dec_tokens,
                 old_tok_to_new_tok_index,
                 event_type, event_trigger, argument_type,
                 enc_input_ids, enc_mask_ids,
                 dec_input_ids, dec_mask_ids,
                 answer_text, start_position=None, end_position=None):
        self.example_id               = example_id
        self.feature_id               = feature_id
        self.enc_text                 = enc_text
        self.dec_text                 = dec_text
        self.enc_tokens               = enc_tokens
        self.dec_tokens               = dec_tokens
        self.old_tok_to_new_tok_index  = old_tok_to_new_tok_index
        self.event_type               = event_type
        self.event_trigger            = event_trigger
        self.argument_type            = argument_type
        self.enc_input_ids            = enc_input_ids
        self.enc_mask_ids             = enc_mask_ids
        self.dec_input_ids            = dec_input_ids
        self.dec_mask_ids             = dec_mask_ids
        self.answer_text              = answer_text
        self.start_position           = start_position
        self.end_position             = end_position

    def __repr__(self):
        s  = ""
        s += "example_id: {}\n".format(self.example_id)
        s += "event_type: {}\n".format(self.event_type)
        s += "trigger_word: {}\n".format(self.event_trigger)
        s += "argument_type: {}\n".format(self.argument_type)
        s += "enc_input_ids: {}\n".format(self.enc_input_ids)
        s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
        s += "dec_input_ids: {}\n".format(self.dec_input_ids)
        s += "dec_mask_ids: {}\n".format(self.dec_mask_ids)
        s += "answer_text: {}\n".format(self.answer_text)
        s += "start_position: {}\n".format(self.start_position)
        s += "end_position: {}\n".format(self.end_position)
        return s


# ─────────────────────────────────────────────────────────────────────────────
# 主处理器基类
# ─────────────────────────────────────────────────────────────────────────────
class DSET_processor:
    def __init__(self, args, tokenizer):
        self.args       = args
        self.tokenizer  = tokenizer
        self.template_dict, self.argument_dict = self._read_roles(self.args.role_path)
        self.collate_fn = None

        # ── 句法解析器（全局单例）────────────────────────────────────
        self.syntax_provider = SyntaxProvider()

        # ── 共指解析器（全局单例）────────────────────────────────────
        # 首次实例化时加载 fastcoref 模型；公开仓库不携带权重，需通过参数或环境变量指定。
        self.coref_provider = CorefProvider(model_path=getattr(self.args, "coref_model_path", None))

    def _read_jsonlines(self, input_file):
        lines = []
        with jsonlines.open(input_file) as reader:
            for obj in reader:
                lines.append(obj)
        return lines

    def _read_json(self, input_file):
        with open(input_file, "r", encoding='utf-8') as f:
            return json.load(f)

    def _read_roles(self, role_path):
        template_dict = {}
        role_dict     = {}
        if 'MLEE' in role_path:
            with open(role_path) as f:
                role_name_mapping = json.load(f)
                for event_type, mapping in role_name_mapping.items():
                    roles = list(mapping.keys())
                    role_dict[event_type] = roles
            return None, role_dict

        with open(role_path, "r", encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                event_type_arg, template = line
                template_dict[event_type_arg] = template
                event_type, arg = event_type_arg.split('_')
                if event_type not in role_dict:
                    role_dict[event_type] = []
                role_dict[event_type].append(arg)
        return template_dict, role_dict

    def _create_example(self, lines, over_sample=None, max_num_event=MAX_NUM_EVENTS):
        W        = self.args.window_size
        examples = []
        for line in lines:
            doc_id          = line["id"]
            context         = line['context']
            events          = line["events"]
            token_chunk_ids = line.get('token_chunk_ids', [])
            num_events      = len(events)
            if num_events < 1:
                print('[num_events < 1]', doc_id)
                continue

            events         = sorted(events, key=lambda x: x['trigger'])
            context_length = len(context)
            if context_length > W:
                for event in events:
                    self.invalid_arg_num += len(event['args'])
                print('[context_length > W] %s\t\t%d' % (doc_id, context_length))
                continue

            if num_events > max_num_event:
                for event in events[max_num_event:]:
                    self.invalid_arg_num += len(event['args'])
                events = events[:max_num_event]
                print('[num_events > max_num_event] %s\t\t%d' % (doc_id, num_events))

            assert len(events) <= MAX_NUM_EVENTS

            if self.args.single:
                for event in events:
                    event_type          = event['event_type']
                    event_type_2_events = {event_type: [event]}
                    examples.append(
                        Events(doc_id, context, event_type_2_events, token_chunk_ids)
                    )
                continue

            event_type_2_events = dict()
            for event in events:
                event_type = event['event_type']
                if event_type not in event_type_2_events:
                    event_type_2_events[event_type] = [event]
                else:
                    event_type_2_events[event_type].append(event)

            examples.append(
                Events(doc_id, context, event_type_2_events, token_chunk_ids)
            )

            if over_sample == 'double' and num_events > 1:
                examples.append(
                    Events(doc_id, context, event_type_2_events, token_chunk_ids)
                )
            elif over_sample == 'power' and num_events > 1:
                power_set = []

                def dfs(tmp, n):
                    if len(tmp) > 1:
                        power_set.append(tmp[:])
                    for i in range(n, num_events):
                        tmp.append(events[i])
                        dfs(tmp, i + 1)
                        tmp.pop()

                dfs([], 0)
                for i, events_ in enumerate(power_set):
                    event_type_2_events_ = dict()
                    for event in events_:
                        etype = event['event_type']
                        if etype not in event_type_2_events_:
                            event_type_2_events_[etype] = [event]
                        else:
                            event_type_2_events_[etype].append(event)
                    examples.append(
                        Events('%d-%s' % (i, doc_id), context,
                               event_type_2_events_, token_chunk_ids)
                    )

        logger.info("{} examples collected. {} arguments dropped.".format(
            len(examples), self.invalid_arg_num))
        print("{} examples collected. {} arguments dropped.".format(
            len(examples), self.invalid_arg_num))
        return examples

    def create_example(self, file_path, set_type):
        self.invalid_arg_num = 0
        lines = self._read_jsonlines(file_path)
        if self.args.dataset_type == 'MLEE':
            return self._create_example(lines, over_sample=None)
        elif self.args.dataset_type == 'rams':
            return self._create_example(
                lines, over_sample=('power' if set_type == 'train' else None)
            )
        elif self.args.dataset_type == 'wikievent':
            return self._create_example(
                lines, over_sample=('double' if set_type == 'train' else None)
            )
        else:
            raise NotImplementedError()

    def convert_examples_to_features(self, examples, role_name_mapping=None):
        raise NotImplementedError("子类实现")

    def convert_features_to_dataset(self, features):
        raise NotImplementedError("子类实现")

    def generate_dataloader(self, set_type):
        assert set_type in ['train', 'dev', 'test']

        if set_type == 'train':
            file_path = self.args.train_file
        elif set_type == 'dev':
            file_path = self.args.dev_file
        else:
            file_path = self.args.test_file

        cache_path = _get_cache_path(self.args, set_type)

        if os.path.exists(cache_path):
            print(f"[Cache] ✓ 命中缓存，跳过解析: {cache_path}")
            logger.info(f"[Cache] Loading {set_type} features from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            examples             = cached['examples']
            features             = cached['features']
            self.invalid_arg_num = cached.get('invalid_arg_num', 0)

        else:
            print(f"[Cache] ✗ 未命中缓存，开始解析 {set_type} 数据 ...")
            logger.info(f"[Cache] Building {set_type} features ...")

            examples = self.create_example(file_path, set_type)

            if set_type == 'train' and self.args.keep_ratio < 1.0:
                sample_num = int(len(examples) * self.args.keep_ratio)
                examples   = sample(examples, sample_num)
                logger.info(
                    "Few shot: keep ratio {}. {} training samples kept.".format(
                        self.args.keep_ratio, len(examples))
                )

            features = self.convert_examples_to_features(
                examples, getattr(self.args, 'role_name_mapping', None)
            )

            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'examples':        examples,
                    'features':        features,
                    'invalid_arg_num': self.invalid_arg_num,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[Cache] 已保存缓存 → {cache_path}")
            logger.info(f"[Cache] Saved {set_type} features to {cache_path}")

        dataset         = self.convert_features_to_dataset(features)
        dataset_sampler = (
            RandomSampler(dataset) if set_type == 'train'
            else SequentialSampler(dataset)
        )

        if self.collate_fn:
            dataloader = DataLoader(
                dataset,
                sampler=dataset_sampler,
                batch_size=self.args.batch_size,
                collate_fn=self.collate_fn,
            )
        else:
            dataloader = DataLoader(
                dataset,
                sampler=dataset_sampler,
                batch_size=self.args.batch_size,
            )

        return examples, features, dataloader, self.invalid_arg_num
    
# #版本5：句法依赖+共指+事件模式
# import csv
# import json
# import jsonlines
# import torch
# import spacy
# import hashlib
# import pickle
# import os

# from random import sample
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# from utils import MAX_NUM_EVENTS
# import copy
# import logging

# logger = logging.getLogger(__name__)


# # ─────────────────────────────────────────────────────────────────────────────
# # 缓存 key 生成
# # ─────────────────────────────────────────────────────────────────────────────
# def _make_cache_key(args, set_type: str) -> str:
#     if set_type == 'train':
#         data_file = args.train_file
#     elif set_type == 'dev':
#         data_file = args.dev_file
#     else:
#         data_file = args.test_file

#     try:
#         stat     = os.stat(data_file)
#         file_sig = f"{stat.st_mtime}_{stat.st_size}"
#     except OSError:
#         file_sig = data_file

#     key_str = "_".join([
#         file_sig,
#         str(getattr(args, 'model_name_or_path', '')),
#         str(getattr(args, 'max_enc_seq_length', '')),
#         str(getattr(args, 'max_prompt_seq_length', '')),
#         str(getattr(args, 'model_type', '')),
#         str(getattr(args, 'dataset_type', '')),
#         str(getattr(args, 'role_path', '')),
#         str(getattr(args, 'prompt_path', '')),
#         set_type,
#     ])
#     return hashlib.md5(key_str.encode()).hexdigest()[:16]


# def _get_cache_path(args, set_type: str) -> str:
#     seed_dir    = os.path.dirname(args.output_dir)
#     dataset_dir = os.path.dirname(seed_dir)
#     cache_dir   = os.path.join(dataset_dir, "feature_cache")
#     os.makedirs(cache_dir, exist_ok=True)
#     key = _make_cache_key(args, set_type)
#     return os.path.join(cache_dir, f"features_{set_type}_{key}.pkl")


# # ─────────────────────────────────────────────────────────────────────────────
# # SyntaxProvider  ── spaCy 句法解析（全局单例）
# # ─────────────────────────────────────────────────────────────────────────────
# class SyntaxProvider:
#     DEPREL_WEIGHTS = {
#         "nsubj": 2.5,  "nsubjpass": 2.5,
#         "obj":   2.5,  "dobj":      2.5,  "iobj": 2.0,
#         "csubj": 2.0,  "ccomp":     1.8,  "xcomp": 1.8,
#         "root":  2.5,  "aux":       1.5,  "auxpass": 1.5,
#         "nmod":  1.2,  "amod":      1.0,  "advmod": 1.0,
#         "nummod":1.0,  "appos":     1.2,
#         "advcl": 0.8,  "acl":       0.8,  "relcl":  0.8,
#         "prep":  0.8,  "pobj":      1.0,
#     }
#     DEFAULT_WEIGHT = 0.5

#     def __init__(self):
#         print("[SyntaxProvider] 加载 en_core_web_trf ...")
#         self.nlp = spacy.load("en_core_web_trf")

#     def get_weighted_mask(self, text, trigger_span, tokenizer, max_len):
#         doc  = self.nlp(text)
#         mask = torch.zeros(max_len)
#         target_node = None
#         for token in doc:
#             if (token.idx >= trigger_span[0] and
#                     (token.idx + len(token.text)) <= trigger_span[1]):
#                 target_node = token
#                 break
#         if target_node:
#             encoding = tokenizer(
#                 text, return_offsets_mapping=True,
#                 max_length=max_len, truncation=True,
#             )
#             offsets = encoding['offset_mapping']
#             for child in target_node.children:
#                 weight = self.DEPREL_WEIGHTS.get(child.dep_, 0.0)
#                 if weight > 0:
#                     c_start, c_end = child.idx, child.idx + len(child.text)
#                     for i, (tok_start, tok_end) in enumerate(offsets):
#                         if tok_start >= c_start and tok_end <= c_end:
#                             mask[i] = weight
#         return mask

# def _merge_overlapping_clusters(clusters, logits):
#     """合并有公共节点的簇，取最大 logit。"""
#     merged = True
#     while merged:
#         merged = False
#         new_clusters = []
#         new_logits   = []
#         used = [False] * len(clusters)
#         for i in range(len(clusters)):
#             if used[i]:
#                 continue
#             current       = set(clusters[i])
#             current_logit = logits[i]
#             for j in range(i + 1, len(clusters)):
#                 if used[j]:
#                     continue
#                 if current & set(clusters[j]):
#                     current       |= set(clusters[j])
#                     current_logit  = max(current_logit, logits[j])
#                     used[j]        = True
#                     merged         = True
#             new_clusters.append(sorted(current))
#             new_logits.append(current_logit)
#             used[i] = True
#         clusters = new_clusters
#         logits   = new_logits
#     return clusters, logits

# class CorefProvider:
#     """
#     使用 fastcoref 进行共指消解，完全离线可用。
#     模型从本地路径加载。

#     安装：pip install fastcoref
#     模型：biu-nlp/f-coref（下载后放到本地路径）
#     """

#     FASTCOREF_MODEL_PATH = './fcoref_model'

#     def __init__(self, syntax_provider=None):
#         import logging
#         logging.getLogger("transformers").setLevel(logging.ERROR)
#         logging.getLogger("datasets").setLevel(logging.ERROR)
#         from datasets import disable_progress_bar
#         disable_progress_bar()
#         print(f"[CorefProvider] 加载 fastcoref 模型: {self.FASTCOREF_MODEL_PATH} ...")
#         try:
#             from fastcoref import FCoref
#             self.model = FCoref(
#                 model_name_or_path=self.FASTCOREF_MODEL_PATH,
#                 device='cpu',
#                 nlp='en_core_web_sm',
#                 enable_progress_bar=False,
#             )
#             self.enabled = True
#             print("[CorefProvider] fastcoref 加载成功。")
#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#             logger.warning(f"[CorefProvider] 加载失败: {e}，共指功能已禁用。")
#             self.enabled = False

#     def get_coref_clusters(
#         self,
#         text:                     str,
#         context:                  list,
#         old_tok_to_new_tok_index: list,
#         seq_len:                  int,
#         logit_threshold:          float = 3.0,
#     ) -> tuple:
#         """
#         返回 (clusters, cluster_logits)
#         clusters:      list[list[int]]  subword 粒度共指簇
#         cluster_logits: list[float]     每个簇的 max pairwise logit
#         """
#         FILTER_WORDS = {
#             'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for',
#             'with', 'by', 'from', 'and', 'or', 'but', 'as', 'that',
#             'this', 'it', 'its', 'be', 'is', 'was', 'are', 'were',
#             ',', '.', '!', '?', ';', ':', '"', "'", '(', ')',
#         }

#         if not self.enabled:
#             return [], []

#         if not hasattr(self, '_call_count'):
#             self._call_count = 0
#         self._call_count += 1
#         is_first = (self._call_count == 1)

#         import logging
#         logging.getLogger("transformers").setLevel(logging.ERROR)
#         logging.getLogger("datasets").setLevel(logging.ERROR)

#         try:
#             preds         = self.model.predict(texts=[text])
#             clusters_char = preds[0].get_clusters(as_strings=False)
#             pred          = preds[0]
#         except Exception as e:
#             logger.warning(f"[CorefProvider] 推理失败: {e}")
#             return [], []

#         if not clusters_char:
#             return [], []

#         # 字符映射
#         char_start_to_word_idx = {}
#         char_pos = 0
#         for w_i, tok in enumerate(context):
#             char_start_to_word_idx[char_pos] = w_i
#             char_pos += len(tok) + 1

#         def char_to_sw(char_start: int) -> int:
#             best_key, best_diff = None, float('inf')
#             for k in char_start_to_word_idx:
#                 diff = char_start - k
#                 if 0 <= diff < best_diff:
#                     best_diff, best_key = diff, k
#             if best_key is None:
#                 return -1
#             w_i = char_start_to_word_idx[best_key]
#             if w_i >= len(old_tok_to_new_tok_index):
#                 return -1
#             sw = old_tok_to_new_tok_index[w_i][0]
#             return sw if sw < seq_len else -1

#         if is_first:
#             print("\n[CorefProvider] ── 第1条样本共指簇详情 ──")

#         clusters      = []
#         cluster_logits = []

#         for cluster_char in clusters_char:
#             # 计算簇内最大 pairwise logit
#             max_logit = float('-inf')
#             for i in range(len(cluster_char)):
#                 for j in range(len(cluster_char)):
#                     if i == j:
#                         continue
#                     try:
#                         logit = pred.get_logit(cluster_char[i], cluster_char[j])
#                         if logit > max_logit:
#                             max_logit = logit
#                     except Exception:
#                         continue

#             if max_logit < logit_threshold:
#                 if is_first:
#                     print(f"  ❌ 过滤 (logit={max_logit:.2f}): "
#                         f"{[text[s:e] for s,e in cluster_char]}")
#                 continue

#             if is_first:
#                 print(f"  ✓ 保留 (max_logit={max_logit:.2f}): "
#                     f"{[text[s:e] for s,e in cluster_char]}")

#             # 转换为 subword 索引
#             sw_reps = []
#             for (char_s, char_e) in cluster_char:
#                 sw = char_to_sw(char_s)
#                 if sw >= 0 and sw not in sw_reps:
#                     sw_reps.append(sw)

#             sw_reps_sorted = sorted(sw_reps)
#             if len(sw_reps_sorted) < 2:
#                 continue

#             # 过滤功能词超级节点
#             rep_text = None
#             rep_idx  = sw_reps_sorted[0]
#             for (char_s, char_e) in cluster_char:
#                 if char_to_sw(char_s) == rep_idx:
#                     rep_text = text[char_s:char_e].strip().lower()
#                     break
#             if rep_text and rep_text in FILTER_WORDS:
#                 if is_first:
#                     print(f"  ⚠ 功能词超级节点过滤: '{rep_text}'")
#                 continue

#             clusters.append(sw_reps_sorted)
#             cluster_logits.append(max_logit)

#         if is_first:
#             print(f"[CorefProvider] 第1条样本最终簇数: {len(clusters)}\n")

#         # 传递闭包合并重叠簇
#         clusters, cluster_logits = _merge_overlapping_clusters(clusters, cluster_logits)
#         return clusters, cluster_logits
            
  
    
    
# # ─────────────────────────────────────────────────────────────────────────────
# # 数据结构
# # ─────────────────────────────────────────────────────────────────────────────
# class Events:
#     def __init__(self, doc_id, context, event_type_2_events,
#                  token_chunk_ids=None,
#                  entity_mentions=None,   # ← 新增
#                  event_mentions=None):
#         self.doc_id              = doc_id
#         self.context             = context
#         self.event_type_2_events = event_type_2_events
#         self.token_chunk_ids     = token_chunk_ids if token_chunk_ids is not None else []
#         self.entity_mentions     = entity_mentions    if entity_mentions    is not None else []
#         self.event_mentions      = event_mentions     if event_mentions     is not None else []


# class InputFeatures(object):
#     def __init__(self, example_id, feature_id,
#                  enc_text, dec_text,
#                  enc_tokens, dec_tokens,
#                  old_tok_to_new_tok_index,
#                  event_type, event_trigger, argument_type,
#                  enc_input_ids, enc_mask_ids,
#                  dec_input_ids, dec_mask_ids,
#                  answer_text, start_position=None, end_position=None):
#         self.example_id               = example_id
#         self.feature_id               = feature_id
#         self.enc_text                 = enc_text
#         self.dec_text                 = dec_text
#         self.enc_tokens               = enc_tokens
#         self.dec_tokens               = dec_tokens
#         self.old_tok_to_new_tok_index  = old_tok_to_new_tok_index
#         self.event_type               = event_type
#         self.event_trigger            = event_trigger
#         self.argument_type            = argument_type
#         self.enc_input_ids            = enc_input_ids
#         self.enc_mask_ids             = enc_mask_ids
#         self.dec_input_ids            = dec_input_ids
#         self.dec_mask_ids             = dec_mask_ids
#         self.answer_text              = answer_text
#         self.start_position           = start_position
#         self.end_position             = end_position

#     def __repr__(self):
#         s  = ""
#         s += "example_id: {}\n".format(self.example_id)
#         s += "event_type: {}\n".format(self.event_type)
#         s += "trigger_word: {}\n".format(self.event_trigger)
#         s += "argument_type: {}\n".format(self.argument_type)
#         s += "enc_input_ids: {}\n".format(self.enc_input_ids)
#         s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
#         s += "dec_input_ids: {}\n".format(self.dec_input_ids)
#         s += "dec_mask_ids: {}\n".format(self.dec_mask_ids)
#         s += "answer_text: {}\n".format(self.answer_text)
#         s += "start_position: {}\n".format(self.start_position)
#         s += "end_position: {}\n".format(self.end_position)
#         return s


# # ─────────────────────────────────────────────────────────────────────────────
# # 主处理器基类
# # ─────────────────────────────────────────────────────────────────────────────
# class DSET_processor:
#     def __init__(self, args, tokenizer):
#         self.args       = args
#         self.tokenizer  = tokenizer
#         self.template_dict, self.argument_dict = self._read_roles(self.args.role_path)
#         self.collate_fn = None

#         # ── 句法解析器（全局单例）────────────────────────────────────
#         self.syntax_provider = SyntaxProvider()

#         # ── 共指解析器（全局单例）────────────────────────────────────
#         # 首次实例化时加载 AllenNLP coref 模型（约 1.6GB，仅加载一次）
#         self.coref_provider = CorefProvider()

#     def _read_jsonlines(self, input_file):
#         lines = []
#         with jsonlines.open(input_file) as reader:
#             for obj in reader:
#                 lines.append(obj)
#         return lines

#     def _read_json(self, input_file):
#         with open(input_file, "r", encoding='utf-8') as f:
#             return json.load(f)

#     def _read_roles(self, role_path):
#         template_dict = {}
#         role_dict     = {}
#         if 'MLEE' in role_path:
#             with open(role_path) as f:
#                 role_name_mapping = json.load(f)
#                 for event_type, mapping in role_name_mapping.items():
#                     roles = list(mapping.keys())
#                     role_dict[event_type] = roles
#             return None, role_dict

#         with open(role_path, "r", encoding='utf-8') as f:
#             csv_reader = csv.reader(f)
#             for line in csv_reader:
#                 event_type_arg, template = line
#                 template_dict[event_type_arg] = template
#                 event_type, arg = event_type_arg.split('_')
#                 if event_type not in role_dict:
#                     role_dict[event_type] = []
#                 role_dict[event_type].append(arg)
#         return template_dict, role_dict

#     def _create_example(self, lines, over_sample=None, max_num_event=MAX_NUM_EVENTS):
#         W        = self.args.window_size
#         examples = []

#          # ── 构建原始数据查找表（读取所有split的原始数据）──────────────────
#         import os, json as _json
#         entity_map = {}
#         event_map  = {}
#         raw_base = './data/WikiEvent/data'
#         for split in ['train', 'dev', 'test']:
#             raw_path = os.path.join(raw_base, f'{split}.jsonl')
#             if not os.path.exists(raw_path):
#                 continue
#             with open(raw_path) as rf:
#                 for raw_line in rf:
#                     raw = _json.loads(raw_line)
#                     did = raw.get('doc_id', '')
#                     entity_map[did] = raw.get('entity_mentions', [])
#                     event_map[did]  = raw.get('event_mentions',  [])
#         # ─────────────────────────────────────────────────────────────────
        
#         for line in lines:
#             doc_id          = line["id"]
#             context         = line['context']
#             events          = line["events"]
#             token_chunk_ids = line.get('token_chunk_ids', [])
#             raw_doc_id      = doc_id.split('-', 1)[-1]
#             entity_mentions = entity_map.get(raw_doc_id, [])
#             event_mentions  = event_map.get(raw_doc_id,  [])
#             num_events      = len(events)
#             if num_events < 1:
#                 print('[num_events < 1]', doc_id)
#                 continue

#             events         = sorted(events, key=lambda x: x['trigger'])
#             context_length = len(context)
#             if context_length > W:
#                 for event in events:
#                     self.invalid_arg_num += len(event['args'])
#                 print('[context_length > W] %s\t\t%d' % (doc_id, context_length))
#                 continue

#             if num_events > max_num_event:
#                 for event in events[max_num_event:]:
#                     self.invalid_arg_num += len(event['args'])
#                 events = events[:max_num_event]
#                 print('[num_events > max_num_event] %s\t\t%d' % (doc_id, num_events))

#             assert len(events) <= MAX_NUM_EVENTS

#             if self.args.single:
#                 for event in events:
#                     event_type          = event['event_type']
#                     event_type_2_events = {event_type: [event]}
#                     examples.append(
#                         Events(doc_id, context, event_type_2_events, token_chunk_ids, entity_mentions, event_mentions)
#                     )
#                 continue

#             event_type_2_events = dict()
#             for event in events:
#                 event_type = event['event_type']
#                 if event_type not in event_type_2_events:
#                     event_type_2_events[event_type] = [event]
#                 else:
#                     event_type_2_events[event_type].append(event)

#             examples.append(
#                 Events(doc_id, context, event_type_2_events, token_chunk_ids, entity_mentions, event_mentions)
#             )

#             if over_sample == 'double' and num_events > 1:
#                 examples.append(
#                     Events(doc_id, context, event_type_2_events, token_chunk_ids, entity_mentions, event_mentions)
#                 )
#             elif over_sample == 'power' and num_events > 1:
#                 power_set = []

#                 def dfs(tmp, n):
#                     if len(tmp) > 1:
#                         power_set.append(tmp[:])
#                     for i in range(n, num_events):
#                         tmp.append(events[i])
#                         dfs(tmp, i + 1)
#                         tmp.pop()

#                 dfs([], 0)
#                 for i, events_ in enumerate(power_set):
#                     event_type_2_events_ = dict()
#                     for event in events_:
#                         etype = event['event_type']
#                         if etype not in event_type_2_events_:
#                             event_type_2_events_[etype] = [event]
#                         else:
#                             event_type_2_events_[etype].append(event)
#                     examples.append(
#                         Events('%d-%s' % (i, doc_id), context,
#                                event_type_2_events_, token_chunk_ids)
#                     )

#         logger.info("{} examples collected. {} arguments dropped.".format(
#             len(examples), self.invalid_arg_num))
#         print("{} examples collected. {} arguments dropped.".format(
#             len(examples), self.invalid_arg_num))
#         return examples

#     def create_example(self, file_path, set_type):
#         self.invalid_arg_num = 0
#         lines = self._read_jsonlines(file_path)
#         if self.args.dataset_type == 'MLEE':
#             return self._create_example(lines, over_sample=None)
#         elif self.args.dataset_type == 'rams':
#             return self._create_example(
#                 lines, over_sample=('power' if set_type == 'train' else None)
#             )
#         elif self.args.dataset_type == 'wikievent':
#             return self._create_example(
#                 lines, over_sample=('double' if set_type == 'train' else None)
#             )
#         else:
#             raise NotImplementedError()

#     def convert_examples_to_features(self, examples, role_name_mapping=None):
#         raise NotImplementedError("子类实现")

#     def convert_features_to_dataset(self, features):
#         raise NotImplementedError("子类实现")

#     def generate_dataloader(self, set_type):
#         assert set_type in ['train', 'dev', 'test']

#         if set_type == 'train':
#             file_path = self.args.train_file
#         elif set_type == 'dev':
#             file_path = self.args.dev_file
#         else:
#             file_path = self.args.test_file

#         cache_path = _get_cache_path(self.args, set_type)

#         if os.path.exists(cache_path):
#             print(f"[Cache] ✓ 命中缓存，跳过解析: {cache_path}")
#             logger.info(f"[Cache] Loading {set_type} features from cache: {cache_path}")
#             with open(cache_path, 'rb') as f:
#                 cached = pickle.load(f)
#             examples             = cached['examples']
#             features             = cached['features']
#             self.invalid_arg_num = cached.get('invalid_arg_num', 0)

#         else:
#             print(f"[Cache] ✗ 未命中缓存，开始解析 {set_type} 数据 ...")
#             logger.info(f"[Cache] Building {set_type} features ...")

#             examples = self.create_example(file_path, set_type)

#             if set_type == 'train' and self.args.keep_ratio < 1.0:
#                 sample_num = int(len(examples) * self.args.keep_ratio)
#                 examples   = sample(examples, sample_num)
#                 logger.info(
#                     "Few shot: keep ratio {}. {} training samples kept.".format(
#                         self.args.keep_ratio, len(examples))
#                 )

#             features = self.convert_examples_to_features(
#                 examples, getattr(self.args, 'role_name_mapping', None)
#             )

#             with open(cache_path, 'wb') as f:
#                 pickle.dump({
#                     'examples':        examples,
#                     'features':        features,
#                     'invalid_arg_num': self.invalid_arg_num,
#                 }, f, protocol=pickle.HIGHEST_PROTOCOL)
#             print(f"[Cache] 已保存缓存 → {cache_path}")
#             logger.info(f"[Cache] Saved {set_type} features to {cache_path}")

#         dataset         = self.convert_features_to_dataset(features)
#         dataset_sampler = (
#             RandomSampler(dataset) if set_type == 'train'
#             else SequentialSampler(dataset)
#         )

#         if self.collate_fn:
#             dataloader = DataLoader(
#                 dataset,
#                 sampler=dataset_sampler,
#                 batch_size=self.args.batch_size,
#                 collate_fn=self.collate_fn,
#             )
#         else:
#             dataloader = DataLoader(
#                 dataset,
#                 sampler=dataset_sampler,
#                 batch_size=self.args.batch_size,
#             )

#         return examples, features, dataloader, self.invalid_arg_num
