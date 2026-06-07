# import torch.nn as nn
# import logging
# from torch.utils.data import DataLoader, RandomSampler
# logger = logging.getLogger(__name__)

# # collate_fn 返回的 batch tuple 结构（共 23 个元素）：
# # [0]  enc_input_ids
# # [1]  enc_mask_ids
# # [2]  all_ids
# # [3]  all_mask_ids
# # [4]  dec_arg_query_ids
# # [5]  dec_arg_query_mask_ids
# # [6]  dec_prompt_ids
# # [7]  dec_prompt_mask_ids
# # [8]  target_info
# # [9]  old_tok_to_new_tok_index
# # [10] arg_joint_prompt
# # [11] arg_lists
# # [12] example_idx
# # [13] feature_idx
# # [14] dec_arg_start_positions
# # [15] dec_arg_end_positions
# # [16] start_position_ids
# # [17] end_position_ids
# # [18] event_trigger          ← 原来的 batch[-2]，现在固定用 [18]
# # [19] enc_attention_mask     ← 原来的 batch[-1]，现在固定用 [19]
# # [20] dep_heads_batch        ← 新增
# # [21] dep_rels_batch         ← 新增
# # [22] event_groups_batch     ← 新增


# class BaseTrainer:
#     def __init__(
#         self,
#         cfg=None,
#         data_loader=None,
#         model=None,
#         optimizer=None,
#         scheduler=None,
#         processor=None
#     ):
#         self.cfg = cfg
#         self.data_loader = data_loader
#         self.data_iterator = iter(self.data_loader)
#         self.model = model
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.processor = processor
#         self._init_metric()

#     def _init_metric(self):
#         self.metric = {
#             "global_steps": 0,
#             "smooth_loss": 0.0,
#         }

#     def write_log(self):
#         logger.info("-----------------------global_step: {} -------------------------------- ".format(self.metric['global_steps']))
#         logger.info('lr: {}'.format(self.scheduler.get_last_lr()[0]))
#         logger.info('smooth_loss: {}'.format(self.metric['smooth_loss']))
#         self.metric['smooth_loss'] = 0.0

#     def train_one_step(self):
#         self.model.train()
#         try:
#             batch = next(self.data_iterator)
#         except StopIteration:
#             if self.processor is not None:
#                 print('re-generate training dataset')
#                 features = self.processor.convert_examples_to_features(self.examples, 'train', self.cfg.marker_range)
#                 dataset = self.processor.convert_features_to_dataset(features)
#                 dataset_sampler = RandomSampler(dataset)
#                 self.dataloader = DataLoader(dataset, sampler=dataset_sampler, batch_size=self.cfg.batch_size,
#                                              collate_fn=self.processor.collate_fn)

#             self.data_iterator = iter(self.data_loader)
#             batch = next(self.data_iterator)

#         inputs = self.convert_batch_to_inputs(batch)
#         loss, _ = self.model(**inputs)

#         if self.cfg.gradient_accumulation_steps > 1:
#             loss = loss / self.cfg.gradient_accumulation_steps
#         loss.backward()

#         if self.cfg.max_grad_norm != 0:
#             nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

#         self.metric['smooth_loss'] += loss.item() / self.cfg.logging_steps
#         if (self.metric['global_steps'] + 1) % self.cfg.gradient_accumulation_steps == 0:
#             self.optimizer.step()
#             self.scheduler.step()
#             self.model.zero_grad()
#             self.metric['global_steps'] += 1
#         else:
#             self.metric['global_steps'] += 1

#     def convert_batch_to_inputs(self, batch):
#         raise NotImplementedError()


# class Trainer(BaseTrainer):
#     def __init__(self, cfg=None, data_loader=None, model=None, optimizer=None, scheduler=None, processor=None):
#         super().__init__(cfg, data_loader, model, optimizer, scheduler)

#     def convert_batch_to_inputs(self, batch):
#         # # ==========================================================
#         # # 暴力调试模式：直接临时加载 Tokenizer，不依赖 self.processor
#         # # ==========================================================
#         # try:
#         #     from transformers import AutoTokenizer
#         #     # 填入你实际使用的模型路径，比如 'roberta-base' 或 'roberta-large'
#         #     temp_tokenizer = AutoTokenizer.from_pretrained('roberta-large')
            
#         #     sample_tensor = batch[2][0] # 取出 batch 中第一条数据的 all_ids
#         #     raw_text = temp_tokenizer.decode(sample_tensor, skip_special_tokens=True)
            
#         #     print("\n" + "🔥"*10 + " 成功拦截原始文本 " + "🔥"*10)
#         #     print(f"【文本内容】: {raw_text}...\n")
            
#         # except Exception as e:
#         #     print(f"【暴力解码失败】: {e}")
#         # # ==========================================================


#         inputs = {
#             'enc_input_ids':          batch[0].to(self.cfg.device),
#             'enc_mask_ids':           batch[1].to(self.cfg.device),
#             'all_ids':                batch[2].to(self.cfg.device),
#             'all_mask_ids':           batch[3].to(self.cfg.device),
#             'dec_prompt_ids':         batch[6].to(self.cfg.device),
#             'dec_prompt_mask_ids':    batch[7].to(self.cfg.device),
#             'target_info':            batch[8],
#             'old_tok_to_new_tok_indexs': batch[9],
#             'arg_joint_prompts':      batch[10],
#             'arg_list':               batch[11],
#             'event_triggers':         batch[18],   # 固定索引，不再用 batch[-2]
#             'enc_attention_mask':     batch[19],   # 固定索引，不再用 batch[-1]
#             # ── 新增三个图字段 ──────────────────────────────────────────
#             'dep_heads_batch':        batch[20],
#             'dep_rels_batch':         batch[21],
#             'event_groups_batch':     batch[22],
#         }
#         return inputs

# ##版本2：采用触发词相似度来构建事件之间的关联
# import torch.nn as nn
# import logging
# from torch.utils.data import DataLoader, RandomSampler
# logger = logging.getLogger(__name__)

# # collate_fn 返回的 batch tuple 结构（共 23 个元素）：
# # [0]  enc_input_ids
# # [1]  enc_mask_ids
# # [2]  all_ids
# # [3]  all_mask_ids
# # [4]  dec_arg_query_ids
# # [5]  dec_arg_query_mask_ids
# # [6]  dec_prompt_ids
# # [7]  dec_prompt_mask_ids
# # [8]  target_info
# # [9]  old_tok_to_new_tok_index
# # [10] arg_joint_prompt
# # [11] arg_lists
# # [12] example_idx
# # [13] feature_idx
# # [14] dec_arg_start_positions
# # [15] dec_arg_end_positions
# # [16] start_position_ids
# # [17] end_position_ids
# # [18] event_trigger          ← 原来的 batch[-2]，现在固定用 [18]
# # [19] enc_attention_mask     ← 原来的 batch[-1]，现在固定用 [19]
# # [20] dep_heads_batch        ← 新增
# # [21] dep_rels_batch         ← 新增
# # [22] event_groups_batch     ← 新增
#             # [23] event_type_batch       ← 新增


# class BaseTrainer:
#     def __init__(
#         self,
#         cfg=None,
#         data_loader=None,
#         model=None,
#         optimizer=None,
#         scheduler=None,
#         processor=None
#     ):
#         self.cfg = cfg
#         self.data_loader = data_loader
#         self.data_iterator = iter(self.data_loader)
#         self.model = model
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.processor = processor
#         self._init_metric()

#     def _init_metric(self):
#         self.metric = {
#             "global_steps": 0,
#             "smooth_loss": 0.0,
#         }

#     def write_log(self):
#         logger.info("-----------------------global_step: {} -------------------------------- ".format(self.metric['global_steps']))
#         logger.info('lr: {}'.format(self.scheduler.get_last_lr()[0]))
#         logger.info('smooth_loss: {}'.format(self.metric['smooth_loss']))
#         self.metric['smooth_loss'] = 0.0

#     def train_one_step(self):
#         self.model.train()
#         try:
#             batch = next(self.data_iterator)
#         except StopIteration:
#             if self.processor is not None:
#                 print('re-generate training dataset')
#                 features = self.processor.convert_examples_to_features(self.examples, 'train', self.cfg.marker_range)
#                 dataset = self.processor.convert_features_to_dataset(features)
#                 dataset_sampler = RandomSampler(dataset)
#                 self.dataloader = DataLoader(dataset, sampler=dataset_sampler, batch_size=self.cfg.batch_size,
#                                              collate_fn=self.processor.collate_fn)

#             self.data_iterator = iter(self.data_loader)
#             batch = next(self.data_iterator)

#         inputs = self.convert_batch_to_inputs(batch)
#         loss, _ = self.model(**inputs)

#         if self.cfg.gradient_accumulation_steps > 1:
#             loss = loss / self.cfg.gradient_accumulation_steps
#         loss.backward()

#         if self.cfg.max_grad_norm != 0:
#             nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

#         self.metric['smooth_loss'] += loss.item() / self.cfg.logging_steps
#         if (self.metric['global_steps'] + 1) % self.cfg.gradient_accumulation_steps == 0:
#             self.optimizer.step()
#             self.scheduler.step()
#             self.model.zero_grad()
#             self.metric['global_steps'] += 1
#         else:
#             self.metric['global_steps'] += 1

#     def convert_batch_to_inputs(self, batch):
#         raise NotImplementedError()


# class Trainer(BaseTrainer):
#     def __init__(self, cfg=None, data_loader=None, model=None, optimizer=None, scheduler=None, processor=None):
#         super().__init__(cfg, data_loader, model, optimizer, scheduler)

#     def convert_batch_to_inputs(self, batch):
#         inputs = {
#             'enc_input_ids':          batch[0].to(self.cfg.device),
#             'enc_mask_ids':           batch[1].to(self.cfg.device),
#             'all_ids':                batch[2].to(self.cfg.device),
#             'all_mask_ids':           batch[3].to(self.cfg.device),
#             'dec_prompt_ids':         batch[6].to(self.cfg.device),
#             'dec_prompt_mask_ids':    batch[7].to(self.cfg.device),
#             'target_info':            batch[8],
#             'old_tok_to_new_tok_indexs': batch[9],
#             'arg_joint_prompts':      batch[10],
#             'arg_list':               batch[11],
#             'event_triggers':         batch[18],   # 固定索引，不再用 batch[-2]
#             'enc_attention_mask':     batch[19],   # 固定索引，不再用 batch[-1]
#             # ── 新增三个图字段 ──────────────────────────────────────────
#             'dep_heads_batch':        batch[20],
#             'dep_rels_batch':         batch[21],
#             'event_groups_batch':     batch[22],
#         }
#         return inputs

# #版本3：共现图+句法依存图
# import torch.nn as nn
# import logging
# from torch.utils.data import DataLoader, RandomSampler

# logger = logging.getLogger(__name__)

# # collate_fn 返回的 batch tuple 结构（共 24 个元素）：
# # [0]  enc_input_ids
# # [1]  enc_mask_ids
# # [2]  all_ids
# # [3]  all_mask_ids
# # [4]  dec_arg_query_ids
# # [5]  dec_arg_query_mask_ids
# # [6]  dec_prompt_ids
# # [7]  dec_prompt_mask_ids
# # [8]  target_info
# # [9]  old_tok_to_new_tok_index
# # [10] arg_joint_prompt
# # [11] arg_lists
# # [12] example_idx
# # [13] feature_idx
# # [14] dec_arg_start_positions
# # [15] dec_arg_end_positions
# # [16] start_position_ids
# # [17] end_position_ids
# # [18] event_trigger
# # [19] enc_attention_mask
# # [20] dep_heads_batch        ← 图1：依存 head 索引
# # [21] dep_rels_batch         ← 图1：依存关系标签
# # [22] event_groups_batch     ← 图1：意群 group id
# # [23] event_types_batch      ← 图2：每个事件的类型字符串（新增）


# class BaseTrainer:
#     def __init__(
#         self,
#         cfg=None,
#         data_loader=None,
#         model=None,
#         optimizer=None,
#         scheduler=None,
#         processor=None,
#     ):
#         self.cfg           = cfg
#         self.data_loader   = data_loader
#         self.data_iterator = iter(self.data_loader)
#         self.model         = model
#         self.optimizer     = optimizer
#         self.scheduler     = scheduler
#         self.processor     = processor
#         self._init_metric()

#     def _init_metric(self):
#         self.metric = {
#             "global_steps": 0,
#             "smooth_loss":  0.0,
#         }

#     def write_log(self):
#         logger.info(
#             "-----------------------global_step: {} -------------------------------- "
#             .format(self.metric['global_steps'])
#         )
#         logger.info('lr: {}'.format(self.scheduler.get_last_lr()[0]))
#         logger.info('smooth_loss: {}'.format(self.metric['smooth_loss']))
#         self.metric['smooth_loss'] = 0.0

#     def train_one_step(self):
#         self.model.train()
#         try:
#             batch = next(self.data_iterator)
#         except StopIteration:
#             if self.processor is not None:
#                 print('re-generate training dataset')
#                 features = self.processor.convert_examples_to_features(
#                     self.examples, 'train', self.cfg.marker_range
#                 )
#                 dataset        = self.processor.convert_features_to_dataset(features)
#                 dataset_sampler = RandomSampler(dataset)
#                 self.dataloader = DataLoader(
#                     dataset,
#                     sampler=dataset_sampler,
#                     batch_size=self.cfg.batch_size,
#                     collate_fn=self.processor.collate_fn,
#                 )

#             self.data_iterator = iter(self.data_loader)
#             batch = next(self.data_iterator)

#         inputs      = self.convert_batch_to_inputs(batch)
#         loss, _     = self.model(**inputs)

#         if self.cfg.gradient_accumulation_steps > 1:
#             loss = loss / self.cfg.gradient_accumulation_steps
#         loss.backward()

#         if self.cfg.max_grad_norm != 0:
#             nn.utils.clip_grad_norm_(
#                 self.model.parameters(), self.cfg.max_grad_norm
#             )

#         self.metric['smooth_loss'] += (
#             loss.item() / self.cfg.logging_steps
#         )
#         if (self.metric['global_steps'] + 1) % self.cfg.gradient_accumulation_steps == 0:
#             self.optimizer.step()
#             self.scheduler.step()
#             self.model.zero_grad()
#             self.metric['global_steps'] += 1
#         else:
#             self.metric['global_steps'] += 1

#     def convert_batch_to_inputs(self, batch):
#         raise NotImplementedError()


# class Trainer(BaseTrainer):
#     def __init__(
#         self,
#         cfg=None,
#         data_loader=None,
#         model=None,
#         optimizer=None,
#         scheduler=None,
#         processor=None,
#     ):
#         super().__init__(cfg, data_loader, model, optimizer, scheduler)

#     def convert_batch_to_inputs(self, batch):
#         inputs = {
#             # ── 原有字段（索引不变）────────────────────────────────────
#             'enc_input_ids':             batch[0].to(self.cfg.device),
#             'enc_mask_ids':              batch[1].to(self.cfg.device),
#             'all_ids':                   batch[2].to(self.cfg.device),
#             'all_mask_ids':              batch[3].to(self.cfg.device),
#             'dec_prompt_ids':            batch[6].to(self.cfg.device),
#             'dec_prompt_mask_ids':       batch[7].to(self.cfg.device),
#             'target_info':               batch[8],
#             'old_tok_to_new_tok_indexs': batch[9],
#             'arg_joint_prompts':         batch[10],
#             'arg_list':                  batch[11],
#             'event_triggers':            batch[18],
#             'enc_attention_mask':        batch[19],
#             # ── 图1 字段 ───────────────────────────────────────────────
#             'dep_heads_batch':           batch[20],   # list[list[int]]
#             'dep_rels_batch':            batch[21],   # list[list[str]]
#             'event_groups_batch':        batch[22],   # list[list[int]]
#             # ── 图2 字段（新增）───────────────────────────────────────
#             'event_types_batch':         batch[23],   # list[list[str]]
#             # arg_role_positions_batch 不传入，
#             # DyGMA.forward() 会从 arg_joint_prompts 自动提取角色 token 位置
#         }
#         return inputs

#版本4：句法依赖+共指
import torch.nn as nn
import logging
from torch.utils.data import DataLoader, RandomSampler

logger = logging.getLogger(__name__)

# collate_fn 返回的 batch tuple 结构（共 25 个元素）：
# [0]  enc_input_ids
# [1]  enc_mask_ids
# [2]  all_ids
# [3]  all_mask_ids
# [4]  dec_arg_query_ids
# [5]  dec_arg_query_mask_ids
# [6]  dec_prompt_ids
# [7]  dec_prompt_mask_ids
# [8]  target_info
# [9]  old_tok_to_new_tok_index
# [10] arg_joint_prompt
# [11] arg_lists
# [12] example_idx
# [13] feature_idx
# [14] dec_arg_start_positions
# [15] dec_arg_end_positions
# [16] start_position_ids
# [17] end_position_ids
# [18] event_trigger
# [19] enc_attention_mask
# [20] dep_heads_batch         ← 图1：依存 head 索引
# [21] dep_rels_batch          ← 图1：依存关系标签
# [22] event_groups_batch      ← 图1：意群 group id
# [23] coref_clusters_batch    ← 图1：共指簇（新增）
#                                list[list[list[int]]]
#                                每个样本是一组共指簇，每个簇是一组 subword 索引
#                                第一个元素为超级节点（代表词），其余被重定向到它
# [24] coref_logits_batch      ← 共指置信度


class BaseTrainer:
    def __init__(
        self,
        cfg=None,
        data_loader=None,
        model=None,
        optimizer=None,
        scheduler=None,
        processor=None,
    ):
        self.cfg           = cfg
        self.data_loader   = data_loader
        self.data_iterator = iter(self.data_loader)
        self.model         = model
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.processor     = processor
        self._init_metric()

    def _init_metric(self):
        self.metric = {
            "global_steps": 0,
            "smooth_loss":  0.0,
        }

    def write_log(self):
        logger.info(
            "-----------------------global_step: {} -------------------------------- "
            .format(self.metric['global_steps'])
        )
        logger.info('lr: {}'.format(self.scheduler.get_last_lr()[0]))
        logger.info('smooth_loss: {}'.format(self.metric['smooth_loss']))
        self.metric['smooth_loss'] = 0.0

    def train_one_step(self):
        self.model.train()
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            if self.processor is not None:
                print('re-generate training dataset')
                features = self.processor.convert_examples_to_features(
                    self.examples, 'train', self.cfg.marker_range
                )
                dataset         = self.processor.convert_features_to_dataset(features)
                dataset_sampler = RandomSampler(dataset)
                self.dataloader = DataLoader(
                    dataset,
                    sampler=dataset_sampler,
                    batch_size=self.cfg.batch_size,
                    collate_fn=self.processor.collate_fn,
                )

            self.data_iterator = iter(self.data_loader)
            batch = next(self.data_iterator)

        inputs  = self.convert_batch_to_inputs(batch)
        loss, _ = self.model(**inputs)

        if self.cfg.gradient_accumulation_steps > 1:
            loss = loss / self.cfg.gradient_accumulation_steps
        loss.backward()

        if self.cfg.max_grad_norm != 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm
            )

        self.metric['smooth_loss'] += loss.item() / self.cfg.logging_steps
        if (self.metric['global_steps'] + 1) % self.cfg.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()
            self.metric['global_steps'] += 1
        else:
            self.metric['global_steps'] += 1

    def convert_batch_to_inputs(self, batch):
        raise NotImplementedError()


class Trainer(BaseTrainer):
    def __init__(
        self,
        cfg=None,
        data_loader=None,
        model=None,
        optimizer=None,
        scheduler=None,
        processor=None,
    ):
        super().__init__(cfg, data_loader, model, optimizer, scheduler)

    def convert_batch_to_inputs(self, batch):
        inputs = {
            # ── 原有字段（索引不变）────────────────────────────────────
            'enc_input_ids':             batch[0].to(self.cfg.device),
            'enc_mask_ids':              batch[1].to(self.cfg.device),
            'all_ids':                   batch[2].to(self.cfg.device),
            'all_mask_ids':              batch[3].to(self.cfg.device),
            'dec_prompt_ids':            batch[6].to(self.cfg.device),
            'dec_prompt_mask_ids':       batch[7].to(self.cfg.device),
            'target_info':               batch[8],
            'old_tok_to_new_tok_indexs': batch[9],
            'arg_joint_prompts':         batch[10],
            'arg_list':                  batch[11],
            'event_triggers':            batch[18],
            'enc_attention_mask':        batch[19],
            # ── 图1 字段 ───────────────────────────────────────────────
            'dep_heads_batch':           batch[20],   # list[list[int]]
            'dep_rels_batch':            batch[21],   # list[list[str]]
            'event_groups_batch':        batch[22],   # list[list[int]]
            # ── 共指簇字段（新增）─────────────────────────────────────
            # 格式：list[list[list[int]]]
            # 外层 list：batch 维度
            # 中层 list：该样本的共指簇列表
            # 内层 list：同一簇内各 mention 的代表 subword 索引
            #            第一个元素为超级节点，其余被重定向到它
            'coref_clusters_batch':      batch[23],
            'coref_logits_batch':        batch[24],
        }
        # print(f"[DEBUG] batch[23] 第一个样本簇数: {len(batch[23][0])}")
        # print(f"[DEBUG] batch[23] 第一个样本内容: {batch[23][0]}")
        # print(f"[DEBUG] batch[12] example_idx: {batch[12]}")
        return inputs

# #版本5：共指+句法依赖+采用事件模式的同类型标签缓解零指代论元
# import torch.nn as nn
# import logging
# from torch.utils.data import DataLoader, RandomSampler

# logger = logging.getLogger(__name__)

# # collate_fn 返回的 batch tuple 结构（共 24 个元素）：
# # [0]  enc_input_ids
# # [1]  enc_mask_ids
# # [2]  all_ids
# # [3]  all_mask_ids
# # [4]  dec_arg_query_ids
# # [5]  dec_arg_query_mask_ids
# # [6]  dec_prompt_ids
# # [7]  dec_prompt_mask_ids
# # [8]  target_info
# # [9]  old_tok_to_new_tok_index
# # [10] arg_joint_prompt
# # [11] arg_lists
# # [12] example_idx
# # [13] feature_idx
# # [14] dec_arg_start_positions
# # [15] dec_arg_end_positions
# # [16] start_position_ids
# # [17] end_position_ids
# # [18] event_trigger
# # [19] enc_attention_mask
# # [20] dep_heads_batch         ← 图1：依存 head 索引
# # [21] dep_rels_batch          ← 图1：依存关系标签
# # [22] event_groups_batch      ← 图1：意群 group id
# # [23] coref_clusters_batch    ← 图1：共指簇（新增）
# #                                list[list[list[int]]]
# #                                每个样本是一组共指簇，每个簇是一组 subword 索引
# #                                第一个元素为超级节点（代表词），其余被重定向到它


# class BaseTrainer:
#     def __init__(
#         self,
#         cfg=None,
#         data_loader=None,
#         model=None,
#         optimizer=None,
#         scheduler=None,
#         processor=None,
#     ):
#         self.cfg           = cfg
#         self.data_loader   = data_loader
#         self.data_iterator = iter(self.data_loader)
#         self.model         = model
#         self.optimizer     = optimizer
#         self.scheduler     = scheduler
#         self.processor     = processor
#         self._init_metric()

#     def _init_metric(self):
#         self.metric = {
#             "global_steps": 0,
#             "smooth_loss":  0.0,
#         }

#     def write_log(self):
#         logger.info(
#             "-----------------------global_step: {} -------------------------------- "
#             .format(self.metric['global_steps'])
#         )
#         logger.info('lr: {}'.format(self.scheduler.get_last_lr()[0]))
#         logger.info('smooth_loss: {}'.format(self.metric['smooth_loss']))
#         self.metric['smooth_loss'] = 0.0

#     def train_one_step(self):
#         self.model.train()
#         try:
#             batch = next(self.data_iterator)
#         except StopIteration:
#             if self.processor is not None:
#                 print('re-generate training dataset')
#                 features = self.processor.convert_examples_to_features(
#                     self.examples, 'train', self.cfg.marker_range
#                 )
#                 dataset         = self.processor.convert_features_to_dataset(features)
#                 dataset_sampler = RandomSampler(dataset)
#                 self.dataloader = DataLoader(
#                     dataset,
#                     sampler=dataset_sampler,
#                     batch_size=self.cfg.batch_size,
#                     collate_fn=self.processor.collate_fn,
#                 )

#             self.data_iterator = iter(self.data_loader)
#             batch = next(self.data_iterator)

#         inputs  = self.convert_batch_to_inputs(batch)
#         loss, _ = self.model(**inputs)

#         if self.cfg.gradient_accumulation_steps > 1:
#             loss = loss / self.cfg.gradient_accumulation_steps
#         loss.backward()

#         if self.cfg.max_grad_norm != 0:
#             nn.utils.clip_grad_norm_(
#                 self.model.parameters(), self.cfg.max_grad_norm
#             )

#         self.metric['smooth_loss'] += loss.item() / self.cfg.logging_steps
#         if (self.metric['global_steps'] + 1) % self.cfg.gradient_accumulation_steps == 0:
#             self.optimizer.step()
#             self.scheduler.step()
#             self.model.zero_grad()
#             self.metric['global_steps'] += 1
#         else:
#             self.metric['global_steps'] += 1

#     def convert_batch_to_inputs(self, batch):
#         raise NotImplementedError()


# class Trainer(BaseTrainer):
#     def __init__(
#         self,
#         cfg=None,
#         data_loader=None,
#         model=None,
#         optimizer=None,
#         scheduler=None,
#         processor=None,
#     ):
#         super().__init__(cfg, data_loader, model, optimizer, scheduler)

#     def convert_batch_to_inputs(self, batch):
#         inputs = {
#             # ── 原有字段（索引不变）────────────────────────────────────
#             'enc_input_ids':             batch[0].to(self.cfg.device),
#             'enc_mask_ids':              batch[1].to(self.cfg.device),
#             'all_ids':                   batch[2].to(self.cfg.device),
#             'all_mask_ids':              batch[3].to(self.cfg.device),
#             'dec_prompt_ids':            batch[6].to(self.cfg.device),
#             'dec_prompt_mask_ids':       batch[7].to(self.cfg.device),
#             'target_info':               batch[8],
#             'old_tok_to_new_tok_indexs': batch[9],
#             'arg_joint_prompts':         batch[10],
#             'arg_list':                  batch[11],
#             'event_triggers':            batch[18],
#             'enc_attention_mask':        batch[19],
#             # ── 图1 字段 ───────────────────────────────────────────────
#             'dep_heads_batch':           batch[20],   # list[list[int]]
#             'dep_rels_batch':            batch[21],   # list[list[str]]
#             'event_groups_batch':        batch[22],   # list[list[int]]
#             # ── 共指簇字段（新增）─────────────────────────────────────
#             # 格式：list[list[list[int]]]
#             # 外层 list：batch 维度
#             # 中层 list：该样本的共指簇列表
#             # 内层 list：同一簇内各 mention 的代表 subword 索引
#             #            第一个元素为超级节点，其余被重定向到它
#             'coref_clusters_batch':      batch[23],
#             'coref_logits_batch':        batch[24],
#             # # [新增] 提取 Schema 信息
#             # 'entity_type_map_batch':   batch[25] if len(batch) > 25 else None,
#             # 'event_type_map_batch':    batch[26] if len(batch) > 26 else None,
#             # # 训练步信息（用于模型内稀疏日志）
#             # 'global_step':             self.metric['global_steps'],
#             # 'max_steps':               self.cfg.max_steps,
#         }
#         # print(f"[DEBUG] batch[23] 第一个样本簇数: {len(batch[23][0])}")
#         # print(f"[DEBUG] batch[23] 第一个样本内容: {batch[23][0]}")
#         # print(f"[DEBUG] batch[12] example_idx: {batch[12]}")
#         return inputs
