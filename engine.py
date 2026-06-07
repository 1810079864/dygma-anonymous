import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"
if os.environ.get('DEBUG', False): print('\033[92m' + 'Running code in DEBUG mode' + '\033[0m')
import logging
from models import build_model
from processors import build_processor
from utils import set_seed
from runner.runner import Runner

logger = logging.getLogger(__name__)


def run(args, model, processor, optimizer, scheduler):
    set_seed(args)

    logger.info("train dataloader generation")
    train_examples, train_features, train_dataloader, args.train_invalid_num = processor.generate_dataloader('train')
    logger.info("dev dataloader generation")
    dev_examples, dev_features, dev_dataloader, args.dev_invalid_num = processor.generate_dataloader('dev')
    logger.info("test dataloader generation")
    test_examples, test_features, test_dataloader, args.test_invalid_num = processor.generate_dataloader('test')

    runner = Runner(
        cfg=args,
        data_samples=[train_examples, dev_examples, test_examples],
        data_features=[train_features, dev_features, test_features],
        data_loaders=[train_dataloader, dev_dataloader, test_dataloader],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_fn_dict=None,
    )
    runner.run()



def main():
    from config_parser import get_args_parser
    args = get_args_parser()

    if not args.inference_only:
        print(f"Output full path {os.path.join(os.getcwd(), args.output_dir)}")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logging.basicConfig(
            filename=os.path.join(args.output_dir, "log.txt"), \
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO
        )
    else:
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO
        )
    set_seed(args)

    model, tokenizer, optimizer, scheduler = build_model(args, args.model_type)
    model.to(args.device)

    processor = build_processor(args, tokenizer)

    logger.info("Training/evaluation parameters %s", args)

    # DropoutRate, batch_size , learning_rate , epochs , LRpatience, seed

    run(args, model, processor, optimizer, scheduler)



if __name__ == "__main__":

    main()

# #版本3：共现图+句法依存图
# import os
# os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"
# if os.environ.get('DEBUG', False): print('\033[92m' + 'Running code in DEBUG mode' + '\033[0m')
# import pickle
# import logging
# import jsonlines

# from models import build_model
# from processors import build_processor
# from utils import set_seed
# from runner.runner import Runner

# # 从 DyGMA 模型导入共现统计工具函数
# from models.DyGMA import compute_cooccurrence_stats, normalize_cooccurrence_stats

# logger = logging.getLogger(__name__)


# # ===========================================================================
# # 共现统计构建（训练前调用一次，结果磁盘缓存）
# # ===========================================================================

# def build_or_load_co_stats(train_file: str) -> dict:
#     """
#     从训练集预统计事件共现，结果缓存到磁盘，多个 seed 共享同一份缓存。

#     缓存路径：与训练文件同目录，固定文件名 co_stats_cache.pkl
#     例：./data/WikiEvent/data_final/co_stats_cache.pkl

#     Returns:
#         norm_co_stats : 归一化共现统计字典
#             {
#               "type_pair": { ("TypeA","TypeB"): float, ... },
#               "type_role": { ("TypeA","Role"):  float, ... },
#             }
#     """
#     cache_path = os.path.join(os.path.dirname(train_file), "co_stats_cache.pkl")

#     if os.path.exists(cache_path):
#         logger.info(f"[CoStats] 命中缓存，直接加载: {cache_path}")
#         print(f"[CoStats] ✓ 命中缓存: {cache_path}")
#         with open(cache_path, "rb") as f:
#             return pickle.load(f)

#     logger.info(f"[CoStats] 未命中缓存，开始统计训练集共现: {train_file}")
#     print(f"[CoStats] ✗ 未命中缓存，开始统计: {train_file}")

#     # 读取训练集，转换为统计函数所需格式
#     # wikievent/KAIROS args 格式：[start, end, text, role]
#     # 取最后一个元素（arg[-1]）作为角色名字符串
#     train_samples = []
#     with jsonlines.open(train_file) as reader:
#         for line in reader:
#             sample = {
#                 "events": [
#                     {
#                         "event_type": ev["event_type"],
#                         "args": [
#                             {"role": arg[-1]}
#                             for arg in ev.get("args", [])
#                         ],
#                     }
#                     for ev in line.get("events", [])
#                 ]
#             }
#             train_samples.append(sample)

#     # 统计并归一化
#     co_stats      = compute_cooccurrence_stats(train_samples)
#     norm_co_stats = normalize_cooccurrence_stats(co_stats)

#     # 缓存到磁盘
#     with open(cache_path, "wb") as f:
#         pickle.dump(norm_co_stats, f)

#     logger.info(
#         f"[CoStats] 已缓存 -> {cache_path} "
#         f"| type_pair: {len(co_stats['type_pair'])} 条 "
#         f"| type_role: {len(co_stats['type_role'])} 条"
#     )
#     print(
#         f"[CoStats] 已缓存 -> {cache_path} "
#         f"| type_pair: {len(co_stats['type_pair'])} 条 "
#         f"| type_role: {len(co_stats['type_role'])} 条"
#     )
#     return norm_co_stats


# # ===========================================================================
# # 主运行函数（与原版完全一致）
# # ===========================================================================

# def run(args, model, processor, optimizer, scheduler):
#     set_seed(args)

#     logger.info("train dataloader generation")
#     train_examples, train_features, train_dataloader, args.train_invalid_num = \
#         processor.generate_dataloader('train')

#     logger.info("dev dataloader generation")
#     dev_examples, dev_features, dev_dataloader, args.dev_invalid_num = \
#         processor.generate_dataloader('dev')

#     logger.info("test dataloader generation")
#     test_examples, test_features, test_dataloader, args.test_invalid_num = \
#         processor.generate_dataloader('test')

#     runner = Runner(
#         cfg=args,
#         data_samples=[train_examples, dev_examples, test_examples],
#         data_features=[train_features, dev_features, test_features],
#         data_loaders=[train_dataloader, dev_dataloader, test_dataloader],
#         model=model,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         metric_fn_dict=None,
#     )
#     runner.run()


# # ===========================================================================
# # 入口
# # ===========================================================================

# def main():
#     from config_parser import get_args_parser
#     args = get_args_parser()

#     if not args.inference_only:
#         print(f"Output full path {os.path.join(os.getcwd(), args.output_dir)}")
#         if not os.path.exists(args.output_dir):
#             os.makedirs(args.output_dir)

#         logging.basicConfig(
#             filename=os.path.join(args.output_dir, "log.txt"),
#             format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#             datefmt='%m/%d/%Y %H:%M:%S',
#             level=logging.INFO,
#         )
#     else:
#         logging.basicConfig(
#             format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#             datefmt='%m/%d/%Y %H:%M:%S',
#             level=logging.INFO,
#         )

#     set_seed(args)

#     # ── 1. 构建模型 ────────────────────────────────────────────────────
#     model, tokenizer, optimizer, scheduler = build_model(args, args.model_type)
#     model.to(args.device)

#     # ── 2. 构建数据处理器 ──────────────────────────────────────────────
#     # 注意：build_processor 内部会给 args 动态添加 train_file / dev_file /
#     # test_file 属性，因此共现统计的注入必须在此之后进行。
#     processor = build_processor(args, tokenizer)

#     # ── 3. 注入事件共现统计 ────────────────────────────────────────────
#     # build_processor 执行后 args.train_file 已经被正确赋值。
#     # 通过 hasattr 判断，仅对 DyGMA 模型注入，不影响其他模型类型。
#     if hasattr(model, 'set_cooccurrence_stats'):
#         norm_co_stats = build_or_load_co_stats(args.train_file)
#         model.set_cooccurrence_stats(norm_co_stats)
#         logger.info("[CoStats] 共现统计已注入模型。")
#         print("[CoStats] 共现统计已注入模型。")

#     logger.info("Training/evaluation parameters %s", args)

#     # ── 4. 启动训练/推理 ───────────────────────────────────────────────
#     run(args, model, processor, optimizer, scheduler)


# if __name__ == "__main__":
#     main()