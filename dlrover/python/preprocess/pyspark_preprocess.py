#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ==========================================================
# 模块名 @ Version: 1.0
#        @ license: LGPL
#        @ Copyright (C) Metax.Corp 2024
#        @ Description
#             Using spark to generate megatron training data
#             Use example:
#              $ $SPARK_HOME/bin/spark-submit --executor-cores=2 --num-executors=2 \
#                   pyspark_preprocess.py --input $INPUT_PATH/$JSONL_FILES --output_prefix $OUTPUT_PATH \
#                   --vocab $TOKENIZER --tokenizer-type $TOKEN_TYPE --append-eod --jsonl-key text \
#                   --workers 64 --append-eod
# =========================================================

from pyspark.sql import SparkSession

import sys
import os
import lm_dataformat as lmd
import argparse
import numpy as np
import cloudpickle
import ftfy
import megatron
from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma separated "
        "list",
    )
    group.add_argument(
        "--jsonl-keys",
        nargs="+",
        default=["text"],
        help="space separate listed of keys to extract from jsonl. Defa",
    )
    group.add_argument(
        "--num-docs",
        default=None,
        help="Optional: Number of documents in the input data (if known) for an accurate progress bar.",
        type=int,
    )
    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        choices=[
            "HFGPT2Tokenizer",
            "HFTokenizer",
            "GPT2BPETokenizer",
            "CharLevelTokenizer",
            "TiktokenTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file"
    )
    group.add_argument(
        "--merge-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )
    group.add_argument(
        "--append-eod",
        action="store_true",
        help="Append an <eod> token to the end of a document.",
    )
    group.add_argument("--ftfy", action="store_true", help="Use ftfy to clean text")
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )
    group.add_argument(
        "--dataset-impl",
        type=str,
        default="mmap",
        choices=["lazy", "cached", "mmap"],
        help="Dataset implementation to use. Default: mmap",
    )

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Interval between progress updates",
    )
    args = parser.parse_args()
    args.keep_empty = False
    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.builders = {}
        self.output_bin_files = {}
        self.output_idx_files = {}
        self.tokenizer = None

    def initializer(self):
        # Use Encoder class as a container for global data
        self.tokenizer = build_tokenizer(self.args)
        for key in self.args.jsonl_keys:
            self.output_bin_files[key] = '{}_{}_{}.bin'.format(
                    self.args.output_prefix, key, 'document'
                    )
            self.output_idx_files[key] = '{}_{}_{}.idx'.format(
                    self.args.output_prefix, key, 'document'
                    )
        return self.tokenizer

    def get_builder(self, key):
        if key not in self.builders:
            self.builders[key] = indexed_dataset.make_builder(
                    self.output_bin_files[key],
                    impl = self.args.dataset_impl,
                    vocab_size = self.tokenizer.vocab_size,
                    )
        return self.builders[key]

    def encode(self, text):
        if self.args.ftfy:
            text = ftfy.fix_text(text)
        ids = {}
        for key in self.args.jsonl_keys:
            doc_ids = []
            text_ids = self.tokenizer.tokenize(text)
            if len(text_ids) > 0:
                doc_ids.append(text_ids)
            if self.args.append_eod:
                doc_ids[-1].append(self.tokenizer.eod)
            ids[key] = doc_ids
        return ids, len(text)

def main():
    spark = SparkSession.builder.appName('gpt-neox-process').getOrCreate()
    args = get_args()
    encoder = Encoder(args)
    tokenizer = encoder.initializer()
    broadcast_encoder = spark.sparkContext.broadcast(cloudpickle.dumps(encoder))

    # 读取数据并缓存
    lm_rdd = spark.sparkContext.parallelize(lmd.Reader(args.input).stream_data()) \
        .repartition(200)  # 调整为适当的分区数

    # 使用 mapPartitions 替代 flatMap，减少数据传输
    def process_partition(partition):
        encoder = cloudpickle.loads(broadcast_encoder.value)
        results = {key: [] for key in args.jsonl_keys}
        for doc in partition:
            encoded_doc, _ = encoder.encode(doc)
            for key, value_list in encoded_doc.items():
                results[key].extend(value_list)
        return results.items()

    def save_partition(partition):
        encoder = cloudpickle.loads(broadcast_encoder.value)
        encoder.initializer()

        def add_item(key, value):
            encoder.get_builder(key)
            if isinstance(value, list):
                for subitem in value:
                    if isinstance(subitem, list):
                        for e in subitem:
                            encoder.builders[key].add_item(np.array(e, dtype=encoder.builders[key].dtype))
            else:
                encoder.builders[key].add_item(np.array(value, dtype=encoder.builders[key].dtype))
            encoder.builders[key].end_document()

        for key, value in partition:
            add_item(key, value)

        for key in encoder.builders:
            encoder.builders[key].finalize(encoder.output_idx_files[key])

    lm_map_data = lm_rdd.mapPartitions(process_partition) \
        .groupByKey().mapValues(list) \
        .persist()  # 持久化结果

    # 触发计算并将结果保存到文件
    result = lm_map_data.foreachPartition(save_partition)

    # 清理缓存
    lm_rdd.unpersist()
    lm_map_data.unpersist()

    # 关闭 SparkSession
    spark.stop()

if __name__ == "__main__":
    main()

# Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

