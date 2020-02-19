# coding=utf-8
# Copyright (c) 2019, The Microsoft DeepSpeed Team. All rights reserved.
#
# Note: please copy webtext data to "Megatron-LM" folder, before running this script.

import unittest
import subprocess
import os
import time
import re
from test_common import BaseTestCase


class GPT2PerfTestCase(BaseTestCase):
    def __init__(self, methodName="DeepSpeed performance test on GPT2 model"):
        super(GPT2PerfTestCase, self).__init__(methodName)

    def test_perf_1_5B(self):
        test_config = {
            "mp": 1,
            "gpus": 16,
            "nodes": 4,
            "bs": 32,
            "steps": 100,
            "layers": 48,
            "hidden_size": 1600,
            "seq_length": 1024,
            "heads": 16,
            "deepspeed": True,
            "json": "ds_config_perf_bs32.json",
        }

        self.run_test(test_config)

    def test_perf_4B(self):
        test_config = {
            "mp": 1,
            "gpus": 16,
            "nodes": 4,
            "bs": 8,
            "steps": 100,
            "layers": 64,
            "hidden_size": 2304,
            "seq_length": 1024,
            "heads": 16,
            "deepspeed": True,
            "json": "ds_config_perf_bs8.json",
        }

        self.run_test(test_config)

    def test_perf_8B(self):
        test_config = {
            "mp": 2,
            "gpus": 16,
            "nodes": 4,
            "bs": 16,
            "steps": 100,
            "layers": 72,
            "hidden_size": 3072,
            "seq_length": 1024,
            "heads": 24,
            "deepspeed": True,
            "json": "ds_config_perf_bs16.json",
        }

        self.run_test(test_config)

    def test_perf_20B(self):
        test_config = {
            "mp": 4,
            "gpus": 16,
            "nodes": 4,
            "bs": 8,
            "steps": 50,
            "layers": 111,
            "hidden_size": 3808,
            "seq_length": 1024,
            "heads": 32,
            "ckpt_num_layers": 1,
            "deepspeed": True,
            "json": "ds_config_perf_bs8.json",
        }

        self.run_test(test_config)

    def run_test(self, test_config):
        print("\n")
        print("{0}: starting......".format(self.id()))
        prefix = "gpt2_perf"

        test_file = self.gen_output_name(test_config, prefix)
        self.run_gpt2_test(test_config, test_file)
        exec_time = self.grep_latency_from_file(test_file)

        if exec_time == 0.0:
            print("{0}: no latency found in file {1}".format(self.id(), test_file))
        else:
            print("{0}: execution time per iteration is {1}ms.".format(
                self.id(),
                exec_time))

    def grep_latency_from_file(self, file_name):
        latency = 0.0
        count = 0

        with open(file_name, 'r') as f:
            lines = f.readlines()
            line_filter = "elapsed time per iteration"
            match_number = re.compile(
                'elapsed time per iteration \(ms\): ([-+]?[0-9]+\.?[0-9]*(?:[Ee][-+]?[0-9]+)?)'
            )

            for line in lines:
                if line_filter in line:
                    ms_per_iter = re.findall(match_number, line)
                    latency += float(ms_per_iter[0])
                    count += 1

        if count > 0:
            latency /= count

        return latency


def suite():
    suite = unittest.TestSuite()
    suite.addTest(GPT2PerfTestCase('test_perf_1_5B'))
    suite.addTest(GPT2PerfTestCase('test_perf_4B'))
    suite.addTest(GPT2PerfTestCase('test_perf_8B'))
    suite.addTest(GPT2PerfTestCase('test_perf_20B'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(suite())
