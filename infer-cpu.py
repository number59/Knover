#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference main program."""

import argparse
from collections import defaultdict
import json
import os
import subprocess
import time

import paddle.fluid as fluid

import models
import tasks
from utils import Timer
from utils.args import parse_args


def setup_args():
    """
    Setup arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="output")
    parser.add_argument("--infer_file", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)

    parser.add_argument("--log_steps", type=int, default=1)

    models.add_cmdline_args(parser)
    tasks.add_cmdline_args(parser)

    args = parse_args(parser)
    args.load(args.config_path, "Model")
    args.run_infer = True # only build infer program
    print(json.dumps(args, indent=2))
    return args


def infer(args):
    """
    Inference main function.
    """
    place = fluid.CPUPlace()

    task = tasks.create_task(args)
    model = models.create_model(args, place)
    infer_generator = task.reader.data_generator(
        input_file=args.infer_file,
        phase='test',
        is_infer=True
    )

    # run inference
    timer = Timer()
    timer.start()
    infer_out = {}
    for step, data in enumerate(infer_generator(), 1):
        predictions = task.infer_step(model, data)
        for info in predictions:
            infer_out[info["data_id"]] = info
        if step % args.log_steps == 0:
            time_cost = timer.pass_time
            print(f"\tstep: {step}, time: {time_cost:.3f}, "
                  f"speed: {step / time_cost:.3f} steps/s")

    time_cost = timer.pass_time
    print(f"[infer] steps: {step} time cost: {time_cost}, "
          f"speed: {step / time_cost} steps/s")

    # save inference outputs
    inference_output = os.path.join(args.save_path, args.output_name)
    with open(inference_output, "w") as f:
        for data_id in sorted(infer_out.keys(), key=lambda x: int(x)):
            f.write(str(infer_out[data_id][args.output_name]) + "\n")
    print(f"save inference result into: {inference_output}")

    return


if __name__ == "__main__":
    args = setup_args()
    infer(args)
