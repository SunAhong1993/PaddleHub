# coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import json
import os
import sys
import ast

import six
import pandas
import numpy as np

from paddlehub.commands.base_command import BaseCommand, ENTRY
from paddlehub.common.arg_helper import add_argument, print_arguments
from paddlehub.autofinetune.autoft import PSHE2
from paddlehub.autofinetune.autoft import HAZero
from paddlehub.autofinetune.evaluator import FullTrailEvaluator
from paddlehub.autofinetune.evaluator import ModelBasedEvaluator
from paddlehub.common.logger import logger

import paddlehub as hub


class AutoFineTuneCommand(BaseCommand):
    name = "autofinetune"

    def __init__(self, name):
        super(AutoFineTuneCommand, self).__init__(name)
        self.show_in_help = True
        self.name = name
        self.description = "Paddlehub helps to finetune a task by searching hyperparameters automatically."
        self.parser = argparse.ArgumentParser(
            description=self.__class__.__doc__,
            prog='%s %s <task to be fintuned in python script>' % (ENTRY,
                                                                   self.name),
            usage='%(prog)s',
            add_help=False)
        self.module = None

    def add_params_file_arg(self):
        self.arg_params_to_be_searched_group.add_argument(
            "--param_file",
            type=str,
            default=None,
            required=True,
            help=
            "Hyperparameters to be searched in the yaml format. The number of hyperparameters searched must be greater than 1."
        )

    def add_autoft_config_arg(self):
        self.arg_config_group.add_argument(
            "--popsize", type=int, default=5, help="Population size")
        self.arg_config_group.add_argument(
            "--cuda",
            type=ast.literal_eval,
            default=['0'],
            required=True,
            help="The list of gpu devices to be used")
        self.arg_config_group.add_argument(
            "--round", type=int, default=10, help="Number of searches")
        self.arg_config_group.add_argument(
            "--output_dir",
            type=str,
            default=None,
            help="Directory to model checkpoint")
        self.arg_config_group.add_argument(
            "--evaluate_choice",
            type=str,
            default="fulltrail",
            help="Choices: fulltrail or modelbased.")
        self.arg_config_group.add_argument(
            "--tuning_strategy",
            type=str,
            default="HAZero",
            help="Choices: HAZero or PSHE2.")
    
    def add_python_config_arg(self):
        self.arg_script_config.add_argument(
            "--model_save_dir",
            type=str,
            default='./output',
            help="the best model save dir")
        self.arg_script_config.add_argument(
            "--data_dir",
            type=str,
            default='./data/ILSVRC2012/',
            help="the training data dir")
        self.arg_script_config.add_argument(
            "--use_pretrained",
            type=bool,
            default=True,
            help="whether to use pretrained model")
        self.arg_script_config.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="whether to resume checkpoint")
        self.arg_script_config.add_argument(
            "--save_step",
            type=int,
            default=1,
            help="the steps interval to save checkpoints")
        self.arg_script_config.add_argument(
            "--model",
            type=str,
            default='ResNet50',
            help="the name of network")
        self.arg_script_config.add_argument(
            "--image_h",
            type=int,
            default=224,
            help="the input image h")
        self.arg_script_config.add_argument(
            "--image_w",
            type=int,
            default=224,
            help="the input image w")
        self.arg_script_config.add_argument(
            "--lr_strategy",
            type=str,
            default='piecewise_decay',
            help="the learning rate decay strategy")
        self.arg_script_config.add_argument(
            "--resize_short_size",
            type=int,
            default=256,
            help="the value of resize_short_size")
        self.arg_script_config.add_argument(
            "--use_default_mean_std",
            type=bool,
            default=False,
            help="whether to use label_smoothing")
        
    def execute(self, argv):
        if not argv:
            print("ERROR: Please specify a script to be finetuned in python.\n")
            self.help()
            return False

        self.fintunee_script = argv[0]

        self.parser.prog = '%s %s %s' % (ENTRY, self.name, self.fintunee_script)
        self.arg_params_to_be_searched_group = self.parser.add_argument_group(
            title="Input options",
            description="Hyperparameters to be searched.")
        self.arg_config_group = self.parser.add_argument_group(
            title="Autofinetune config options",
            description=
            "Autofintune configuration for controlling autofinetune behavior, not required"
        )
        self.arg_script_config = self.parser.add_argument_group(
            title="Python script config options",
            description=
            "Python script configuration for controlling autofinetune behavior, part required"
        )

        self.add_params_file_arg()
        self.add_autoft_config_arg()
        self.add_python_config_arg()

        if not argv[1:]:
            self.help()
            return False

        self.args = self.parser.parse_args(argv[1:])
        if self.args.evaluate_choice.lower() == "fulltrail":
            evaluator = FullTrailEvaluator(self.args.param_file,
                                           self.fintunee_script,
                                           self.args)
        elif self.args.evaluate_choice.lower() == "modelbased":
            evaluator = ModelBasedEvaluator(self.args.param_file,
                                            self.fintunee_script,
                                            self.args)
        else:
            raise ValueError(
                "The evaluate %s is not defined!" % self.args.evaluate_choice)

        if self.args.tuning_strategy.lower() == "hazero":
            autoft = HAZero(
                evaluator,
                cudas=self.args.cuda,
                popsize=self.args.popsize,
                output_dir=self.args.output_dir)
        elif self.args.tuning_strategy.lower() == "pshe2":
            autoft = PSHE2(
                evaluator,
                cudas=self.args.cuda,
                popsize=self.args.popsize,
                output_dir=self.args.output_dir)
        else:
            raise ValueError("The tuning strategy %s is not defined!" %
                             self.args.tuning_strategy)

        run_round_cnt = 0
        solutions_ckptdirs = {}
        print("PaddleHub Autofinetune starts.")
        while (not autoft.is_stop()) and run_round_cnt < self.args.round:
            print("PaddleHub Autofinetune starts round at %s." % run_round_cnt)
            output_dir = autoft._output_dir + "/round" + str(run_round_cnt)
            res = autoft.step(output_dir)
            solutions_ckptdirs.update(res)
            evaluator.new_round()
            run_round_cnt = run_round_cnt + 1
        print("PaddleHub Autofinetune ends.")

        with open("./log_file.txt", "w") as f:
            best_hparams = evaluator.convert_params(autoft.get_best_hparams())
            print('-----------------------')
            print('Train to get teh best model!')
            lr = best_hparams[0]
            batch_size= best_hparams[1]
            num_epochs = best_hparams[2]
            gpu_id = int(self.args.cuda[0])
            run_cmd = "python run.py --gpu_id=%s --model_save_dir=%s --data_dir=%s --use_pretrained=%s --checkpoint=%s --save_step=%s --model=%s --image_h=%s --image_w=%s --lr_strategy=%s --resize_short_size=%s --use_default_mean_std=%s --lr=%s --batch_size=%s --num_epochs=%s >%s 2>&1" % \
            (gpu_id, \
             self.args.model_save_dir, \
             self.args.data_dir, \
             self.args.use_pretrained, \
             self.args.checkpoint, \
             self.args.save_step, \
             self.args.model, \
             self.args.image_h, \
             self.args.image_w, \
             self.args.lr_strategy, \
             self.args.resize_short_size, \
             self.args.use_default_mean_std, \
             lr, \
             batch_size, \
             num_epochs, \
             os.path.join(self.args.model_save_dir, 'best_model_log.txt'))
            os.system(run_cmd)
            print('-----------------------')
            print("The final best hyperparameters:")
            f.write("The final best hyperparameters:\n")
            for index, hparam_name in enumerate(autoft.hparams_name_list):
                print("%s=%s" % (hparam_name, best_hparams[index]))
                f.write(hparam_name + "\t:\t" + str(best_hparams[index]) + "\n")
            f.write("\n\n\n")
            f.write("\t".join(autoft.hparams_name_list) + "\toutput_dir\n\n")
            logger.info(
                "The checkpont directory of programs ran with hyperparamemters searched are saved as log_file.txt ."
            )
            print(
                "The checkpont directory of programs ran with hyperparamemters searched are saved as log_file.txt ."
            )
            for solution, ckptdir in solutions_ckptdirs.items():
                param = evaluator.convert_params(solution)
                param = [str(p) for p in param]
                f.write("\t".join(param) + "\t" + ckptdir + "\n\n")

        return True


command = AutoFineTuneCommand.instance()
