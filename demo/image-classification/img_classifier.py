import argparse
import os

import paddle.fluid as fluid
import paddlehub as hub
import numpy as np

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch",      type=int,   default=1,                          help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu",        type=bool,  default=True,                      help="Whether use GPU for fine-tuning.")
parser.add_argument("--checkpoint_dir", type=str,   default="paddlehub_finetune_ckpt",  help="Path to save log data.")
parser.add_argument("--batch_size",     type=int,   default=16,                         help="Total examples' number in batch for training.")
parser.add_argument("--module",         type=str,   default="resnet50",                 help="Module used as feature extractor.")
parser.add_argument("--dataset",        type=str,   default="flowers",                  help="Dataset to finetune.")
parser.add_argument("--slanted_triangle_lr_ration", type=float, default=32, help="ration param for slanted triangle learning rate strategy")
parser.add_argument("--slanted_triangle_lr_cut_frac", type=float, default=0.1, help="cut fraction param for slanted triangle learning rate strategy")
# yapf: enable.

module_map = {
    "resnet50": "resnet_v2_50_imagenet",
    "resnet101": "resnet_v2_101_imagenet",
    "resnet152": "resnet_v2_152_imagenet",
    "mobilenet": "mobilenet_v2_imagenet",
    "nasnet": "nasnet_imagenet",
    "pnasnet": "pnasnet_imagenet"
}


def finetune(args):
    module = hub.Module(name=args.module)
    input_dict, output_dict, program = module.context(trainable=True)

    if args.dataset.lower() == "flowers":
        dataset = hub.dataset.Flowers()
    elif args.dataset.lower() == "dogcat":
        dataset = hub.dataset.DogCat()
    elif args.dataset.lower() == "indoor67":
        dataset = hub.dataset.Indoor67()
    elif args.dataset.lower() == "food101":
        dataset = hub.dataset.Food101()
    elif args.dataset.lower() == "stanforddogs":
        dataset = hub.dataset.StanfordDogs()
    else:
        raise ValueError("%s dataset is not defined" % args.dataset)

    data_reader = hub.reader.ImageClassificationReader(
        image_width=module.get_expected_image_width(),
        image_height=module.get_expected_image_height(),
        images_mean=module.get_pretrained_images_mean(),
        images_std=module.get_pretrained_images_std(),
        dataset=dataset)

    feature_map = output_dict["feature_map"]
    task = hub.create_img_cls_task(
        feature=feature_map, num_classes=dataset.num_labels)

    img = input_dict["image"]
    feed_list = [img.name, task.variable('label').name]

    # Slanted Triangle Learning Rate FineTune Strategy
    lr_strategy = hub.SlantedTriangleLRFineTuneStrategy(
        ratio=args.slanted_triangle_lr_ration,
        cut_fraction=args.slanted_triangle_lr_cut_frac,
        learning_rate=1e-4)

    config = hub.RunConfig(
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        enable_memory_optim=False,
        checkpoint_dir=args.checkpoint_dir,
        strategy=lr_strategy)  # hub.finetune.strategy.DefaultFinetuneStrategy()

    hub.finetune_and_eval(
        task, feed_list=feed_list, data_reader=data_reader, config=config)


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.module in module_map:
        hub.logger.error("module should in %s" % module_map.keys())
        exit(1)
    args.module = module_map[args.module]

    finetune(args)
