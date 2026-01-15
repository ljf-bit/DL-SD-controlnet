import json
import datasets

# 指向你刚才生成的 jsonl 文件的绝对路径
JSONL_PATH = "/root/autodl-tmp/datasets/dataset_controlnet.jsonl"

class ControlNetDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "image": datasets.Image(),              # 目标图 (RGB)
                "conditioning_image": datasets.Image(), # 条件图 (语义+边缘)
                "text": datasets.Value("string"),       # 提示词
            })
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": JSONL_PATH},
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for id_, line in enumerate(f):
                data = json.loads(line)
                yield id_, {
                    "image": data["image"],
                    "conditioning_image": data["conditioning_image"],
                    "text": data["text"],
                }