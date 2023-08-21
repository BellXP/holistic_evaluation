import torch

from .llama2_accessory.model.meta import MetaModel
from .llama2_accessory.data.alpaca import transform_val, transform_train, format_prompt
from fairscale.nn.model_parallel import initialize as fs_init
from .llama2_accessory.util.misc import init_distributed_mode
from .llama2_accessory.util.tensor_parallel import load_tensor_parallel_model_list
import torch.distributed as dist

from . import get_image


llama_type = "llama_ens"
tokenizer_path = "/mnt/petrelfs/share_data/gaopeng/llama-accessory/llama_v2/tokenizer.model"
llama_config = "/mnt/petrelfs/share_data/gaopeng/llama-accessory/llama_v2/llama-2-13b/params.json"
pretrained_path = "/mnt/petrelfs/share_data/gaopeng/llama-accessory/finetune/mm/ens_llama_13B/doc_mixed/epoch2"


class Args:
    def __init__(self) -> None:
        self.dist_on_itp = False
        self.gpu = 0
        self.rank = 0
        self.world_size = 2
        self.dist_url = 'env://'
init_distributed_mode(Args())
fs_init.initialize_model_parallel(2)


class TestLLama2Accessory:
    def __init__(self, device=None) -> None:
        model = MetaModel(llama_type, [llama_config], tokenizer_path, with_visual=True)
        print(f"load pretrained from {pretrained_path}")
        load_tensor_parallel_model_list(model, [pretrained_path])
        # print("Model = %s" % str(model))
        model.bfloat16().cuda()
        model.eval()

        self.model = model

        if dist.get_rank() != 0:
            while True:
                dist.barrier()
                input_data = [None for _ in range(5)]
                dist.broadcast_object_list(input_data)
                prompts, imgs, max_new_tokens, temperature, top_p = input_data
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    _ = self.model.generate(prompts, imgs.cuda(), max_gen_len=max_new_tokens, temperature=temperature, top_p=top_p)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256, temperature=0.1, top_p=0.75):
        image = get_image(image)
        image = transform_val(image).unsqueeze(0)
        prompt = question

        dist.barrier()
        dist.broadcast_object_list([prompt, image, max_new_tokens, temperature, top_p])

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            results = self.model.generate([prompt], image.cuda(), max_gen_len=max_new_tokens, temperature=temperature, top_p=top_p)
        text_output = results[0].strip()

        return text_output

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256, temperature=0.1, top_p=0.75):
        imgs = [get_image(img) for img in image_list]
        # imgs = [transform_val(x) for x in imgs]
        imgs = [transform_val(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0)
        # prompts = [question for question in question_list]
        prompts = [format_prompt(question) for question in question_list]

        dist.barrier()
        dist.broadcast_object_list([prompts, imgs, max_new_tokens, temperature, top_p])

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            results = self.model.generate(prompts, imgs.cuda(), max_gen_len=max_new_tokens, temperature=temperature, top_p=top_p)
        results = [result.strip() for result in results]

        return results
