import torch
import torch.nn as nn
import json
from .tokenizer import Tokenizer
from .utils import sample_top_p
from .LLM.llama_qformerv2_peft import Transformer, ModelArgs
from .LLM.llama_multimodal import BERTTransformer

import clip


class MetaModel(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, llama_config, tokenizer_path, reversible_grad=False,
                 clip_model='ViT-L/14@336px', v_embed_dim=1024, query_len=577,
                 max_seq_len=2048, max_batch_size=32):
        super().__init__()

        with open(llama_config, "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        )
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = self.tokenizer.n_words
        if reversible_grad:
            if hasattr(model_args, "reversible_gradient"):
                model_args.reversible_gradient = True
            else:
                raise KeyError (f"{ModelArgs} object has no attribute reversible_gradient")

        model = Transformer(model_args)
        self.llma = model

        # 2. clip
        self.clip, self.clip_transform = clip.load(clip_model)

        clip_dim = self.clip.visual.proj.shape[1]
        self.clip_proj = nn.Linear(clip_dim, v_embed_dim)
        self.clip_proj_norm = nn.LayerNorm(v_embed_dim)

        self.query_len = query_len
        self.query_layer = model_args.n_layers

        visual_model_args = ModelArgs(dim=1024, n_layers=16, n_heads=8, max_seq_len=query_len)
        visual_model_args.vocab_size = 1024
        self.visual_blocks = BERTTransformer(visual_model_args)
        self.visual_proj = nn.Linear(v_embed_dim, model_args.dim)
        self.visual_proj_norm = nn.LayerNorm(model_args.dim)

        # 3. adapter query
        self.adapter_query = nn.Embedding(
            query_len * self.query_layer, model_args.dim)

        # 4. criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #        print(f"Trainable param: {name}, {param.shape}, {param.dtype}")
        # count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print(f"Parameter count : {count}")
        # exit(0)

    def clip_encode_image(self, x):
        # modified from CLIP
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                      x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x

    def forward_visual(self, imgs):
        clip_feats = self.clip_encode_image(imgs)
        clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))

        visual_query = clip_feats
        visual_query = self.visual_blocks(visual_query)

        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)

        return visual_query

    def forward(self, tokens, labels, imgs):
        _bsz, seqlen = tokens.shape

        visual_proj = self.forward_visual(imgs).unsqueeze(0)  # (1, bs, seq_len, dim)
        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)  # (self.query_layer, 1, seq_len, dim)
        prompt_all_layers = adapter + visual_proj

        output = self.llma(tokens, prompt_all_layers)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
           c_loss = output.mean() * 0
        else:
           c_loss = self.criterion(output.reshape(-1, 32000), labels.flatten())
        pred = 0
        mask = 0
        return c_loss, c_loss, pred, mask
    
    @torch.inference_mode()
    def forward_inference(self, visual_query, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.llma.tok_embeddings(tokens)
        freqs_cis = self.llma.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.llma.layers[:-1 * self.query_layer]:
            h = layer(h, start_pos, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llma.layers[-1 * self.query_layer:]:
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            dynamic_adapter = dynamic_adapter + visual_query
            h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1

        h = self.llma.norm(h)
        output = self.llma.output(h[:, -1, :])

        return output.float()

    @torch.inference_mode()
    def generate(self, imgs, prompts,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
    ):
        bsz = len(imgs)
        params = self.llma.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        assert len(imgs) == len(prompts)

        with torch.cuda.amp.autocast():
            visual_query = self.forward_visual(imgs)

        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
        
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits = self.forward_inference(visual_query, tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded