"""
GradCache for memory-efficient large-batch CLIP training - Simplified version

Algorithm:
1. Forward (no grad): Split batch into chunks, encode, cache embeddings
2. All-gather: Collect cached embeddings from all GPUs
3. Backward: Re-encode each chunk with grad, compute loss with global cache, accumulate gradients

Reference: https://arxiv.org/abs/2101.06983
"""
import torch
import torch.nn as nn
from typing import Tuple, List
from contextlib import nullcontext
from shared.training.distributed import is_distributed, all_gather


class GradCacheCLIP:
    """
    GradCache for CLIP contrastive learning

    Usage:
        grad_cache = GradCacheCLIP(model, chunk_size=8)
        loss = grad_cache.forward_backward(images, texts, device)
        optimizer.step()
    """

    def __init__(self, model: nn.Module, chunk_size: int, use_fp16: bool = True):
        """
        Args:
            model: CT-CLIP model (may be DDP-wrapped)
            chunk_size: Samples per chunk (smaller = less memory)
            use_fp16: Use mixed precision
        """
        # Unwrap DDP if needed
        self.model = model.module if hasattr(model, 'module') else model
        self.chunk_size = chunk_size
        self.use_fp16 = use_fp16

    def _split_batch(self, images: torch.Tensor, texts: dict) -> List[Tuple]:
        """Split batch into chunks"""
        batch_size = images.shape[0]
        chunks = []

        for i in range(0, batch_size, self.chunk_size):
            end = min(i + self.chunk_size, batch_size)
            chunks.append((
                images[i:end],
                {
                    'input_ids': texts['input_ids'][i:end],
                    'attention_mask': texts['attention_mask'][i:end]
                }
            ))

        return chunks

    def _encode_no_grad(
        self,
        images: torch.Tensor,
        texts: dict,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 1: Encode all chunks without gradient (cache)

        Returns:
            (cached_image_latents, cached_text_latents): [batch_size, dim]
        """
        chunks = self._split_batch(images, texts)
        all_image_latents = []
        all_text_latents = []

        with torch.no_grad():
            with torch.amp.autocast('cuda') if self.use_fp16 else nullcontext():
                for img_chunk, txt_chunk in chunks:
                    outputs = self.model(
                        image=img_chunk,
                        text=txt_chunk,
                        device=device,
                        return_loss=False
                    )
                    all_image_latents.append(outputs['image_latents'].detach())
                    all_text_latents.append(outputs['text_latents'].detach())
                    # Clear outputs immediately after extracting latents
                    del outputs

        cached_img = torch.cat(all_image_latents, dim=0)
        cached_txt = torch.cat(all_text_latents, dim=0)

        # Clear intermediate lists
        del all_image_latents, all_text_latents

        return cached_img, cached_txt

    def _compute_chunk_loss(
        self,
        img_chunk: torch.Tensor,
        txt_chunk: torch.Tensor,
        cached_img: torch.Tensor,
        cached_txt: torch.Tensor,
        chunk_start_idx: int,
        temperature: float
    ) -> torch.Tensor:
        """
        Compute loss for one chunk against global cached embeddings

        Args:
            img_chunk: Current chunk image latents (WITH grad)
            txt_chunk: Current chunk text latents (WITH grad)
            cached_img: Global cached image latents (NO grad)
            cached_txt: Global cached text latents (NO grad)
            chunk_start_idx: Global start index
            temperature: CLIP temperature

        Returns:
            loss: Chunk loss
        """
        chunk_size = img_chunk.shape[0]
        global_batch = cached_img.shape[0]

        # Create hybrid: cached (frozen) + current chunk (with grad)
        img_hybrid = cached_img.clone()
        txt_hybrid = cached_txt.clone()
        img_hybrid[chunk_start_idx:chunk_start_idx + chunk_size] = img_chunk
        txt_hybrid[chunk_start_idx:chunk_start_idx + chunk_size] = txt_chunk

        # Similarity matrix
        logits_img = (img_hybrid @ txt_hybrid.T) / temperature
        logits_txt = logits_img.T

        # Labels (diagonal)
        labels = torch.arange(global_batch, device=img_hybrid.device)

        # InfoNCE loss
        loss_img = nn.functional.cross_entropy(logits_img, labels)
        loss_txt = nn.functional.cross_entropy(logits_txt, labels)

        loss = (loss_img + loss_txt) / 2

        # Explicitly delete large intermediate tensors
        del img_hybrid, txt_hybrid, logits_img, logits_txt, labels

        return loss

    def forward_backward(
        self,
        images: torch.Tensor,
        texts: dict,
        device: torch.device,
        temperature: float = 0.07
    ) -> float:
        """
        GradCache forward-backward pass

        Args:
            images: [batch_size, C, D, H, W]
            texts: {'input_ids': ..., 'attention_mask': ...}
            device: Device
            temperature: CLIP temperature (from config)

        Returns:
            loss: Average loss (for logging)
        """
        batch_size = images.shape[0]
        num_chunks = (batch_size + self.chunk_size - 1) // self.chunk_size

        # Step 1: Cache embeddings (no grad)
        cached_img, cached_txt = self._encode_no_grad(images, texts, device)

        # Step 2: All-gather across GPUs
        if is_distributed():
            cached_img = all_gather(cached_img)
            cached_txt = all_gather(cached_txt)

        # Calculate global offset for this GPU
        rank = torch.distributed.get_rank() if is_distributed() else 0
        local_offset = rank * batch_size

        # Step 3: Backward pass - re-encode and accumulate gradients
        chunks = self._split_batch(images, texts)
        total_loss = 0.0

        for chunk_idx, (img_chunk, txt_chunk) in enumerate(chunks):
            with torch.amp.autocast('cuda') if self.use_fp16 else nullcontext():
                outputs = self.model(
                    image=img_chunk,
                    text=txt_chunk,
                    device=device,
                    return_loss=False
                )

                global_start = local_offset + chunk_idx * self.chunk_size

                loss = self._compute_chunk_loss(
                    outputs['image_latents'],
                    outputs['text_latents'],
                    cached_img,
                    cached_txt,
                    global_start,
                    temperature
                )

                # Scale by num_chunks for proper averaging
                (loss / num_chunks).backward()

            total_loss += loss.item()

            # Clear intermediate outputs to free memory
            del outputs, loss

        # Clear cached embeddings after backward pass
        del cached_img, cached_txt

        return total_loss / num_chunks
