"""
CT-CLIP - Contrastive Learning for CT and Text

Vision-language model for aligning CT scans with radiology reports
"""
from shared.models.ctclip.ct_clip import CTCLIP, TextTransformer

__all__ = ['CTCLIP', 'TextTransformer']
