"""
Llama Score Evaluation for Radiology Report Generation

Uses Meta-Llama-3.1-70B-Instruct as an LLM judge to evaluate clinical accuracy
of generated radiology reports against ground truth.

Scoring: 0-10 scale based on clinical accuracy, completeness, and correctness.

Usage as function:
    from shared.metrics.medical.llama_score import compute_llama_score
    
    score = compute_llama_score(
        hypotheses=["generated report 1", "generated report 2"],
        references=["ground truth 1", "ground truth 2"],
        task_type="report_generation"
    )
    print(f"Llama Score: {score['llama_score']:.2f}")

Usage as script:
    python llama_score.py --predictions predictions.jsonl --model_path /path/to/model

Predictions JSONL from stage3_visd evaluate always includes "question"; the script requires it
and does not use fixed prompts when loading from file.
"""

import re
import json
import torch
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Fallback only when questions not provided (e.g. programmatic call without JSONL)
TASK_QUESTIONS = {
    "report_generation": "Please generate a comprehensive radiology report for this chest CT scan.",
    "long_answer": "",
    "short_answer": "",
    "multiple_choice": "",
}

# Task type display names (matching CT-CHAT)
TASK_TYPE_NAMES = {
    "report_generation": "Report generation",
    "long_answer": "Long answer",
    "short_answer": "Short answer",
    "multiple_choice": "Multiple choice",
}


# CT-CHAT's exact system prompt
SYSTEM_PROMPT = """Role: You are an expert medical evaluator specializing in the assessment of clinical accuracy for AI-generated reports and answers derived from 3D chest CT volumes. Your task is to compare the AI-generated answers with the provided ground truth answers and assign a clinical accuracy score based on specific evaluation criteria. Your assessment should focus on precision, relevance, and clinical soundness.

Scoring:

    •	10/10: The generated response is fully aligned with the ground truth—completely accurate, relevant, comprehensive, clear, and clinically sound.
    •	7-9/10: The generated response is mostly accurate, with minor discrepancies or omissions that do not significantly affect clinical interpretation.
    •	4-6/10: The generated response contains noticeable errors or omissions that could impact clinical understanding, though some correct information is present.
    •	1-3/10: The generated response has significant inaccuracies, irrelevant content, or lacks essential information, potentially leading to clinical misunderstanding.
    •	0/10: The generated response is completely incorrect or irrelevant, offering no clinical value.

Question Types:

For each evaluation, you will be provided with the question type, question, ground truth answer, and generated answer.

    •	Report Generation: Requires a detailed narrative report summarizing findings from the 3D chest CT volume, covering all relevant clinical observations, including pathology, location, and significant findings. The report should follow the CT report format. Consider the formatting of the output for this question in addition to clinical accuracy. Non-existent pathologies are not necessarily required to be mentioned in the generated report. Do not deduct points if this is the case (e.g., "no pleural effusion" might not be in the output, but if "pleural effusion" is present, it must be included). Deduct points if the model hallucinates pathologies that are not in the ground truth.
    •	Long Answer Questions: Require comprehensive, detailed responses that thoroughly address the question, including necessary explanations or justifications.
    •	Short Answer Questions: Require brief, direct answers addressing the question with essential information, such as identifying pathology or confirming a diagnosis.
    •	Multiple Choice Questions: Require the selection of the most clinically accurate option from predefined choices, matching the ground truth answer. There should be only one correct answer for this question type. If more than one answer is provided, assign a very low score. If the answer is incorrect, assign a very low score as well.

Instructions:

Please review the question type, question, ground truth answer, and generated answer. Assign a score out of 10 based on the evaluation criteria above. Provide only the score without additional commentary or explanation."""


def parse_score(text: str) -> Optional[float]:
    """
    Parse score from LLM output
    
    Tries to extract a number between 0-10 from the text.
    Handles formats like: "8", "8/10", "Score: 8", "8.5", etc.
    
    Args:
        text: LLM output text
    
    Returns:
        Score as float, or None if parsing fails
    """
    # Remove common prefixes
    text = text.lower().strip()
    text = re.sub(r'^(score|rating|evaluation):\s*', '', text)
    
    # Try to find a number (integer or decimal)
    patterns = [
        r'(\d+\.?\d*)\s*/\s*10',  # "8/10" or "8.5/10"
        r'(\d+\.?\d*)\s*out of\s*10',  # "8 out of 10"
        r'^(\d+\.?\d*)$',  # Just "8" or "8.5"
        r'(\d+\.?\d*)',  # Any number
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                score = float(match.group(1))
                # Validate range
                if 0 <= score <= 10:
                    return score
            except (ValueError, IndexError):
                continue
    
    return None


def create_evaluation_prompt(
    question: str,
    ground_truth: str,
    generated: str,
    task_type: str
) -> str:
    """
    Create evaluation prompt for LLM judge
    
    Args:
        question: Question text (or task description)
        ground_truth: Reference answer
        generated: Generated answer
        task_type: Type of task (report_generation, long_answer, etc.)
    
    Returns:
        Formatted prompt string
    """
    task_name = TASK_TYPE_NAMES.get(task_type, task_type)
    
    user_prompt = f"""Question type: {task_name}

Question: {question}

Ground truth answer: {ground_truth}

Generated answer: {generated}"""
    
    return user_prompt


def compute_llama_score(
    hypotheses: List[str],
    references: List[str],
    task_type: str = "report_generation",
    questions: Optional[List[str]] = None,
    model_path: str = "./checkpoints/Meta-Llama-3-70B-Instruct",
    temperature: float = 0.0,
    max_tokens: int = 256,
    tensor_parallel_size: int = 1
) -> Dict[str, float]:
    """
    Compute Llama Score using Meta-Llama-3.1-70B-Instruct as judge
    
    Evaluates clinical accuracy of generated radiology reports using an LLM judge.
    Scores range from 0-10, with higher scores indicating better clinical accuracy.
    
    Args:
        hypotheses: List of generated reports/answers
        references: List of ground truth reports/answers
        task_type: Type of task ('report_generation', 'long_answer', 'short_answer', 'multiple_choice')
        questions: Optional list of questions (if None, uses default for task_type)
        model_path: Path to Llama model
        temperature: Sampling temperature (0 = deterministic)
        max_tokens: Maximum tokens to generate
        tensor_parallel_size: Number of GPUs for model parallelism (uses device_map="auto")
    
    Returns:
        {'llama_score': float}  # Average score across all samples
    
    Example:
        >>> hyps = ["Cardiomegaly present.", "Normal chest CT."]
        >>> refs = ["Enlarged heart noted.", "No acute findings."]
        >>> result = compute_llama_score(hyps, refs)
        >>> print(f"Score: {result['llama_score']:.2f}/10")
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"Length mismatch: {len(hypotheses)} hypotheses vs {len(references)} references"
        )
    
    if len(hypotheses) == 0:
        raise ValueError("Empty input: need at least one hypothesis-reference pair")
    
    # Check model exists
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            f"Please ensure Meta-Llama-3.1-70B-Instruct is downloaded."
        )
    
    # Use default questions if not provided
    if questions is None:
        default_question = TASK_QUESTIONS.get(task_type, "")
        questions = [default_question] * len(hypotheses)
    
    if len(questions) != len(hypotheses):
        raise ValueError(
            f"Length mismatch: {len(questions)} questions vs {len(hypotheses)} hypotheses"
        )
    
    print("=" * 80)
    print("Computing Llama Score (LLM-as-Judge Evaluation)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Task type: {task_type}")
    print(f"Number of samples: {len(hypotheses)}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print()
    
    # Initialize model and tokenizer (Transformers)
    print("Loading model with Transformers...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("Model loaded successfully!")
    print()
    
    # Create prompts for all samples
    print("Creating evaluation prompts...")
    prompts = []
    for question, reference, hypothesis in zip(questions, references, hypotheses):
        user_prompt = create_evaluation_prompt(
            question=question,
            ground_truth=reference,
            generated=hypothesis,
            task_type=task_type
        )
        
        # Format as chat messages for Llama-3-Instruct
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        prompts.append(messages)
    
    # Generate evaluations
    print(f"Generating evaluations for {len(prompts)} samples...")
    print("This may take a while for large batches...")
    print()
    
    scores = []
    failed_parses = 0
    
    with torch.no_grad():
        for i, messages in enumerate(tqdm(prompts, desc="Evaluating")):
            # Apply chat template and tokenize
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            device = next(model.parameters()).device
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            # Generate
            generate_kwargs = {
                "max_new_tokens": max_tokens,
                "top_p": 1.0,
                "pad_token_id": tokenizer.eos_token_id,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
            else:
                generate_kwargs["do_sample"] = False
            
            outputs = model.generate(**inputs, **generate_kwargs)
            
            # Decode only the generated part (exclude input)
            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_length:]
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            ).strip()
            
            score = parse_score(generated_text)
            
            if score is not None:
                scores.append(score)
            else:
                print(f"  Warning: Failed to parse score from output {i+1}: '{generated_text[:50]}...'")
                failed_parses += 1
                scores.append(0.0)
    
    # Calculate average
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    print()
    print("=" * 80)
    print("Llama Score Results")
    print("=" * 80)
    print(f"Total samples: {len(scores)}")
    print(f"Failed parses: {failed_parses}")
    print(f"Average Score: {avg_score:.4f} / 10")
    print(f"Min Score: {min(scores):.1f}")
    print(f"Max Score: {max(scores):.1f}")
    print("=" * 80)
    
    return {
        'llama_score': float(avg_score),
        'num_samples': len(scores),
        'failed_parses': failed_parses,
        'min_score': float(min(scores)),
        'max_score': float(max(scores))
    }


def main():
    """
    Command-line interface for computing Llama Score
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute Llama Score for radiology report generation evaluation"
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions JSONL (must have "prediction", "reference", and "question" per row)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default="./checkpoints/Meta-Llama-3-70B-Instruct",
        help='Path to Meta-Llama-3.1-70B-Instruct model'
    )
    parser.add_argument(
        '--task_type',
        type=str,
        default='report_generation',
        choices=['report_generation', 'long_answer', 'short_answer', 'multiple_choice'],
        help='Type of evaluation task'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (default: 0.0 for deterministic)'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=256,
        help='Maximum tokens to generate (default: 256)'
    )
    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=1,
        help='Kept for API compatibility. Transformers uses device_map="auto" to use all available GPUs (default: 1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save results JSON (optional)'
    )
    
    args = parser.parse_args()
    
    # Load predictions; require "question" per row (stage3_visd evaluate always writes it)
    print(f"Loading predictions from: {args.predictions}")
    predictions = []
    references = []
    questions = []
    with open(args.predictions, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            predictions.append(data['prediction'])
            references.append(data['reference'])
            if 'question' not in data:
                raise KeyError(
                    f"Row {i + 1} missing 'question'. "
                    "Predictions JSONL must include 'question' (e.g. from stage3_visd evaluate)."
                )
            questions.append(data['question'])
    if len(questions) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(questions)} questions vs {len(predictions)} predictions"
        )
    print(f"Loaded {len(predictions)} prediction-reference pairs (with questions)")
    print()
    
    # Compute Llama Score
    results = compute_llama_score(
        hypotheses=predictions,
        references=references,
        task_type=args.task_type,
        questions=questions,
        model_path=args.model_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    # Save results if output path provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print()
        print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
