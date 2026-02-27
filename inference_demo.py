import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from lmms_eval.models.simple.early_exit import compute_first_boxed_answer_probs


MODEL_PATH = "IVUL-KAUST/VideoAuto-R1-Qwen3-VL-8B"
EARLY_EXIT_THRESHOLD = 0.98
VIDEO_PATH = "assets/validation_Finance_2.mp4"
PROMPT = (
    "Using the Arbitrage Pricing Theory model shown above, calculate the expected return E(rp) if the risk-free rate increases to 5%. All other risk premiums (RP) and beta (\\beta) values remain unchanged.\n"
    "Options:\n"
    "A. 13.4%\n"
    "B. 14.8%\n"
    "C. 15.6%\n"
    "D. 16.1%\n"
    "E. 16.5%\n"
    "F. 16.9%\n"
    "G. 17.5%\n"
    "H. 17.8%\n"
    "I. 17.2%\n"
    "J. 18.1%\n"
    "Put your final answer in \\boxed{}."
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VideoAuto-R1 model
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
).eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Prepare message for Qwen3-VL
system_prompt = (
    "You are a helpful assistant.\n"
    "FIRST: Output your initial answer inside the first \\boxed{...} without any analysis or explanations. "
    "If you cannot determine the answer without reasoning, output \\boxed{Let's analyze the problem step by step.} instead.\n"
    "THEN: Think through the reasoning as an internal monologue enclosed within <think>...</think>.\n"
    "AT LAST: Output the final answer again inside \\boxed{...}.\n"
    "Output format: \\boxed{...}<think>...</think>\\boxed{...}"
)
messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": VIDEO_PATH, "fps": 2.0, "max_frames": 64},
            {"type": "text", "text": PROMPT},
        ],
    },
]
text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs, video_kwargs = process_vision_info(
    [messages],
    image_patch_size=16,
    return_video_kwargs=True,
    return_video_metadata=True,
)

if video_inputs is not None:
    video_inputs, video_metadatas = zip(*video_inputs)
    video_inputs = list(video_inputs)
    video_metadatas = list(video_metadatas)
else:
    video_metadatas = None

inputs = processor(
    text=text,
    images=image_inputs,
    videos=video_inputs,
    video_metadata=video_metadatas,
    do_resize=False,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
).to(device)


# ============================================================================
# STAGE 1: Initial Answer Inference (Generate up to <think>)
# ============================================================================
with torch.no_grad():
    gen_out_stage1 = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        tokenizer=tokenizer,
        stop_strings=["<think>"],
    )

stage1_gen_ids = gen_out_stage1.sequences[0][len(inputs.input_ids[0]) :]
stage1_answer = processor.decode(stage1_gen_ids, skip_special_tokens=True)
stage1_answer = stage1_answer.split("<think>")[0].strip()  # Extract the part before <think>

# Compute confidence of the first boxed answer
confidence = compute_first_boxed_answer_probs(
    b=0,
    gen_ids=stage1_gen_ids,
    gen_out=gen_out_stage1,
    ans=stage1_answer + "<think>\n</think>\\boxed{}",  # psuedo-answer to ensure boxed content is detected
    task="",
    tokenizer=tokenizer,
)

# ============================================================================
# STAGE 2: Adaptive Reasoning
# ============================================================================
print("\n" + "=" * 60)
print("Initial Model Response:", stage1_answer.strip())
print(f"Confidence Score: {confidence:.4f}")

if confidence >= EARLY_EXIT_THRESHOLD:
    print("Adaptive Inference Status: EARLY EXIT")
    print("-" * 60)
    print(f"Final Model Response:\n{stage1_answer.strip()}")
    print("=" * 60 + "\n")

else:
    print("Adaptive Inference Status: NEEDS CoT (Continuing Generation...)")
    print("-" * 60)

    # 1. Update input_ids with the tokens generated in Stage 1
    new_input_ids = gen_out_stage1.sequences

    # 2. Update attention_mask to account for the new tokens
    new_attention_mask = torch.cat(
        [
            inputs["attention_mask"],
            torch.ones(
                (1, new_input_ids.shape[1] - inputs["input_ids"].shape[1]),
                dtype=inputs["attention_mask"].dtype,
                device=device,
            ),
        ],
        dim=1,
    )

    # 3. Retain vision tensors, swap in the updated text inputs
    stage2_inputs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}
    stage2_inputs["input_ids"] = new_input_ids
    stage2_inputs["attention_mask"] = new_attention_mask

    # 4. Generate the remaining reasoning and final answer
    with torch.no_grad():
        gen_out_stage2 = model.generate(
            **stage2_inputs,
            max_new_tokens=2048 - len(stage1_gen_ids),
            temperature=0.0,
            do_sample=False,
        )

    # Decode the full sequence (Stage 1 + Stage 2 combined)
    final_gen_ids = gen_out_stage2[0][len(inputs.input_ids[0]) :]
    final_answer = processor.decode(final_gen_ids, skip_special_tokens=True)

    print(f"Final Model Response:\n{final_answer.strip()}")
    print("=" * 60 + "\n")
