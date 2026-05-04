# System Prompts

All system prompts used across the VI-Bench pipeline.

## Inference

| File | Description |
|------|-------------|
| `unified_prompt.txt` | Main VLM inference prompt: analyze video and infer generation prompts |
| `PyVision_Video_Agent.txt` | PyVision-Video agent with code execution for video analysis |
| `VPI_Agent_Stage1_Shot_Segmentation.txt` | VPI-Agent shot boundary detection with code execution |

## Evaluation

| File | Description |
|------|-------------|
| `LLM_Judge.txt` | 5-dimension (subject/action/scene/style/camera) prompt scoring |
| `Human_Preference_Judge.txt` | Human preference experiment judge |
| `Semantic_Unit_Extract.txt` | Extract semantic units from prompts |
| `Semantic_Unit_Judge_Text.txt` | Judge semantic unit recovery (text-based) |
| `Semantic_Unit_Judge_Video.txt` | Judge semantic unit execution (video-based) |
| `Video_EvalAgent_Memory_*.txt` | Video-EvalAgent memory generation (5 dimensions) |
| `Video_EvalAgent_Judge_*.txt` | Video-EvalAgent dimension scoring (5 dimensions) |
| `Video_EvalAgent_OneMemory_Memory.txt` | OneMemory variant: single-pass memory |
| `Video_EvalAgent_OneMemory_Judge.txt` | OneMemory variant: single-pass judge |

## Data Construction

| File | Description |
|------|-------------|
| `Easy_Prompt_Generate.txt` | Generate Easy-level prompts from topic pairs |
| `Medium_Prompt_Generate.txt` | Rewrite prompts with style/camera descriptions |
| `Hard_Prompt_Generate.txt` | Split prompts into multi-shot format |
| `Rewrite_Easy_Prompts.txt` | Expand short descriptions into ~100-word paragraphs |
| `LLM_Generate_Topic.txt` | Label prompt clusters with topic categories |

## Legacy

| File | Description |
|------|-------------|
| `easy_medium_prompt.txt` | Early Easy/Medium inference prompt (superseded by `unified_prompt.txt`) |
| `hard_prompt.txt` | Early Hard inference prompt (superseded by `unified_prompt.txt`) |
