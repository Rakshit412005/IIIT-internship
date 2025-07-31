from llama_cpp import Llama

# Load the model (change path to your model)
llm = Llama(
    model_path="/home/pi/intern_llma/obj_detec/llama.cpp/models-phi-2/phi-2.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=20,  # If using GPU
    use_mlock=True    # Optional, to avoid swapping
)

# Prompt generator
def generate_cot_prompt(task, object_list):
    return f"""
You are an intelligent assistant that helps choose the most suitable object for a task by reasoning step by step.

### Task:
{task}

### Detected objects:
{', '.join(object_list)}

### Step 1: For each object, explain whether and how it can be used to perform the task.

### Step 2: For objects that are usable, compare their effectiveness and prioritize them based on suitability.

### Step 3: Choose the highest-priority object and explain why.


Let’s reason step by step.
""".strip()

# Inference function
def get_affordance_reasoning(task, object_list):
    prompt = generate_cot_prompt(task, object_list)
    output = llm(prompt, max_tokens=256, stop=["</s>"])
    return output["choices"][0]["text"].strip()


# 🧪 Example use
if __name__ == "__main__":
    #task = "open a parcel"
    #object_list = ["knife", "scissors", "key","person","blade"]

    result = get_affordance_reasoning(task, object_list)
    print("\n🔎 Affordance Reasoning Output:\n")
    print(result)
