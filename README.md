

---

# Project Overview: Task-Driven Object Detection with LLM + TFLite on Raspberry Pi

## Objective

The goal of this project is to develop an **intelligent, task-aware object detection system** that can:

* Detect objects in real time using a camera,
* Accept a human-given task (e.g., *"open a parcel"*, *"sit comfortably"*),
* Use an **LLM (Local Large Language Model)** like Phi-2 or Gemma to reason about **which detected objects are relevant** to the task,
* Optionally **rank** the detected objects based on their **suitability**,
* Highlight only the relevant objects with bounding boxes,
* Measure performance metrics such as object detection time and LLM inference time.

---

## System Architecture

### Components:

| Module                                    | Role                                                           |
| ----------------------------------------- | -------------------------------------------------------------- |
| **TFLite MobileNet-SSD**                  | Real-time object detection                                     |
| **Picamera2**                             | Captures live feed/images from Pi Camera                       |
| **Llama.cpp (with Phi-2 / Gemma models)** | Performs task-driven reasoning using natural language          |
| **Custom Prompt Generator**               | Builds dynamic prompts from detected objects and selected task |
| **Chain-of-Thought (CoT)**                | Enhances LLM reasoning accuracy                                |
| **Ranking Module**                        | Ranks objects based on task suitability                        |
| **Performance Monitor**                   | Logs timing and token statistics for LLM processing            |

---

## Capabilities

* **Real-time live object detection** on Raspberry Pi.
* **LLM-powered reasoning** with task-based filtering.
* Choose from **14 predefined CoTDet tasks**.
* **Gemma and Phi-2 LLM models supported locally**.
* **Bounding box visualization** for task-relevant objects.
* **Execution timing and token rate monitoring**.
* **Object ranking with suitability scores**.
* **CoT reasoning prompts** for better LLM answers.

---

## Sample Use Case

> Live camera detects: `["person", "chair", "knife", "monitor"]`
> Task given: `"open parcel"`
> LLM Response: `["knife"]`
> Only `knife` is shown in bounding box.

> Task: `"sit comfortably"`
> LLM ranks: `{ "chair": 1, "person": 2 }`
> Visual feedback only on `chair` (top rank).

---

## Tools Used

* **Python**
* **OpenCV**
* **TFLite Runtime**
* **Llama.cpp**
* **Hugging Face GGUF models (Phi-2, Gemma)**

---

## Major Features Implemented

1. Image and live-feed object detection
2. Integration of local LLM for task-based reasoning
3. Timing logs: Detection time, LLM response time, Total time
4. Token analysis: Prompt & output token count, tokens/sec
5. CoT (Chain-of-Thought) reasoning
6. Object ranking based on suitability for a task

---

## Project Foundation (Progress Steps)

1. **Built static object detector with TFLite**
2. **Added live feed object detection with PiCamera2**
3. **Installed and integrated Llama.cpp**
4. **Used Phi-2 model (from Hugging Face) for task reasoning**
5. **Handled FP16 error by disabling it in llama.cpp**
6. **Switched to Gemma for performance comparison**
7. **Logged inference time and token stats**
8. **Introduced 14 CoTDet tasks**
9. **Added CoT prompting for better reasoning**
10. **Final step: Implemented object ranking for suitability**

---

## Final Output

* A **self-contained intelligent assistant** on Raspberry Pi that:

  * Uses TFLite + LLM to detect and reason,
  * Accepts tasks via terminal,
  * Only highlights useful objects,
  * Works offline using local models.

---




---

## Prerequisites

* Raspberry Pi 4B or 5

* PiCamera2 or USB camera (configured via `libcamera`)

* Python 3.9+

* Create a virtual environment:

  ```bash
  python3 -m venv llm_env
  source llm_env/bin/activate
  ```

* Install dependencies:

  ```bash
  pip install opencv-python tflite-runtime numpy llama-cpp-python
  sudo apt install libcamera-dev
  ```

---

## Step 1: Static Image Object Detection using TFLite

###  Files Required:

* `ssd-mobilenet-v1-tflite-default-v1.tflite`
* `coco_labels_dict.json`

### Code:

```python
import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import json

interpreter = tflite.Interpreter(model_path="ssd-mobilenet-v1-tflite-default-v1.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("coco_labels_dict.json") as f:
    labels = list(json.load(f).keys())

image = cv2.imread("test.jpg")
input_data = cv2.resize(image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
input_data = np.expand_dims(input_data, axis=0)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

boxes = interpreter.get_tensor(output_details[0]['index'])[0]
classes = interpreter.get_tensor(output_details[1]['index'])[0]
scores = interpreter.get_tensor(output_details[2]['index'])[0]

for i in range(len(scores)):
    if scores[i] > 0.5:
        class_id = int(classes[i])
        print("Detected:", labels[class_id])
```

---

## Step 2: Live Feed Object Detection

###  Install dependencies:

```bash
pip install picamera2 opencv-python
```

### Code:

```python
from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

while True:
    frame = picam2.capture_array()
    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
```

---

## Step 3: Installing and Using LLM with llama.cpp

###  Installation:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build .
cmake --build build -j4
```

###  Install Python bindings:

```bash
pip install llama-cpp-python
```

---

## Step 4: Use Phi-2 model for Task-Driven Object Detection

###  Download:

```bash
wget https://huggingface.co/microsoft/phi-2/resolve/main/phi-2.Q4_K_M.gguf
```

### Code:

```python
from llama_cpp import Llama
llm = Llama(model_path="phi-2.Q4_K_M.gguf")

prompt = "Which object is suitable for opening a parcel: knife, chair, person?"
response = llm(prompt, max_tokens=100)
print(response['choices'][0]['text'])
```

---

## Step 5: Switching from Phi-2 to Gemma

###  Download with Hugging Face Token:

```bash
wget --header="Authorization: Bearer <your_token>" \
https://huggingface.co/google/gemma-1.1-7b-it-GGUF/resolve/main/gemma-1.1-7b-it.Q4_K_M.gguf
```

### Code:

```python
llm = Llama(model_path="gemma-1.1-7b-it.Q4_K_M.gguf")
```

---

## Step 6: Timing Analysis

We added timing metrics to compare model speed.

### Code:

```python
import time
prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
llm_start = time.time()
response = llm(prompt, max_tokens=100)
llm_end = time.time()
gen_tokens = len(llm.tokenize(response['choices'][0]['text'].encode("utf-8")))

# === Performance summary ===
        gen_tokens = len(llm.tokenize(raw.encode("utf-8")))
        print("\n PERFORMANCE SUMMARY:")
        print(f"• Load Duration:        {(end_load - start_load):.2f} sec")
        print(f"• Total LLM Time:       {(llm_end - llm_start):.2f} sec")
        print(f"• Prompt Tokens:        {prompt_tokens}")
        print(f"• Generated Tokens:     {gen_tokens}")
        print(f"• Prompt Eval Rate:     {prompt_tokens / (llm_end - llm_start):.2f} tokens/sec")
        print(f"• Generation Eval Rate: {gen_tokens / (llm_end - llm_start):.2f} tokens/sec")
```

---

## Step 7: Add 14 CoTDet Tasks to Code

### Code: to get an idea of what our actual code should look like

# === CoTDet task list ===
```python
tasks = {
    1: "step on",
    2: "sit comfortably",
    3: "place flowers",
    4: "get potatoes out of fire",
    5: "water plant",
    6: "get lemon out of tea",
    7: "dig hole",
    8: "open bottle of beer",
    9: "open parcel",
    10: "serve wine",
    11: "pour sugar",
    12: "smear butter",
    13: "extinguish fire",
    14: "communication"
}

```
```python
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite
from llama_cpp import Llama
import json
import re
import ast

# === Load COCO label descriptions ===
with open("/home/pi/intern_llma/obj_detec/models/coco_labels_dict.json", "r") as f:
    label_desc_map = json.load(f)
labels = list(label_desc_map.keys())

# === Load LLM (Phi-2)
llm = Llama(
    model_path="/home/pi/intern_llma/obj_detec/llama.cpp/build/models/phi-2.Q4_K_M.gguf",
    n_ctx=512,
    n_threads=4,
    n_gpu_layers=0
)

# === Load TFLite object detection model
interpreter = tflite.Interpreter(model_path="/home/pi/intern_llma/obj_detec/models/ssd-mobilenet-v1-tflite-default-v1.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# === CoTDet task list
tasks = {
    1: "step on",
    2: "sit comfortably",
    3: "place flowers",
    4: "get potatoes out of fire",
    5: "water plant",
    6: "get lemon out of tea",
    7: "dig hole",
    8: "open bottle of beer",
    9: "open parcel",
    10: "serve wine",
    11: "pour sugar",
    12: "smear butter",
    13: "extinguish fire",
    14: "pound carpet"
}

# === Start Pi Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

print(" Live feed started. Press 's' to capture, 'q' to quit.")

while True:
    frame = picam2.capture_array()
    cv2.imshow("Live Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        total_start = time.time()

        # === Object detection
        det_start = time.time()
        resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
        input_data = np.expand_dims(resized, axis=0)

        if input_dtype == np.float32:
            input_data = input_data.astype(np.float32) / 255.0
        else:
            input_data = input_data.astype(np.uint8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        det_end = time.time()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

        detected_objects = []
        object_locations = []

        for i in range(num_detections):
            if scores[i] > 0.5:
                class_id = int(classes[i])
                label = labels[class_id] if class_id < len(labels) else "Unknown"
                detected_objects.append(label)

                ymin, xmin, ymax, xmax = boxes[i]
                left = int(xmin * frame.shape[1])
                top = int(ymin * frame.shape[0])
                right = int(xmax * frame.shape[1])
                bottom = int(ymax * frame.shape[0])
                object_locations.append((label, left, top, right, bottom))

        print(" Detected objects:", detected_objects)

        if not detected_objects:
            print("️ No high-confidence objects found.")
            continue

        try:
            task_number = int(input(" Enter task number (1–14): "))
            task_name = tasks[task_number]
        except:
            print(" Invalid task number.")
            continue

        # === Prompt construction ===
        prompt = (
            f"You are an expert object reasoning assistant.\n"
            f"Task: '{task_name}'\n"
            f"Objects: {', '.join(detected_objects)}\n"
            f"\nReturn only a Python list of object names relevant to the task. "
            f"No explanation. No code. Only output a list like this: ['knife', 'scissors']"
        )

        print("\n Prompt sent to LLM:\n", prompt)

        # === LLM call
        llm_start = time.time()
        output = llm(prompt, max_tokens=100, stop=["</s>"])
        llm_end = time.time()

        raw_output = output['choices'][0]['text'].strip()
        print("\n LLM Response:\n", raw_output)

        # === Parse response safely
        try:
            match = re.search(r"\[.*?\]", raw_output)
            if match:
                matched_list = ast.literal_eval(match.group(0))
            else:
                matched_list = []
        except Exception as e:
            print(" Could not parse LLM output:", e)
            matched_list = []

        # === Draw matched object boxes
        timestamp = int(time.time())
        filtered_img = frame.copy()
        for label, left, top, right, bottom in object_locations:
            if label in matched_list:
                cv2.rectangle(filtered_img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(filtered_img, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        result_img = f"task_{task_number}_filtered_{timestamp}.jpg"
        cv2.imwrite(result_img, filtered_img)
        print(f"\n Saved filtered result: {result_img}")

        # === Timing summary
        total_end = time.time()
        print("\n TIMING SUMMARY")
        print(f"•  Object detection: {(det_end - det_start):.2f} sec")
        print(f"•  LLM reasoning:    {(llm_end - llm_start):.2f} sec")
        print(f"•  Total time:       {(total_end - total_start):.2f} sec")

    elif key == ord('q'):
        print(" Exiting...")
        break

cv2.destroyAllWindows()

```
---

## Step 8: Chain-of-Thought (CoT) Reasoning

###  CoT Prompt:
```python

def generate_cot_prompt(task, object_list):
    prompt = f"""
You are an intelligent assistant that helps choose the most suitable object for a task by reasoning step by step.

### Task:
{task}

### Detected objects:
{', '.join(object_list)}

### Step 1: For each object, explain whether and how it can be used to perform the task.

### Step 2: For objects that are usable, compare their effectiveness and prioritize them based on suitability.

### Step 3: Choose the highest-priority object and explain why.

Let’s reason step by step.
"""
    return prompt.strip()
```

###  `test10.py`:

```python
def get_affordance_reasoning(task, objects):
    prompt = f"Task: '{task}'\nObjects: {', '.join(objects)}\nReturn only Python list."
    response = llm(prompt, max_tokens=100)
    return response['choices'][0]['text']
```

---

## Step 9: Add Object Ranking (Suitability Score)

```python   
ranked_objects.sort(reverse=True)
print("\n Ranked Affordance Results:\n")
for score, label, reason in ranked_objects:
    print(f"{label} (score {score}): {reason}")

### Expected Output:
```
```python
{"knife": 1, "scissors": 2, "pen": 3}
```

---

## Step 10: Using smolVLM model
* go to https://github.com/AIWinetrermuteAI/llama.cpp
* clone the repository
* get the model from hugging face , model name - SmolVLM-256-Instruct
* follow the instructions given in the github README file
* one error will come like this :
 
  

##  Error Handling: FP16 Not Supported

###  Error:

```bash
llama_kv_cache_unified: fp16 not supported
```

### Fix:

We **disabled FP16** in `llama.cpp` by changing:

```cpp
#define GGML_USE_FP16 0  # or disable fp16 in CMake
```

Or rebuild without FP16:

```bash
cmake -DLLAMA_F16C=OFF .
cmake --build build -j4
```

---

## Final Result

A live system where:

* Objects are detected from the Pi Camera.
* You choose a task from 14 options.
* LLM filters relevant objects.
* LLM response time is logged.
* Ranking can be applied.
* Optional: Chain-of-Thought reasoning boosts accuracy

---


