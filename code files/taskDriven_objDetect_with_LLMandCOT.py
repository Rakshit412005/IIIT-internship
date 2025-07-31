from picamera2 import Picamera2
import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite
from llama_cpp import Llama
import json
import re
import ast
from datetime import datetime

#  Import CoT reasoning function
from COT import get_affordance_reasoning

# === Load COCO label descriptions ===
with open("/home/pi/intern_llma/obj_detec/models/coco_labels_dict.json", "r") as f:
    label_desc_map = json.load(f)
labels = list(label_desc_map.keys())

# === Load LLM ===
start_load = time.time()
llm = Llama(
    model_path="/home/pi/intern_llma/obj_detec/llama.cpp/model-Qwen/Qwen_Qwen3-1.7B-Q4_K_M.gguf",
    n_ctx=512,
    n_threads=4,
    n_gpu_layers=0
)
end_load = time.time()

# === Load TFLite Object Detection Model ===
interpreter = tflite.Interpreter(model_path="/home/pi/intern_llma/obj_detec/models/ssd-mobilenet-v1-tflite-default-v1.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# === Start Camera ===
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

print(" Live feed started. Press 's' to analyze the scene for 'open a parcel'. Press 'q' to quit.")

while True:
    frame = picam2.capture_array()

    # === Object Detection ===
    resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(resized, axis=0)

    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        input_data = input_data.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

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
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Live Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if not detected_objects:
            print(" No objects detected with high confidence.")
            continue

        print(" Detected objects:", detected_objects)

        # === Save image with all detected objects ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"all_detected_objects_{timestamp}.png"
        cv2.imwrite(image_filename, frame)
        print(f" Saved image with all detected objects as '{image_filename}'")

        # === Hardcoded Task ===
        task_name = "open parcel"

        # === Generate prompt and run LLM ===
        prompt = get_affordance_reasoning(task_name, detected_objects)
        print("\n Prompt:\n", prompt)

        prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))

        output_text = ""
        first_token_time = None
        llm_start = time.time()

        stream = llm.create_completion(
            prompt=prompt,
            max_tokens=128,
            stream=True
        )

        for chunk in stream:
            if not first_token_time:
                first_token_time = time.time()
            delta = chunk.get("choices", [{}])[0].get("delta", "")
            output_text += delta

        llm_end = time.time()
        gen_tokens = len(llm.tokenize(output_text.encode("utf-8")))
        first_token_delay = (first_token_time - llm_start) if first_token_time else 0.0

        print("\n📄 Generated Answer:\n", output_text)

        print("\n LLM PERFORMANCE METRICS:")
        print(f"• Load Duration:          {(end_load - start_load):.2f} sec")
        print(f"• Total Duration:         {(llm_end - llm_start):.2f} sec")
        print(f"• Time to First Token:    {first_token_delay:.2f} sec")
        print(f"• Prompt Tokens:          {prompt_tokens}")
        print(f"• Generated Tokens:       {gen_tokens}")
        print(f"• Prompt Eval Rate:       {prompt_tokens / (llm_end - llm_start):.2f} tokens/sec")
        print(f"• Generation Eval Rate:   {gen_tokens / (llm_end - llm_start):.2f} tokens/sec")

    elif key == ord('q'):
        print(" Exiting...")
        break

cv2.destroyAllWindows()