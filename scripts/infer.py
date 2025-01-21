# vllm_model.py
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import torchvision.transforms as transforms
import os
import io
import json
from PIL import Image
import torch
tokenizer = None
'''
VLLM inferece for single img
'''
class VLMModel(object):
    def __init__(self, ) -> None:
        pass


    def load_model(self, model_path):
        # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
        self.sampling_params = SamplingParams(temperature=0.1, 
                                              top_p=0.7,
                                              top_k=50, 
                                              max_tokens=4096,#Output length 
                                              stop_token_ids=[151329, 151336, 151338])
        self.processor = AutoProcessor.from_pretrained(model_path)
        '''from factory{'model': 'Qwen/Qwen2-VL-7B-Instruct', 'trust_remote_code': True, 
        'dtype': 'auto', 'tensor_parallel_size': 8, 'disable_log_stats': True, '
        enable_lora': False, 'limit_mm_per_prompt': {'image': 4, 'video': 2}}'''
        self.model = LLM(model=model_path, 
                         limit_mm_per_prompt={"image": 1},
                         tokenizer=None, 
                         max_model_len=4096, 
                         gpu_memory_utilization=0.4,
                         trust_remote_code=True)

    def inference(self, image):
        image = Image.open(image)
        width, height = image.size
        image = image.resize((width // 2, height // 2))
        transform = transforms.Compose([
        transforms.PILToTensor()])
        transform = transforms.PILToTensor()
        # Convert the PIL image to Torch tensor
        image = transform(image)
        # buffered = io.BytesIO()
        # image.save(buffered)
        # image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        messages = [
            {"role": "system", "content": "Given a single image that represents a concatenation of six cameras mounted on a car, arranged in the following relative positions:\nTop row: Front Left, Front, Front Right\nBottom row: Rear Left, Rear, Rear Right\nThe relative positioning of the cameras in the concatenated image corresponds to their real-world placement on the car.\nYour objective is to analyze the scene depicted in the six-camera view, understand the environment surrounding the vehicle, and provide trajectory coordinates in the BEV (Bird's Eye View) space based on the current driving conditions and history trajectory provided by the user.\n"},
            {"role": "user", "content": [
                {"type": "image"},#, "image": image},  # 直接传入PIL.Image对象
                {"type": "text", "text": "<image>Can you describe the current weather conditions and the general environment depicted in the images?"},
                #{"type": "text", "text": "<image>Please describe the current driving conditions based on the input images"},
                #{"type": "text", "text": "Please provide the planning trajectory for the ego car without reasons."}#Please provide the planning trajectory for the ego car without reasons."},
            ]}
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,#make sure return answer instead of text completion
        )
        outputs = self.model.generate({"prompt": prompt,"multi_modal_data": {"image": image}}, self.sampling_params)

        # 输出是一个包含 prompt、生成文本和其他信息的 RequestOutput 对象列表。
        # 打印输出。
        #print(outputs[0].text)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print('G'*100,generated_text)
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            return generated_text
if __name__ == "__main__":
    vlm_model = VLMModel()
    vlm_model.load_model(model_path='/mnt/ve_share/hezijian/LLaMA-Factory/models/qwen2_vl_V6')#'/mnt/ve_share/hezijian/LLaMA-Factory/models/qwen2_vl_V6')#'Qwen/Qwen2-VL-7B-Instruct')
    test_img = 'saves/test.jpg'
    res = vlm_model.inference(test_img)
    test_img = 'saves/test2.jpg'
    res = vlm_model.inference(test_img)
    test_img = 'saves/test3.jpg'
    res = vlm_model.inference(test_img)
    