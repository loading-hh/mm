from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from transformers import AutoProcessor, CLIPVisionModel, CLIPVisionModelWithProjection

# 这一部分是文本模型
tokenizer = AutoTokenizer.from_pretrained(r"e:\BaiduNetdiskDownload\huggingface\openai\clip-vit-base-patch32")
inputs1 = tokenizer(["a photo of a cat", "dog"], padding=True, return_tensors="pt")
print(tokenizer.model_max_length)
# 此模型比下面的CLIPTextModelWithProjection要少最后一层
# model1 = CLIPTextModel.from_pretrained(r"e:\BaiduNetdiskDownload\huggingface\openai\clip-vit-base-patch32")
# outputs1 = model1(**inputs1)
# print(model1)
# last_hidden_state = outputs1.last_hidden_state
# pooled_output = outputs1.pooler_output
# print(last_hidden_state)
# print(pooled_output)

# 此模型比上面的CLIPTextModel要多最后一层
model2 = CLIPTextModelWithProjection.from_pretrained(r"e:\BaiduNetdiskDownload\huggingface\openai\clip-vit-base-patch32")
outputs2 = model2(**inputs1)
text_embeds = outputs2.text_embeds
print(outputs2)


# 这一部分是视觉模型

processor = AutoProcessor.from_pretrained(r"e:\BaiduNetdiskDownload\huggingface\openai\clip-vit-base-patch32")
image = Image.open(r"C:\Users\CCU6\Desktop\222.png")
inputs2 = processor(images=image, return_tensors="pt")
print(inputs2["pixel_values"].shape)
# 此模型比下面的CLIPTextModelWithProjection要少最后一层
# model3 = CLIPVisionModel.from_pretrained(r"e:\BaiduNetdiskDownload\huggingface\openai\clip-vit-base-patch32")
# output3 = model3(**inputs2)
# last_hidden_state = output3.last_hidden_state
# pooled_output = output3.pooler_output  # pooled CLS states
# print(last_hidden_state)
# print(pooled_output)

# 此模型比上面的CLIPTextModel要多最后一层
model4 = CLIPVisionModelWithProjection.from_pretrained(r"e:\BaiduNetdiskDownload\huggingface\openai\clip-vit-base-patch32")
outputs4 = model4(**inputs2)
image_embeds = outputs4.image_embeds
print(image_embeds)