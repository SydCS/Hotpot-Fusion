from transformers import CLIPProcessor, CLIPModel
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from openai import OpenAI
import argparse

# import t2v_metrics

client = OpenAI(
    api_key="**-*", base_url="https://api.deepseek.com"
)

# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False,
# )

# print(response.choices[0].message.content)


def get_stable_diffusion_pipeline():
    # pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    # pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
    pipe = DiffusionPipeline.from_pretrained(
        "dataroot/models/stabilityai/stable-diffusion-xl-base-1.0"
    )
    pipe.to("cuda")
    return pipe


def get_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor


# def expand_prompt(initial_prompt):
#     # 使用语言模型来生成变体，这里用简单的替换示例
#     prompts = [
#         initial_prompt + " in the style of a summer day",
#         initial_prompt + " with a futuristic theme",
#         initial_prompt + " as seen in a dream",
#         initial_prompt + " in a surreal artistic style",
#         initial_prompt + " with a focus on vivid colors",
#         initial_prompt + " under the moonlight",
#     ]
#     return prompts


def expand_prompt_with_llm(initial_prompt, client):
    # 使用语言模型来生成变体
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a creative assistant"},
            {
                "role": "user",
                "content": f"You are an expert prompt optimizer for text-to-image models. Text-to-image models take a text prompt as input and generate images depicting the prompt as output. \
                You translate prompts written by humans into better prompts for the text-to-image models. Your answers should be concise and effective. \
                Here is the prompt to be optimized: {initial_prompt}. Please generate *8* variations of the prompt, rich in expressiveness while keeping the general semantic meaning. \
                For example: \
                initial_prompt: 'The unmasked wrestler hits the masked wrestler.' \
                variations 1: 'Photo of a wrestling ring where an unmasked male wrestler with a muscular physique is in the midst of delivering a powerful blow to a masked male wrestler donning a lucha libre style mask. The spectators in the background are on the edge of their seats, watching the action closely.'\
                variations 2: '...' \
                And 6 more variations. \
                Requirements: Only return the pure content of *8* generated variations. No additional sequence number or quotation mark is needed. Each prompt should be separated by a newline.",
            },
        ],
        stream=False,
    )
    prompts = response.choices[0].message.content.split(
        "\n"
    )  # 假设每个新prompt是通过换行符分隔
    for i, prompt in enumerate(prompts):
        print("Expanded prompts " + str(i) + ": " + prompt)
    prompts = [initial_prompt] + prompts
    return prompts


def generate_images(prompts, sd_pipeline):
    images = []
    for prompt in prompts:
        images.append(sd_pipeline(prompt).images[0])
    return images


def calculate_clip_scores(prompt, images, clip_model, clip_processor):
    # scores = []
    # for image in images:
    #     inputs = clip_processor(
    #         text=[prompt] * len(images),
    #         images=image,
    #         return_tensors="pt",
    #         padding=True,
    #     )
    #     outputs = clip_model(**inputs)
    #     logits_per_image = outputs.logits_per_image
    #     scores.append(logits_per_image.diag().cpu().tolist())
    # return scores

    # TODO detect each objects individually

    inputs = clip_processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
        padding=True,
    )

    outputs = clip_model(**inputs)
    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score

    return logits_per_image.detach().numpy().flatten().tolist()


# def calculate_clip_scores(prompt, images):
#     clip_flant5_score = t2v_metrics.VQAScore(
#         model="clip-flant5-xxl"
#     )  # our recommended scoring model
#     scores = []
#     for image in images:
#         scores.append(clip_flant5_score(prompt, image))
#     return scores


def analyze_and_create_new_prompt_with_llm(feedback, client):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are an expert prompt optimizer for text-to-image models. Text-to-image models take a text prompt as input and generate images depicting the prompt as output.",
            },
            {
                "role": "user",
                "content": f"Here is the initial prompt: {initial_prompt}. Below are two lists of some previous attempts. One with good variations and the other with bad variations. \
                    Each prompt is paired with a score indicating its presence in the generated image. The prompts are arranged in ascending order based on their scores, which range from 0 to 100. Higher scores indicate higher likelihood of presence. \
                    {feedback} \
                    Generate one optimized version of the initial prompt which keep the semantic meaning and that have higher scores than all the prompts above. Prioritize optimizing for object with lowest scores. Favor substitutions and reorderings over additions. \
                    Respond with the content of new prompt only.",
            },
        ],
        stream=False,
    )

    new_prompt = response.choices[0].message.content.strip()
    print("Synthesized prompt: " + new_prompt)
    return new_prompt


def create_final_image(new_prompt, sd_pipeline):
    return sd_pipeline(new_prompt).images[0]


def main(initial_prompt, num_iterations=3, num_instances=3):
    sd_pipeline = get_stable_diffusion_pipeline()
    clip_model, clip_processor = get_clip_model()

    prompts_scores = {}  # 字典用于存储prompts及其分数

    # 初始prompt扩展和生成
    initial_expanded_prompts = expand_prompt_with_llm(initial_prompt, client)
    initial_images = generate_images(initial_expanded_prompts, sd_pipeline)
    initial_scores = calculate_clip_scores(
        initial_prompt, initial_images, clip_model, clip_processor
    )
    for prompt, score in zip(initial_expanded_prompts, initial_scores):
        prompts_scores[prompt] = score

    for i in range(num_iterations):
        # 排序并选择优化
        sorted_prompts = sorted(
            prompts_scores.items(), key=lambda x: x[1], reverse=True
        )

        # 构建反馈并生成新prompt
        good_prompts = sorted_prompts[:num_instances]
        bad_prompts = sorted_prompts[-num_instances:]

        good_feedback = ", ".join(
            [
                f"{idx + 1}. '{prompt[0]}' with score {prompt[1]}"
                for idx, prompt in enumerate(good_prompts)
            ]
        )
        bad_feedback = ", ".join(
            [
                f"{idx + 1}. '{prompt[0]}' with score {prompt[1]}"
                for idx, prompt in enumerate(bad_prompts)
            ]
        )

        feedback = f"Good prompts: {good_feedback};\nBad prompts: {bad_feedback}"
        print(feedback)

        new_prompt = analyze_and_create_new_prompt_with_llm(feedback, client)

        # 生成新prompt的图像并计算分数
        new_image = sd_pipeline(new_prompt).images[0]
        new_score = calculate_clip_scores(
            initial_prompt, new_image, clip_model, clip_processor
        )

        # 添加新prompt及其分数到字典
        prompts_scores[new_prompt] = new_score[0]
        print(
            f"In iteration {i}, prompt pool size {len(prompts_scores)}",
        )

    # prompts = expand_prompt_with_llm(initial_prompt, client)

    # images = generate_images(prompts, sd_pipeline)

    # scores = calculate_clip_scores(initial_prompt, images, clip_model, clip_processor)

    # new_prompt = analyze_and_create_new_prompt_with_llm(scores, prompts, client)

    # 取出prompts_scores中score最高的prompt
    final_prompt, _ = max(prompts_scores.items(), key=lambda item: item[1])
    print(final_prompt)
    final_image = create_final_image(final_prompt, sd_pipeline)

    return final_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--n_iter", type=int, default=3)
    parser.add_argument("--n_inst", type=int, default=3)

    args = parser.parse_args()
    initial_prompt = args.prompt
    # initial_prompt = "A realistic photo of a steaming hot pot filled with a roast duck, some noodles, some beef balls and some sea-buckthorn."
    # initial_prompt = "There are fewer forks than spoons."

    num_iterations = args.n_iter
    num_instances = args.n_inst

    final_image = main(initial_prompt)
    final_image.save("{initial_prompt}.png")
