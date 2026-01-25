from qwen_vl_utils import process_vision_info

KOREAN_MODEL_REPO_MAPPING = {
    'other_condemning': 'kimyeonz/Multimodal_Moral_Emotion_Classifier_KOR_Other_Condemning',
    'other_praising': 'kimyeonz/Multimodal_Moral_Emotion_Classifier_KOR_Other_Praising',
    'other_suffering': 'kimyeonz/Multimodal_Moral_Emotion_Classifier_KOR_Other_Suffering',
    'self_conscious': 'kimyeonz/Multimodal_Moral_Emotion_Classifier_KOR_Self_Conscious',
    'non_moral_emotion': 'kimyeonz/Multimodal_Moral_Emotion_Classifier_KOR_Non_Moral_Emotion',
    'neutral': 'kimyeonz/Multimodal_Moral_Emotion_Classifier_KOR_Neutral'
}

ENGLISH_MODEL_REPO_MAPPING = {
    'other_condemning': 'kimyeonz/Multimodal_Moral_Emotion_Classifier_ENG_Other_Condemning',
    'other_praising': 'kimyeonz/Multimodal_Moral_Emotion_Classifier_ENG_Other_Praising',
    'other_suffering': 'kimyeonz/Multimodal_Moral_Emotion_Classifier_ENG_Other_Suffering',
    'self_conscious': 'kimyeonz/Multimodal_Moral_Emotion_Classifier_ENG_Self_Conscious',
    'non_moral_emotion': 'kimyeonz/Multimodal_Moral_Emotion_Classifier_ENG_Non_Moral_Emotion',
    'neutral': 'kimyeonz/Multimodal_Moral_Emotion_Classifier_ENG_Neutral'
}

KOREAN_MORAL_EMOTION_THRESHOLDDS = {
    'other_condemning': 0.75,
    'other_praising': 0.22,
    'other_suffering': 0.56,
    'self_conscious': 0.26,
    'non_moral_emotion': 0.19,
    'neutral': 0.55
}

ENGLISH_MORAL_EMOTION_THRESHOLDDS = {
    'other_condemning': 0.63,
    'other_praising': 0.54,
    'other_suffering': 0.52,
    'self_conscious': 0.17,
    'non_moral_emotion': 0.24,
    'neutral': 0.59
}

MORAL_EMOTION_MAPPING = {
    'other_condemning': 'Other-condemning',
    'other_praising': 'Other-praising',
    'other_suffering': 'Other-suffering',
    'self_conscious': 'Self-conscious',
    'non_moral_emotion': 'Non-moral emotion',
    'neutral': 'Neutral'
}

KOREAN_EMOTION_DESCRIPTION = {
    'Other-condemning': '분노, 경멸, 혐오 등과 같이 타인을 비난하는 감정',
    'Other-praising': '감탄, 감사, 경외감 등과 같이 타인을 칭찬하는 감정',
    'Other-suffering': '연민, 동정 등과 같이 타인을 공감하는 감정',
    'Self-conscious': '수치심, 죄책감, 당혹감 등과 같이 자신을 부정적으로 평가하는 감정',
    'Non-moral emotion': '두려움, 놀라움, 기쁨, 낙관주의 등과 같이 감정이 있지만 다른 도덕감정 기준에 속하지 않는 감정',
    'Neutral': '감정이 없거나 거의 없는 중립적인 카테고리'
}

ENGLISH_EMOTION_DESCRIPTION = {
    'Other-condemning': 'Emotions that condemn others, such as anger, contempt, or disgust.',
    'Other-praising': 'Emotions that praise others, such as admiration, gratitude, or awe.',
    'Other-suffering': 'Emotions of empathy for the suffering of others, such as compassion, or sympathy.',
    'Self-conscious': 'Emotions that negatively evaluate oneself, such as shame, guilt, or embarrassment.',
    'Non-moral emotion': 'Emotions that are emotional but not one of the other emotions, such as fear, surprise, joy, optimism, etc.',
    'Neutral': 'A neutral category with no or few emotions.'
}

def get_prompt(emotion, korean=True):
    emotion = MORAL_EMOTION_MAPPING[emotion]
    if korean:
        system_prompt = f"""
            당신은 ‘도덕감정’ 분야에 대한 전문가입니다. 주어진 이미지와 텍스트에 나타나는 도덕감정이 {emotion} 인지 여부를 분류하세요. {emotion}은 다음과 같이 정의 할 수 있습니다:

            1. {emotion}: {KOREAN_EMOTION_DESCRIPTION[emotion]}

            주어진 데이터가 {emotion}의 도덕감정을 나타내는지 True 혹은 False 로 분류하고, 답을 할 때는 추가적인 설명 없이 True 혹은 False 로만 답하세요.
        """

        message_gpt = """
            주어진 Thumbnail image와 Title text에 모두 나타난 도덕감정이 {emotion}이 맞나요? 
            추가적인 설명 없이, 도덕감정이 {emotion} 인지 여부만 True 또는 False로 답변해주세요.
        """

        return system_prompt, message_gpt
    
    system_prompt = f"""
        You are an AI expert in 'moral emotions'. Classify whether the moral emotion expressed in the given image and text is {emotion}. {emotion} is defined as follows:
        
        1. {emotion}: {ENGLISH_EMOTION_DESCRIPTION[emotion]}

        Classify the given data as True or False to indicate the moral emotion of {emotion} and answer only True or False without your further explanation.
    """

    message_gpt = """
        Is the moral emotion expressed in both the given Thumbnail image and Title text {emotion}? 
        Without further explanation, answer only True or False to indicate whether the moral emotion is {emotion}.
        {Answer}
    """

    return system_prompt, message_gpt

def format_data(sample, target_emotion, korean=True):
    system, message = get_prompt(target_emotion, korean)
    return {"messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message + '\n' + sample.title,
                        },{
                            "type": "image",
                            "image": sample.thumbnail,
                        }
                    ],
                }
            ],
            "id": sample.id
        }


def collate_fn(examples, processor, device):
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]
    ids = [example['id'] for example in examples]
    
    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)    
    batch = {k: v.to(device) for k, v in batch.items()}

    batch['ids'] = ids
    
    return batch