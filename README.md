# Moral Outrage Shapes Commitments Beyond Attention: Multimodal Moral Emotions on YouTube in Korea and the US

In Proceedings of the ACM Web Conference (WWW), 2026. To appear.

## Abstract
Understanding how media rhetoric shapes user engagement is crucial in the attention economy. This study examines how moral-emotional rhetoric used by mainstream news media channels on YouTube influences audience engagement, drawing on cross-cultural data from Korea and the United States. To capture the multimodal nature of YouTube, which combines thumbnail images and video titles, we develop a multimodal moral emotion classifier by fine-tuning a vision–language model. The model is trained on human-annotated multimodal datasets in both Korean and English and applied to approximately 400,000 videos from major news outlets in both countries. We analyze three engagement levels—views, likes, and comments—representing increasing degrees of commitment. The results show that *other-condemning* rhetoric—expressions of moral outrage that criticize others’ morality—consistently increases all three forms of engagement across cultures, with effect sizes strengthening from passive viewing to active commenting. These findings suggest that moral outrage is the most effective emotional strategy, attracting not only attention but also active participation. We discuss concerns about the potential misuse of *other-condemning* rhetoric as such practices can deepen polarization by reinforcing in-group/out-group divisions. To facilitate future research and ensure reproducibility, we publicly release our Korean and English multimodal moral emotion classification models.

## Method Overview
![Overview](https://github.com/Paul-scpark/Multimodal-Moral-Emotion/blob/main/image/overview.png)
Overview of the research framework. Multimodal YouTube data and human-labeled moral emotions are used to model and analyze how predicted moral emotions relate to user engagement (views, likes, and comments).

## Result
![Result](https://github.com/Paul-scpark/Multimodal-Moral-Emotion/blob/main/image/result.png)
Predicted engagement by *other-condemning* emotion probability. Top panels show fitted counts from the negative binomial models, and bottom panels show relative engagement (IRR) for Korea and the United States. As moral outrage intensity increases, engagement progressively rises from views to comments. Shaded areas denote 95\% confidence intervals.

## Model
- Korean
  - [Other-condemning](https://huggingface.co/kimyeonz/Multimodal_Moral_Emotion_Classifier_KOR_Other_Condemning)
  - [Other-praising](https://huggingface.co/kimyeonz/Multimodal_Moral_Emotion_Classifier_KOR_Other_Praising)
  - [Other-suffering](https://huggingface.co/kimyeonz/Multimodal_Moral_Emotion_Classifier_KOR_Other_Suffering)
  - [Self-conscious](https://huggingface.co/kimyeonz/Multimodal_Moral_Emotion_Classifier_KOR_Self_Conscious)
  - [Neutral](https://huggingface.co/kimyeonz/Multimodal_Moral_Emotion_Classifier_KOR_Neutral)
  - [Non-moral emotion](https://huggingface.co/kimyeonz/Multimodal_Moral_Emotion_Classifier_KOR_Non_Moral_Emotion)
- English
  - [Other-condemning](https://huggingface.co/kimyeonz/Multimodal_Moral_Emotion_Classifier_ENG_Other_Condemning)
  - [Other-praising](https://huggingface.co/kimyeonz/Multimodal_Moral_Emotion_Classifier_ENG_Other_Praising)
  - [Other-suffering](https://huggingface.co/kimyeonz/Multimodal_Moral_Emotion_Classifier_ENG_Other_Suffering)
  - [Self-conscious](https://huggingface.co/kimyeonz/Multimodal_Moral_Emotion_Classifier_ENG_Self_Conscious)
  - [Neutral](https://huggingface.co/kimyeonz/Multimodal_Moral_Emotion_Classifier_ENG_Neutral)
  - [Non-moral emotion](https://huggingface.co/kimyeonz/Multimodal_Moral_Emotion_Classifier_ENG_Non_Moral_Emotion)
 
## Citation
