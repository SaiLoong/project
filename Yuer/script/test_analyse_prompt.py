# -*- coding: utf-8 -*-
# @file test_analyse_prompt.py
# @author zhangshilong
# @date 2024/9/6

from collections import Counter

# AWPortrait v1.4的示例prompt。https://civitai.com/posts/4180791
prompt_stacking = """

upper body portrait of 1girl wear a black color turtleneck sweater,A proud and confident expression,long hair,look at viewers,studio fashion portrait,studio light,pure white background,

upper body portrait of 1girl wear (red color turtleneck sweater:1),A proud and confident smile expression,long hair,look at viewers,studio fashion portrait,studio light,pure white background,

upper body portrait of 1girl wear suit with tie,A proud and confident smile expression,long hair,look at viewers,studio fashion portrait,studio light,pure white background,

A woman striking a poised stance,with a traditional East Asian influence. She's wearing a floral-patterned qipao in rich red and green hues,complementing her classic hairstyle,set against a plain backdrop with delicate flowers.,

upper body portrait of 1girl,sling dress,standing,wearing a floral-patterned qipao in rich red and green hues,(outdoors:1)  look at view,

A black and white portrait of a young woman with a captivating gaze. She's bundled up in a cozy black sweater,hands gently cupped near her face. The monochromatic tones highlight her delicate features and the contemplative mood of the image.,

Fashion photography portrait,((upper body:1)),(1girl is surrounded by Peacock feather from head to neck , ((long hair:1)), (wear Peacock feather outfit fashion with ruffled layers:1)

a 10-year-old little girl,happy,bokeh,motion blur,facula,documentary photography,in winter,major snow,(snow floating in the air:1.2),high_heels,white pantyhose,white dress,feature article,face close-up,(masterpiece:1.2, best quality)

Asian female relax,Thin cardigan sweater,Capris,sports shoes,OOTD,Outfit of the Day,full_body,solo,simple_background,light green background,

upper body portrait of 1girl,sling dress,standing,(Best quality:1.2),(masterpiece:1.2),(ultra high res:1.2),8k,Photography,super detailed,Depth of field,Bright color,Korean style photography,fresh and natural,girlish,bright colors,exquisite makeup,elegant posture,fashionable elements,light and shadow interweaving,atmosphere creation,detail processing,visual appeal,

A woman dressed in a traditional East Asian outfit poses against an earth-toned background. Her elegant attire and the intricate hair accessories set against her long,flowing hair evoke a sense of historical beauty and poise.,

upper body portrait of asian female 80 years old,Wearing luxurious clothing,standing in front of a simple background,studio,

upper body of asian female 60 years old,Wearing luxurious clothing,standing in front of a simple background,studio,

realistic 1girl,solo,looking at viewer,realistic,upper body,long hair,earrings,blurry background,brown hair,blurry,jewelry,plant,bangs,green sweater,smile,shirt,lips,leaf,signature,black eyes,closed mouth,black hair,

A pensive schoolgirl with a bow in her hair and wearing a dark blazer over a white shirt with a red ribbon tie looks away thoughtfully. The soft daylight accentuates her youthful innocence.,eyes look down,

a 16-year-old girl,n,bokeh,motion blur,(facula),feature article,face close-up,teeth,smile,an adoring expression,be shy,blush,model,female model,beautiful face,upper_body,chignon,clothes of diamonds,stage,diamonds floating in the air,(masterpiece:1.2, best quality),(a hand waving to the audience:1),

A close-up portrait of a young woman with a delicate bow in her hair. Her features are highlighted by subtle makeup,accentuating her youthful appearance and innocent expression. The background is simple,focusing attention on her detailed features and the reflective quality in her eyes.,

"""

prompts = [prompt for prompt in prompt_stacking.split("\n") if prompt]
print(prompts)

features = list()
for prompt in prompts:
    for feature in prompt.split(","):
        if feature == "":
            continue

        feature = feature.strip().replace("(", "").replace(")", "").lower()
        if ":" in feature:
            feature = feature.rsplit(":", 1)[0]

        features.append(feature)
# print(features)

counter = Counter(features)
