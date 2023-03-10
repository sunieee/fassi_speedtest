import torch
# https://download.pytorch.org/whl/cu110/torch-1.7.1%2Bcu110-cp39-cp39-linux_x86_64.whl
# https://github.com/openai/CLIP
# pip install git+https://github.com/openai/CLIP.git
import clip
import codecs
import csv
import pickle
from tqdm import *
from time import time
import random
import pandas as pd
import torch.nn.functional as F
import faiss
import faiss.contrib.torch_utils

co = faiss.StandardGpuResources()
# co = faiss.GpuMultipleClonerOptions()
# co.shard = True
# co.useFloat16 = True


available_prompts = [
    'Maple leaf,autumn,brook,{{{dynamic light}}},spring,a girl,white dress,white long hair,small chest,cat ears,half body,full moon,masterpiece,best quality,official art,extremely detailed CG unity 8k wallpaper',
    'Japanese temple,dusk,dynamic light,a girl,perspective,back,kimono,highlight,bird house,best quality,masterpiece,official art,extremely detailed CG unity 8k wallpaper',
    '((illustration)), ((floating hair)), ((chromatic aberration)), ((caustic)), lens flare, dynamic angle, ((portrait)), (1 girl), ((solo)), cute face, ((hidden hands)), asymmetrical bangs, (beautiful detailed eyes), eye shadow, ((huge clocks)), ((glass strips)), (floating glass fragments), ((colorful refraction)), (beautiful detailed sky), ((dark intense shadows)), ((cinematic lighting)), ((overexposure)), (expressionless), blank stare, big top sleeves, ((frills)), hair_ornament, ribbons, bowties, buttons, (((small breast))), pleated skirt, ((sharp focus)), ((masterpiece)), (((best quality))), ((extremely detailed)), colorful, hdr',
    'library,Columnarbookshelf,limestone cave,best quality,in the desert ,cactus,Architectural relics,wilderness,a sunny day,official art,extremely detailed CG unity 8k wallpaper,masterpiece',
    'desert,karst landform,dry,rock,Huge mechanical parts,Temple Relics,Black stele,Desert Ruins,best quality,official art,extremely detailed CG unity 8k wallpaper,Genshin Impact,beautiful detailed sky,on a desert+nsfw,{{{{{{{Oasis}}}}}}}',
    '{{{{{{Indoor scientific experiment center}}}}}},science fiction,factory,{{{display device}}},Structure of large blocks,Digital painting,pipe,best quality,official art,extremely detailed CG unity 8k wallpaper,Genshin Impact,absurdres,huge filesize',
    'Cattle, animal ear, medium chest, kimono, lakeside, summer, green, forest, {{{birds}}}}, blue sky, white clouds, dynamic light, sunlight, highlight, masterpiece, a girl, bloom',
    'Girl,photo,dynamic light,sunshine,highlight,masterpiece,sunset,exquisite girl,ASK,bloom,illustration,white hair,red eyes,cherry tree,ponytail,straw hat,dress,Genshin Impact,seashore,shell,distant view,mountain,exquisite golden beach,coconut tree with good texture,daytime,white clouds,thick coating,transparent,blue sea,seabirds,flowers,rocks,starfish,fruits,Seagull,butterfly,forest,crab,dolphin,lighthouse,reef,wild flower,banana,fill light,reflective,,best quality,official art,extremely detailed CG unity 8k wallpaper,',
    'Girl,medium chest,masterpiece,brown hair,high horsetail,school uniform,village,green long scarf,summer,sunshine,dynamic light,highlight,bloom,,best quality,official art,extremely detailed CG unity 8k wallpaper,',
    'A girl,standing,armor,shock wave,mountain slope,dragon,rain,thunder,tears,dynamic light,highlight,masterpiece,particle special effect,sunlight,best quality,official art,extremely detailed CG unity 8k wallpaper, + nsfw, lowres, bad anatomy, bad hands, text error, missing fingers, extra digits, fewer digits, cropped, worst quality, low quality, standard quality, jpeg artifacts, signature, watermark, username, blurry',
    'At noon,Sallulu,yellow light,reflection,sunlight,highlight,particle effect,edge light,light transmission,softness and transition,illustration,masterpiece,high quality,lady,girl,temptation,blue eyes,long hair,white hair,white necklace,white lace underwear,white stockings,bedroom,indoor light,pillow,rr bed,medium chest,liquid,curtain,vase,desk lamp,yellow plush toys,desk,Luxury,,official art,best quality,extremely detailed CG unity 8k wallpaper,',
    'medium chest,masterpiece,dynamic light,highlight,bloom,white hair,pink lipstick,pink powder blusher,long eyelashes,{{{{eye shadow}}}},sweater,off shoulder,short skirt,pink stockings,black hair band,{{best quality}},official art,{{extremely detailed CG unity 8k wallpaper}},standing,{{{mole under eye}}},{hand on own chest},in the market,{{{dyed bangs}}},{{{{{slim}}}}},{{{one adorable girl}}},{{hair over shoulder}},{{{{{cute face}}}}},{{alternate hair length}},{{floating hair}}, {{chromatic aberration}},{{caustic}}, lens flare, dynamic angle, {{portrait}},{{cinematic lighting}}, {beautiful detailed eyes},{{{{playgrr}}}},colorful, hdr, {{extremely detailed}},',
    '((illustration)), ((floating hair)), ((chromatic aberration)), ((caustic)), lens flare, dynamic angle, ((portrait)), (1 girl), ((solo)), cute face, ((hidden hands)), asymmetrical bangs, (beautiful detailed eyes), eye shadow, ((huge clocks)), ((glass strips)), (floating glass fragments), ((colorful refraction)), (beautiful detailed sky), ((dark intense shadows)), ((cinematic lighting)), ((overexposure)), (expressionless), blank stare, big top sleeves, ((frills)), hair_ornament, ribbons, bowties, buttons, (((small breast))), pleated skirt, ((sharp focus)), ((masterpiece)), (((best quality))), ((extremely detailed)), colorful, hdr',
    'ramatic angle,(fluttered detailed ink splashs),(illustration),(((1 girl))),(long hair),(rain:0.6),((expressionless)),((carmine hair ornament:1.4)),(there is a palace far away from the girl),chinese clothes,((focus on the girl)),color ink wash painting,(ink splashing),(huaqing splashing),((colorful)),[sketch],masterpiece,best quality,beautifully painted,highly detailed,(denoising:0.7),[splash ink],yin yang',
    'over the sea,one girl,white pupils,high ponytail,wedding dress,necklace,official art',
    '{{{masterpiece}}},{{{best quality}}},{{ultra-detailed}},{illustration},{{an extremely delicate and beautiful}},dynamic angle,floating,{{detailed light 1girl}},loli,small_breasts,floating_hair,pointy_ears,halter dress,feather,leaves,river,{forest},{painting},{sketch},{bloom},depth of field,rough sketch,gothic,halloween,faceless male,crazy,dark persona,wings,straw hat,monster,ghost,star (symbol,knife,sword',
    '1 little girl,long white hair,center frills dress,beautiful face,sitting arr a desk,in winter,snow,female,nose,pale skin,twin braids,idol,asian,eyelashes,freckles,wavy hair,long hair,brown hair,eyebrows,sweater jacket,green eyes,light blush,hoodie,high-waist skirt',
    'dynamic light,crazy mind,diamond,{joker},kill,hell,star,{black hole},giant machine',
    '{{{masterpiece}}},{{the best quality}},super fine illustrations,beautiful and delicate water,Depth of field,fine 8KCG wallpapers,{extremely delicate and beautiful},{clear face},{delicate light},{{cinematic lighting}},detailed face,{portrait},Portrait lens,{{{Alphonse Mucha}}},{{Fantasy style}},{{shine}},{{{Tarot card}}},{lace},{ribbon},{detailed clothes},{multicolored Butterfly},{neon palette},{{{detailed flowers fill the screen}}},{{{{{{{Fill the screen 1 girl}}}}}}},long golden hair,pose,sage,druid,{{{{decorated green cloak}}}},decorated green hood,{{{detailed leaves fill the screen}}},no bra,',
    '{{{masterpiece}}},best quality,illustration,beautiful detailed glow,clear edges,detailed ice,red moon,{magic circle:1},{beautiful detailed eyes},expressionless,beautiful detailed white gloves,own hands clasped,{floating palaces:1.1},azure hair,long bangs,hairs between eyes,dark dress,{dark magician girl:1.1},white bowties,{{{half closed eyes}}},,big forhead,blank stare,flower,large top sleeves,{start sky:1.2},{{{solo}}},only one girl in the picture,nontraditional miko,gothic lolita,silhouette,yokozuwari,in a meadow,in the cyberpunk city,1girl,happy,smile,open mouth,cute face,long hair,white hair,lingerie,socks removed,hair ornament,hair flower',
    '{{illustration}},{{floating hair}},{{chromatic aberration}},{{caustic}},lens flare,dynamic angle,{1 girl},{{solo}},cute face,{{hidden hands}},asymmetrical bangs,{beautiful detailed eyes},eye shadow,{{huge clocks}},{{glass strips}},{floating glass fragments},{{colorful refraction}},{{dark intense shadows}},{{cinematic lighting}},{{overexposure}},{expressionless},blank stare,big top sleeves,{{frills}},hair_ornament,ribbons,bowties,buttons,{{{small breast}}},pleated skirt,{{sharp focus}},{{masterpiece}},{{{best quality}}},{{extremely detailed}},colorful,hdr,portrait,game cg,backgrr characters,contemporary,yokozuwari,motion blur,character:moriya suwako,character:kagamine len,tanlines,doggystyle,retro artstyle,bubble,demon girl,smile,longeyelashes,crying,wide eyed,brown hair,eyebrows visible through hair,silver hair,grey hair,earrings,glasses,bracelet,rabbit tail,hand fan,gem,hair ear,sunglasses,black nails,half gloves,own hands together,back-to-back,fisting,bunny,in winter,in the cyberpunk city,nature,snow,floral backgrr,black eyes,pointy ears,otoko no ko',
    'golden dragon,night,full of star,full moon',
    '((master piece)),((best quality)),illustration,colorful,(ultra-detailed),beautiful detailed eyes,beautiful detailed hair,exquisite  eyes,((an extremely delicate and beautiful)),(dynamic angle),hair spread out,ruffling hair,hair floral print,hair half undone,1girl,dragon girl,a big dragon stands behind the girl,green eyes,mechanical equipment,mechanical armor,exposing the belly,broken clothes,small breasts,(cold face),rain day,thunder,arknights,exquisite and beautiful silver dragon armor,(( white hair)),watercolor,(aestheticism painting),filigree,exquisite and beautiful dragon,grey day',
    'bug,robot,character design,strong rim light,surrred,wings,blood on face,forehead mark,',
    'a young goddess,reclining,in the air,{{holding a beautiful bird}},long dress,queen,ancient,crown,snow in the backgrr,sitting on a huge bird,{{game cg}},{{retro artstyle}},fire,white hair,,jewelry,surrred by birds',
    'christmas,fire,brown eyes,short hair,black hair,multiple girls,indian style,retro artstyle,snowing,starry sky,dark,red backgrr,',
    'Columnarbookshelf,limestone cave,best quality,in the desert ,cactus,Architectural relics,wilderness,a sunny day,official art,extremely detailed CG unity 8k wallpaper,masterpiece',
    'girl, sheep ear, sheep body,glowing,watercolor, universe, cute, 8k, technology, star',
    'muscular male,one boy,dark skin,wings,smile,disheveled hair,wind',
]


rwidth = 4

def rr(x):
    return round(x, rwidth)

method_dic = {
    'default': '',
    'Flat': 'Flat',
    'IVFxFlat': 'IVF100,Flat',
    'PQx': 'PQ16',
    'IVFxPQy': 'IVF100,PQ16',
    'LSH': 'LSH',
}

def replace(tags, model, feature_matrix, thr=0.5, index=None):
    t = time()
    tokenized_input = torch.cat([clip.tokenize(tag) for tag in tags]).to(device)
    input_features = model.encode_text(tokenized_input)
    embed_time = rr(time() - t)

    # input_features = torch.cat(input_features, dim=0)
    # print(input_features.shape)
    # input_features /= input_features.norm(2,1)
    input_features = F.normalize(input_features, p=2, dim=1)
    # print(input_features.shape)
    # print(feature_matrix.shape)
    # print(input_features.norm(2, 1))
    # print(feature_matrix.norm(2, 0))
    t = time()
    if index:
        input_features = input_features.to(torch.float32)
        try:
            D, idxs = index.search(input_features, 1)
        except:
            D, idxs = index.search(input_features.cpu(), 1)
        
        return idxs, [D[i]>0 and 0 <=idxs[i]<feature_matrix.shape[1] for i in range(len(tags))], t, embed_time
    else:
        similarities = input_features @ feature_matrix
        idxs = similarities.argmax(dim=1)
        return idxs, [similarities[i, idx]>thr for i, idx in enumerate(idxs)], t, embed_time

def get_random_prompt():
    ret = random.choice(available_prompts)
    ret = {'prompt': ret}

# load model
cuda_index = 1
device = f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
# model, preprocess = clip.load("ViT-B/32", device=device)
# load tags
tag_list = []
with codecs.open('./tags_count.csv') as f:
    for row in csv.DictReader(f, skipinitialspace=True):
        tag_list.append(row['solo'])
# load tag_features
f = open('tag_clip_features_norm.pkl', 'rb')
normed_tag_features = pickle.load(f).to(device)
f.close()

dim = normed_tag_features.shape[0]
print(type(normed_tag_features))
# xb = normed_tag_features.cpu().numpy().transpose()
xb = normed_tag_features.transpose(0, 1)
# not contiguous: https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch
xb = xb.contiguous()
# assert x.dtype == torch.float32
xb = torch.tensor(xb, dtype=torch.float32)
print(xb.dtype)

statistic = pd.DataFrame(columns=['inference time', 'train time', 'add time', 'search time', 'embed time',
                                'accuracy', '#modify', '#right'])
standard_tag = {} 

def main(method, **params):
    name = method
    if method == 'HNSW':
        name = f"{method}{params['M']}-{params['efSearch']}-{params['efConstruction']}"
    print('='*20, name, '='*20)

    if method != 'default':
        if method == 'HNSW':
            index = faiss.IndexHNSWFlat(dim, params['M'])
            index.hnsw.efConstruction = params['efConstruction']
            index.hnsw.efSearch = params['efSearch']
        else:
            index = faiss.index_factory(dim, method_dic[method], faiss.METRIC_L2)
            index = faiss.index_cpu_to_gpu(co, cuda_index, index)
        # index = faiss.index_cpu_to_all_gpus(index, co)
        t = time()
        if not index.is_trained:
            index.train(xb.cpu())
        trian_time = rr(time() - t)

        # https://github.com/facebookresearch/faiss/issues/2074
        # ??????gpu????????????bug??????????????????gpu

        t = time()
        index.add(xb.cpu())
        add_time = rr(time() - t)
    else:
        index = None
        trian_time = 0
        add_time = 0

    df = pd.DataFrame(columns=['time', 'count', 'length', 'before', 'after'])
    predict_modify = 0
    predict_right = 0
    search_time = 0
    inference_time = 0
    embed_time = 0

    for prompt in available_prompts:
        # prompt = 'Cattle, animal ear, medium chest, kimono, lakeside, summer, green, forest, {{{birds}}}, blue sky, white clouds, dynamic light, sunlight, highlight, masterpiece, a girl, bloom'
        new_tags = []
        prompt_list = prompt.split(',')
        for idx, tag in enumerate(prompt_list):
            tag = tag.strip('(').strip(')').strip('[').strip(']').strip('{').strip('}').split(':')[0].strip()
            new_tags.append(tag)
        feature_matrix = normed_tag_features
        replaced_prompt = []
        
        t = time()        
        idxs, modify, st, et = replace(new_tags, model, feature_matrix, index=index)
        embed_time += et
        search_time += rr(time() - st)
        inference_time += rr(time() - t)
        for i in range(len(new_tags)):
            if modify[i]:
                replaced_prompt.append(prompt_list[i].replace(new_tags[i], tag_list[idxs[i]]))
                predict_modify += 1

            # ??????acc
            predict_tag = tag_list[idxs[i]] if modify[i] else None
            if method == 'default':
                standard_tag[new_tags[i]] = predict_tag
            
            predict_right += standard_tag[new_tags[i]] == predict_tag

        new_prompt = ','.join(replaced_prompt)
        df.loc[len(df)] = {
            'time': rr(time() - t),
            'count': len(prompt_list),
            'length': len(prompt),
            'before': prompt,
            'after': new_prompt
        }
    
    df.to_csv(f'output/{name}.csv')
    statistic.loc[name] = {
        'accuracy': rr(predict_right / 703),
        'inference time': rr(inference_time), 
        'search time': rr(search_time),
        'embed time': rr(embed_time),
        'train time': trian_time,
        'add time': add_time,
        '#modify': predict_modify,
        '#right': predict_right,
    }
    for k, v in statistic.loc[name].items():
        print(k + ':', v)

if __name__ == '__main__':
    try:
        for method in method_dic:
            main(method)
        # params = {
        #     'M': 32,
        #     'efSearch': 100,  # number of entry points (neighbors) we use on each layer
        #     'efConstruction': 100, # number of entry points used on each layer during construction
        # }
        # for M in [16, 32, 64]:
        #     for efSearch in [50, 100]:
        #         for efConstruction in [50, 100]:
        #             main('HNSW', M=M, efSearch=efSearch, efConstruction = efConstruction)
        
        for M in [16, 32, 64]:
            main('HNSW', M=M, efSearch=100, efConstruction = 100)

    finally:
        statistic.to_csv('statistic.csv')

