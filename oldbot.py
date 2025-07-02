import asyncio
import base64
import copy
import json
import os
import random
import re
import time
import traceback
from datetime import datetime as Date
from io import BytesIO
from pathlib import Path
from queue import Queue, PriorityQueue
from re import RegexFlag
from textwrap import wrap
from threading import Thread
from time import sleep

import discord.permissions
import requests
from PIL import Image
from civitai_api import api as capi
from civitai_downloader import download_file
from civitai_api.models import ModelVersion, BaseModel
from civitai_api import Civitai
from discord import ApplicationContext, RawReactionActionEvent
from discord import Colour
from discord import Embed
from discord.ext import commands
from discord.ext.commands.context import Context
from discord.file import File
from discord.message import PartialMessage

api_url = "http://127.0.0.1:7860/"


def sdapi(api): return api_url + 'sdapi/v1/' + api


outputDir = 'output/'
sdDir: Path | None
loraDir: Path | None
wildcardDir: Path | None

if os.path.exists('config.json'):
    with open('config.json', 'r') as cfg:
        props = json.load(cfg)
        channelId = props['channelId']
        mToken = props['token']
        server = discord.Object(id=props['serverId'])
        civToken = props['civitai']
        civclient = Civitai(civToken)
        cfg.close()
else:
    cnf = {
        'token': 'discord bot token',
        'channelId': 0,
        'serverId': 0,
        'civitai': 'civitai account token'
    }
    with open('config.json', 'w') as cfg:
        json.dump(cnf, cfg, indent=2)
        cfg.close()
    print('Generated a blank config file. Fill it out and restart the bot.')
    exit(0)

placeholderRemover = r'(\s?{}\s?)'

wigwam = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=wigwam)
genNum = 0
q: Queue[tuple] = PriorityQueue()
currentGen: dict | None = None
samplers: list = []
loras: dict = {}
wildcardList: list = []
userdata = {}
payload = {
    "prompt": "{}",
    "negative_prompt": "rating_explicit, text, watermark, logo, {}",
    "width": 720,
    "height": 960
}

alwayson = {
    "alwayson_scripts": {
        "Never OOM Integrated": {
            "args": [
                #True, True
                False, False
            ]
        },
        "ADetailer": {
            "args": [
                False,
                False
            ]
        },
        "forge couple": {
            "args": [
                False,
                False,
                "Basic",
                "[SEP]",
                "Horizontal",
                "First Line",
                0.3,
                None
            ]
        },
        "ControlNet": {
            "args": [
                {
                    "enabled": False
                }
            ]
        }
    }
}

ad_models = {
    'box': "face_yolov8n.pt",
    'mask': "Anzhc Face seg 640 v2 y8n.pt",
    'hand': 'hand_yolov8n.pt'
}

cn_modes = {'balanced': 'Balanced', 'prompt': 'My prompt is more important', 'cnet': 'ControlNet is more important'}
cn_payload = {
    "control_mode": "My prompt is more important",
    "enabled": True,
    "guidance_end": 0.4,
    "guidance_start": 0,
    "hr_option": "Both",
    "image": "",
    "input_mode": "simple",
    "mask_image": None,
    "model": "CN-anytest_v4-marged_pn_dim256 [2fe07d89]",
    "module": "None",
    "pixel_perfect": True,
    "resize_mode": "Crop and Resize",
    "save_detected_map": False,
    "use_preview_as_input": False,
    "weight": 1.6
}

txtDefaults = {
    "steps": 16,
    "cfg_scale": 2.5,
    "sampler_name": "Euler a",
    'override_settings': {
        "emphasis": "No norm"
    }
}

imgDefaults = {
    'denoising_strength': 0.75,
    'resize_mode': 1,
    'include_init_images': False,
    'override_settings': {
        "img2img_fix_steps": True,
        "emphasis": "No norm"
    }
}

lightningDefaults = {
    "steps": 6,
    "cfg_scale": 2.25,
    "sampler_name": "DPM++ 2M SDE SGMUniform"
}


def connectToSD():
    if response := requests.get(api_url + 'internal/sysinfo'):
        if not response.ok: return False
        sddata = response.json()
        global sdDir, loraDir, wildcardDir
        sdDir = Path(sddata['Data path'])
        loraDir = sdDir.joinpath('models/Lora')
        wildcardDir = sdDir.joinpath('wildcards')
        if not wildcardDir.exists():
            print('Couldn\'t find the \"Dynamic Prompts\" extension. Wildcards are disabled.')
            wildcardDir = None
        return True
    return False


@bot.event
async def on_ready():
    # with open('avatar.png', 'rb') as f:
    #    with open('banner.png', 'rb') as b:
    #        await bot.user.edit(username="Transfield", avatar=f.read(), banner=b.read())
    if connectToSD():
        global payload
        payload = payload | txtDefaults
        loadUserData()
        getSamplers()
        getLoras(refresh=False)
        getWildcards()
        print(f'We have logged in as {bot.user}')


class AutismHelp(commands.MinimalHelpCommand):
    async def send_pages(self):
        destination = self.get_destination()
        for page in self.paginator.pages:
            emby = discord.Embed(description=page, colour=Colour.blurple())
            await destination.send(embed=emby)


bot.help_command = AutismHelp(command_attrs={
    'name': 'autism_help',
    'hidden': True,
    # 'paginator': discord.ext.commands.Paginator(prefix='!')
})


async def send_message_to_specific_channel(message: str | list, file: File, ref: PartialMessage | ApplicationContext):
    if type(message) is list:
        l = message.copy()
        message = ''
        for s in l:
            message += s

    if type(ref) is ApplicationContext:
        await ref.respond(content=message, file=file)
    else:
        channel = bot.get_channel(channelId)
        await channel.send(content=message, file=file, reference=ref)


def sendMessage(message: str, file: File = None, ref=None):
    asyncio.run_coroutine_threadsafe(send_message_to_specific_channel(message, file, ref), bot.loop)


'''
@bot.slash_command(name='autism', description='Advanced prompt for SD', guild=server)
@discord.permissions.default_permissions(view_channel=True)
@option('prompt', input_type=str, description='Base prompt', required=True)
@option('negs', input_type=str, description='Negative prompt (Optional)', default=None)
@option('steps', input_type=int, description='Steps (Optional)', default=6, min_value=2, max_value=24)
@option('prompt', input_type=float, description='CFG Scale (Optional)', default=2.5)
async def advPrompt(ctx: discord.ApplicationContext, prompt: str, negs: str = None, steps: int = 6,
                    cfg_scale: float = 2.5, seed: int = -1, no_defaults: bool = False) -> None:
    request = {'author': ctx.author.name, 'timestamp': timestamp()}
    prompt = prompt.strip()
    if prompt:
        request['prompt'] = prompt
    else:
        await ctx.message.reply(content='Base prompt can\'t be empty', ephemeral=True)
        return
    if negs:
        request['negative_prompt'] = negs
    if steps != 6:
        request['steps'] = steps
    if cfg_scale != 2.5:
        request['cfg_scale'] = cfg_scale
    if seed > -1:
        request['seed'] = seed
    if no_defaults:
        request['no_defaults'] = True
    request['ref'] = ctx
    q.put(request)
    await ctx.response.defer(ephemeral=False)
'''


def parseRes(mRes:int, obj: dict, x: str, width: str, height: str):
    res = x.split('x')
    if len(res) == 2:
        w = int(res[0])
        h = int(res[1])
        if w * h > mRes**2:
            return f'Can\'t generate an image larger than {str(mRes)}x{str(mRes)} due to vram constraints.'
        obj[width] = w
        obj[height] = h
    elif len(res) == 1:
        obj[width] = obj[height] = min(mRes, int(res[0]))
    else:
        return f'Invalid value passed to the `res:{x}` parameter. Example resolution: `res:1024x768`.'


def matchSettings(obj: dict, message: str, genType: str = 'txt2img'):
    if match := re.search(r'<?(http[^> \n]+)>?', message):
        url = match.group(1)
        try:
            response = requests.get(url)
            if response.ok:
                img = Image.open(BytesIO(response.content))
                if img.format.lower() in ['png', 'jpg', 'jpeg', 'webp']:
                    if img.mode in ['RGBA', 'P']: img = img.convert('RGB')
                    obj['width'] = img.width
                    obj['height'] = img.height
                    obj['init_images'] = [base64.b64encode(response.content).decode()]
                    obj['image_url'] = url
                    img.close()
                else:
                    return 'Provided image URL is not a valid format. The API only expects png, jpg and webp images.'
        except Exception as e:
            return traceback.format_exception_only(e)
        message = message[match.span()[1]:].strip()

    match = re.search(r'^\[([^]]+)]', message)
    if match:
        cfgs = match.group(1).split(';')
        cnet = {}
        try:
            for s in cfgs:
                s: str = s.strip()
                line = re.search(r'(\w+)[:=]\s?([^$]+)', s)
                if line and line.lastindex == 2:
                    k = line.group(1)
                    v = line.group(2).strip()
                    if k.startswith('neg'):
                        obj['negative_prompt'] = v
                    elif k.startswith('cfg'):
                        obj['cfg_scale'] = float(v)
                    elif k.startswith('stp'):
                        obj['steps'] = int(v)
                    elif k.startswith('sampler'):
                        obj['sampler_name'] = v
                    elif k == 'batch_size' or k == 'enable_hr':
                        raise ValueError(f'{k} is not available at this time.')
                    elif k.startswith('den'):
                        obj['denoising_strength'] = float(v)
                    elif k.startswith('scale'):
                        obj['img_scale'] = min(max(0.5, float(v)), 2)
                    elif k.startswith('aden'):
                        obj['ad_denoising_strength'] = float(v)
                    elif k.startswith('amode'):
                        obj['ad_model'] = v
                    elif k.startswith('cnstp'):
                        values = re.findall('([0-9.]+)[,: ]*', v)
                        if len(values) == 1:
                            cnet['guidance_end'] = float(values[0])
                        else:
                            cnet['guidance_start'] = float(values[0])
                            cnet['guidance_end'] = float(values[1])
                    elif k == 'cn':
                        obj['controlnet'] = True
                        cnet['weight'] = float(v)
                    elif k.startswith('cnmode'):
                        if v.isdigit() and 0 <= int(v) <= 2:
                            cnet['control_mode'] = int(v)
                        elif v in cn_modes:
                            cnet['control_mode'] = cn_modes[v]
                        else:
                            raise ValueError(f'Unknown ControlNet mode `{v}` passed in generation settings.')
                    else:
                        obj[k] = v
                elif s == 'ndt':
                    obj['no_defaults'] = True
                elif s == 'adt':
                    obj['adetailer'] = True
                elif s == 'cn':
                    obj['controlnet'] = True
        except Exception as e:
            return traceback.format_exception_only(e)
        message = message[match.span()[1]:].strip()
        if 'controlnet' in obj and cnet:
            obj['cnet_cfg'] = cnet
    obj['prompt'] = message
    return ''


def splitPrompts(prompt: str):
    matches = re.findall(r'\{[^}]+}', prompt)
    if not matches: return [prompt]
    l = []
    for match in matches:
        trim = match[1:-1].strip()
        if '|' in trim:
            split = trim.split('|')
            for s in split:
                newprompt = prompt.replace(match, s.strip(), 1)
                l += splitPrompts(newprompt)
    return l


def formatPrompt(key: str, nodef: bool, obj: dict, defSettings: dict, shouldFormat=True):
    exists = key in obj
    if not defSettings or key not in defSettings:
        return obj[key] if exists else payload[key]
    defPrompt = defSettings[key].strip()
    if exists:
        if nodef or not defPrompt:
            key = obj[key]
        elif shouldFormat and '{}' in defPrompt:
            key = defPrompt.format(obj[key])
        else:
            key = defPrompt + (' ' if defPrompt.endswith(',') else ', ') + obj[key]
    else:
        key = '' if nodef else defPrompt
    return key


def formatWildcards(prompt: str) -> str:
    if wildcardDir:
        if matches := re.findall(r'__[^_]+__', prompt):
            for match in matches:
                trimmed = match[2:-2].split('|')
                if len(trimmed) > 1:
                    trimmed[0] = random.choice(trimmed)
                trimmed = trimmed[0].strip()
                for card in wildcardList:
                    if trimmed == card:
                        with open(wildcardDir.joinpath(card + '.txt'), 'r') as f:
                            tags = f.read().splitlines()
                            prompt = prompt.replace(match, random.choice(tags))
                            f.close()
                            break
    return prompt


def formatLoras(prompt:str) -> str:
    blah = dict(loras)
    if loras:
        if matches := re.findall(r'(#[^,]+),?', prompt):
            for match in matches:
                split = match.split(':')
                trimmed = split[0][1:].strip()
                weight = 1 if len(split) == 1 else float(split[1])
                for lora in loras:
                    if trimmed.lower() in lora.lower():
                        lname = f'<lora:{lora}:{weight}>'
                        if lname not in prompt:
                            prompt = prompt.replace(match, f'{lname} {loras[lora][2] or ''}')
                        continue
    return prompt

def validateSettings(actual: dict, obj: dict, defSettings: dict, shouldFormat=True, genType='txt2img'):
    if defSettings is None:
        defSettings = payload
    img = genType == 'img2img'
    nodef = 'no_defaults' in obj
    adt = 'adetailer' in obj
    mRes = 1536 if img and not adt else 1024
    for k in ['prompt', 'negative_prompt']:
        actual[k] = formatPrompt(k, nodef, obj, defSettings, shouldFormat)
    if 'res' in obj:
        if r := parseRes(mRes, obj, obj.pop('res'), 'width', 'height'):
            return r
    if 'width' in obj and 'height' in obj:
        width = obj['width']
        height = obj['height']
    else:
        width = defSettings['width']
        height = defSettings['height']
    if not adt:
        if 'img_scale' in obj:
            scale = obj['img_scale']
            width *= scale
            height *= scale
        elif 'img_scale' in defSettings:
            scale = defSettings['img_scale']
            width *= scale
            height *= scale
    if width * height > (mRes**2):
        x = mRes / max(width, height)
        width = int(width * x)
        height = int(height * x)
    obj['width'] = int(width - (width % 8))
    obj['height'] = int(height - (height % 8))
    if 'ares' in obj:
        if r:= parseRes(mRes, obj, obj.pop('ares'), 'ad_inpaint_width', 'ad_inpaint_height'):
            return r
    if 'ad_inpaint_width' in obj and 'ad_inpaint_height' in obj:
        if adt:
            width = obj['ad_inpaint_width']
            height = obj['ad_inpaint_height']
            obj['ad_inpaint_width'] = int(width - (width % 8))
            obj['ad_inpaint_height'] = int(height - (height % 8))
        else:
            obj.pop('ad_inpaint_width')
            obj.pop('ad_inpaint_height')
    if 'ad_denoising_strength' in obj and not adt:
        obj.pop('ad_denoising_strength')
    for k, v in obj.items():
        if k == 'prompt' or k == 'negative_prompt':
            continue
        elif k == 'sampler_name':
            if v in samplers:
                actual[k] = v
            else:
                return f'No sampler found for \"{v}\", use `!autism samplers` to see the list of available samplers.'
        elif k == 'steps':
            actual[k] = max(min(int(v), 24), 1)
        else:
            actual[k] = v
    return ''


def prettyPrintSettings(obj: dict, seed: int = -1, dflt: dict = None, format=False):
    prettyPrintMapping = {
        'no_defaults': 'ndt',
        'negative_prompt': 'neg',
        'cfg_scale': 'cfg',
        'steps': 'stp',
        'sampler_name': 'sampler',
        'denoising_strength': 'den',
        'img_scale': 'scale',
        'ad_denoising_strength': 'aden',
        'ad_model': 'amode'
    }
    skip = ['author', 'prompt', 'seed', 'init_images', 'infotext', 'include_init_images', 'img_scale', 'controlnet',
            'cnet_cfg',
            'image_url']
    merged = dict(dflt | obj) if dflt else obj
    if 'image_url' in merged:
        reply = f'!autism <{merged['image_url']}> ['
    else:
        reply = '!autism ['
    res = 'wxh'
    ares = 'wxh'
    defres = f'{payload['width']}x{payload['height']}'
    cns = ''
    if 'controlnet' in merged:
        cns = 'cn:{wt};'
    try:
        for k, v in merged.items():
            if k == 'no_defaults':
                reply += 'ndt;'
            elif k == 'adetailer':
                reply += 'adt;'
            elif k == 'cnet_cfg' and 'controlnet' in merged:
                cnet = merged['cnet_cfg']
                if 'weight' in cnet:
                    cns = cns.replace('{wt}', str(cnet['weight']), 1)
                if 'guidance_end' in cnet:
                    if 'guidance_start' in cnet:
                        cns += f'cnstp:{cnet['guidance_start']}, {cnet['guidance_end']};'
                    else:
                        cns += f'cnstp:{cnet['guidance_end']};'
                if 'control_mode' in cnet:
                    mode = cnet['control_mode']
                    if type(mode) == int:
                        cns += f'cnmode:{'balanced' if mode == 0 else 'prompt' if mode == 1 else 'cnet'};'
                    else:
                        for cnk, cnv in cn_modes.items():
                            if mode == cnv:
                                cns += f'cnmode:{cnk};'
                                break
            elif k in skip:
                continue
            elif k == 'width':
                res = res.replace('w', str(v))
            elif k == 'height':
                res = res.replace('h', str(v))
            elif k == 'ad_inpaint_width':
                ares = ares.replace('w', str(v))
            elif k == 'ad_inpaint_height':
                ares = ares.replace('h', str(v))
            elif k == 'negative_prompt':
                reply += f'{prettyPrintMapping[k]}: {formatPrompt(k, not dflt, obj, dflt, format)};'
            else:
                reply += f'{prettyPrintMapping[k] if k in prettyPrintMapping else k}: {v};'
    except Exception as e:
        return traceback.format_exception_only(e)
    if ares != 'wxh':
        reply += f'ares: {ares};'
    if cns:
        reply += cns.replace(':{wt}', '',1)
    if res != 'wxh' and res != defres or obj is payload:
        reply += f'res: {res};'
    if 'img_scale' in merged:
        reply += 'scale: 1;' if format else f'scale: {merged['img_scale']}'
    if seed > -1:
        reply += f'seed: {seed}'
    elif 'seed' in obj and (int(obj['seed']) > -1 or obj is payload):
        reply += f'seed: {obj['seed']}'
    else:
        reply += 'seed: -1'
    reply += f'] {formatPrompt('prompt', not dflt, obj, dflt, format)}'
    return reply.replace(';]', ']', 1).replace('[] ', '', 1)


def checkChannel(ctx: Context):
    if ctx.channel.id != channelId or ctx.author == bot.user:
        return False
    return True


@bot.group(invoke_without_command=True)
@commands.check(predicate=checkChannel)
async def autism(ctx: Context, *, message=''):
    """
    **Generate an image from text**
    Usage guidelines:
    **!autism <prompt>** - simple generation request using a prompt appended to default settings
    **!autism [settings] <prompt>** - tweaked generation request using params overlaid on top of default settings

    Common setting explanations:
        **ndt** - makes positive/negative prompts provided by you override defaults, __passed without a value__
        **neg** - 'negative_prompt', hints the model to avoid certain tags and concepts *(**not** guaranteed; default: rating_explicit)*
        **stp** - 'steps', amount of iterations the model makes, more isn't always better *(default: 15)*
        **cfg** - 'cfg_scale', how hard the model should bake the image per iteration, more than necessary and you'll get artifacts *(default: 3)*
        **seed** - the seed used for generation, can be useful if you find a good base image but want to tweak the prompt *(default: random, shown after generation)*
        **res** - 'width' and 'height', how big the end image should be, capped to 1024x1024 by total amount of pixels so you can play with aspect ratios *(default: 768x1024)*
        **sampler** - 'sampler_name', governs how the image is generated - different samplers need different settings and not all of them work *(default: Euler a)*

        All the settings above have to be enclosed in square brackets and separated by semicolons like this: *[ndt;neg: 3d; seed: 1234]*
    Examples:
        `!autism [ndt;neg: 3d, source_furry, source_cartoon, source_anime;cfg: 2.5;stp: 18;sampler: DPM++ 2M Karras] source_anime, 1girl, solo`

        `!autism [ndt;neg:rating_explicit] gjem, bowsette, 1girl, solo, bunnysuit, (small breasts:0.8008), (tired face), standing behind counter, bar background, depth of field, question mark, looking at viewer`
            ^ https://discord.com/channels/237433042413551617/1191321188262232084/1304371901648474153
        `!autism [seed: 2311242424] <lora:mcnsfwv7_8dim_pdxl:0.9> slime girl, 1girl, solo, naked, featureless breasts, featureless crotch, dripping, simple background, on knees`
            ^ https://discord.com/channels/237433042413551617/1191321188262232084/1304867298733064343
    """
    if ctx.invoked_subcommand is not None:
        return
    if not message:
        await ctx.send_help('autism')
        return
    global genNum
    obj = {'gen_type': 'txt2img'}
    message = matchSettings(obj, message)
    notcnt = 'controlnet' not in obj
    if message:
        sendMessage(message, ref=ctx.message)
        return

    def imgCheck(msg: discord.Message, obj: dict):
        if msg.attachments:
            img = msg.attachments[0]
            if img.content_type.endswith('/png') or img.content_type.endswith('/jpeg'):
                if notcnt or 'denoising_strength' in obj:
                    obj['gen_type'] = 'img2img'
                if not 'width' in obj:
                    obj['width'] = img.width
                if not 'height' in obj:
                    obj['height'] = img.height
                obj['init_images'] = [img.url]
        if obj['gen_type'] == 'txt2img' and 'img_scale' in obj and notcnt:
            obj.pop('img_scale')

    if 'image_url' in obj:
        if notcnt or 'denoising_strength' in obj: obj['gen_type'] = 'img2img'
    elif ctx.message.attachments:
        imgCheck(ctx.message, obj)
    elif ctx.message.type is discord.MessageType.reply:
        reply = await ctx.message.channel.fetch_message(ctx.message.reference.message_id)
        imgCheck(reply, obj)
    split = splitPrompts(obj['prompt'])
    obj['SD'] = {'author': ctx.author.name,
                 'timestamp': timestamp(),
                 'ref': ctx.message}
    if len(split) > 1:
        i = 0
        if not 'seed' in obj:
            obj['seed'] = random.randint(1, 2 ** 31 - 1)
        for s in split:
            qObj = dict(obj)
            qObj['prompt'] = s
            qObj['SD'] = dict(obj['SD'])
            qObj['SD']['timestamp'] += '-' + str(i + 1)
            q.put((genNum + i, qObj))
            i += 1
            genNum += 1
        await ctx.reply(f'Queued up {len(split)} prompts.')
        return
    q.put((genNum, obj))
    genNum += 1


@autism.command(name='help')
async def seekHelp(ctx: Context, message: str = ''):
    try:
        await ctx.send_help('autism ' + message)
    except Exception as e:
        await ctx.reply(traceback.format_exception_only(e)[0])


@autism.command()
async def cancel(ctx: Context):
    """
    **Sends an 'interrupt' command to the generation queue**
    Fails if the sender didn't queue the active generation.
    """
    global currentGen
    if currentGen is None:
        sendMessage(f'Nothing is being generated right now.')
    elif True:  # ctx.author.name == currentGen['author'] or ctx.author.get_role(751937393292214292):
        response = requests.post(url='http://127.0.0.1:7860/sdapi/v1/interrupt')
        if response.ok and 'author' in currentGen:
            currentGen.pop('author')
            sendMessage(f'Aborted generation for {prettyPrintSettings(currentGen, format=True)}', ref=ctx.message)
            currentGen = None


@autism.command(invoke_without_command=True, aliases=['defaults'])
async def default(ctx: Context, genType: str = '', *, message: str = ''):
    """
    **Print or change default generation settings**
    Usage guidelines:
    **!autism default** - print current defaults, either global or set by the user
    **!autism default <type> <settings>** - update user defaults for generation type with soft overrides per-setting
        ^ Types can be **txt** or **img**
    **!autism default clear** - reset user defaults to global

    All of the settings are identical to ones you'd use with **!autism**, except the positive/negative prompts.
    You can use curly brackets **{}** to specify a point in the default prompt where you will be able to insert the rest of the prompt while generating images.
    Examples:
        `!autism default txt [neg:source_furry]` - replaces the default **rating_explicit** negative tag with `source_furry`.
            Any negatives passed during generation will be appended after the default.
        `!autism default img [neg: realistic, {}, 3d] {}, vector, flat color, 2d` - uses curly brackets to specify insertion points for the default prompts.
            Using generation with `!autism [neg:simple background] interior, bedroom` would generate an image
            using these combined prompts: `[neg: realistic, simple background, 3d] interior, bedroom, vector, flat color, 2d`
    """
    author = ctx.author.name
    exists = author in userdata
    fp = os.path.join('data', author + '.json')
    if not genType:
        s = 'Default generation parameters:\n'
        if exists:
            for t in ['txt2img', 'img2img']:
                if t in userdata[author]:
                    s += f'**{t}**: `{prettyPrintSettings(userdata[author][t], format=False)}`\n'

        return await ctx.reply(s if exists else f'txt2img/img2img: `{prettyPrintSettings(payload)}`')
    elif genType == 'clear':
        userdata.pop(author)
        os.remove(fp)
        return await ctx.reply('Purged user data for ' + author)
    elif not genType in ['txt', 'img']:
        return await ctx.reply('Tried to set default settings for invalid generation type.\n'
                               'Try using `!autism default txt <settings>` or `!autism default img <settings>`.')
    genType += '2img'
    user = userdata[author] if exists else {}
    actual = dict(user[genType]) if genType in user else {}
    pending = {}
    obj = {}
    reply = matchSettings(obj, message)
    if reply:
        sendMessage(
            'You used the wrong formatting somewhere.\nDefault settings not updated.\nExample config: [neg:rating_explicit, 3d;cfg:5.5;stp:18;seed=42069;ndt]')
    validationerror = validateSettings(pending, obj, actual, shouldFormat=False)
    if validationerror:
        await ctx.reply(validationerror)
        return
    obj.clear()
    changed = False
    if actual:
        for k, v in pending.items():
            if k not in actual or actual[k] != v:
                changed = True
                actual[k] = v
    else:
        changed = True
        actual = pending
    if changed:
        reply = prettyPrintSettings(actual, format=False)

        user[genType] = actual
        userdata[author] = user
        with open(fp, 'w') as file:
            json.dump(userdata[author], file)
            file.close()
        await ctx.reply(f'Updated default generation parameters for {genType}:  {reply}')
    else:
        await ctx.reply('Default settings already set to those values. Use `!autism default` to check.')


@autism.command()
async def wildcards(ctx: Context, *, name: str = None):
    """
        **List available wildcards or their contents**
        Wildcards let you apply a random tag from a predefined list to your prompt.
        Insertion into the prompt is done with `__double underlines__` (not escaped with slashes).
        Most of these wildcards are sourced from 4chan pads, with minor edits or replacements using danbooru tags.

        Usage guidelines:
        **!autism wildcards** - print all of the existing wildcards
        **!autism wildcards <name>** - print the contents of a specific wildcard

        Examples:
            `!autism wildcards angle` - shows you which tags are defined in the __angle__ wildcard
            `!autism 1girl, __character__, __clothing__, __clothing print__, __clothing style__`
                ^ will generate an image using 4 random tags from the respective wildcards.
    """
    if not wildcardDir:
        return await ctx.reply('No wildcards are available due to missing extension.')
    embed = Embed(colour=discord.Colour.blurple())
    if name:
        file = wildcardDir.joinpath(name + '.txt')
        if file.exists():
            embed.title = f'Wildcard: {name}'
            with open(file, 'r') as f:
                embed.description = ' | '.join(f.read().splitlines())
                f.close()
        else:
            return await ctx.reply(f'Couldn\'t find any wildcards named \"{name}\".')
    elif wildcardDir:
        cards = sorted(wildcardDir.glob('*.txt'))
        embed.title = 'Available wildcards:'
        embed.description = ' | '.join([card.stem for card in cards])
    await ctx.reply(embed=embed)


@autism.group()
async def civitai(ctx: Context):
    pass


@civitai.command()
async def search(ctx: Context, query: str, page: int = 1, limit: int = 20, tag: str | None = None):
    models = civclient.models.list_models(limit, page=page, query=query, tag=tag, sort=capi.ModelSort.HIGHEST_RATED,
                                          period=capi.ModelPeriod.ALL_TIME, types=[capi.models.ModelType.LORA, capi.models.ModelType.LoCon],
                                          base_models=[capi.models.BaseModel.PONY, capi.models.BaseModel.ILLUSTRIOUS, capi.models.BaseModel.NOOBAI])
    reply = f'Search results for {query}:\n'
    cache = {}
    for model in models.items:
        versions = ''
        for version in model.modelVersions:
            if version.baseModel == capi.models.BaseModel.PONY.value or version.baseModel == capi.models.BaseModel.ILLUSTRIOUS.value or version.baseModel == capi.models.BaseModel.NOOBAI.value:
                versions += f'**!autism civitai info {version.id}** | Model: {'[🐴💖]' if version.baseModel == BaseModel.PONY.value else '[🇽🇱]'} | Version: __{version.name}__'
                if version.trainedWords:
                    versions += f'| Trigger Words:\n```\n{', '.join(version.trainedWords)}\n```\n'
                else:
                    versions += '\n'
        if versions:
            cache[model.id] = {
                'name': model.name,
                'tags': model.tags,
                'stats': model.stats,
                'nsfw': model.nsfw,
                'versions': versions
            }
    if not cache:
        await ctx.reply(
            f'Couldn\'t find any Pony models for \"{query}\" on page {page} with a limit of {limit} entries.')
        return
    for modelid, obj in cache.items():
        reply += f'### {obj['name']}\n'
        tagnum = len(obj['tags'])
        reply += f'Tags: *{', '.join(obj['tags']) if tagnum < 5 else ', '.join(obj['tags'][:4]) + ' +' + str(tagnum - 4)}*| Upboats: *{obj['stats'].get('thumbsUpCount')}* | NSFW: *{obj['nsfw']}*\n'
        reply += obj['versions']
    replies = wrap(reply, 1900, expand_tabs=False, replace_whitespace=False, break_long_words=False,
                   drop_whitespace=False)

    for reply in replies:
        await ctx.reply(reply)


def sample_image_meta(embed: Embed, name: str, image: capi.models.ModelVersionImage):
    def populateData(key: str, obj: dict, meta: dict):
        if key in meta:
            obj[key] = meta[key]

    meta = image.meta
    data = {}
    populateData('seed', data, meta)
    warning = ''
    if 'prompt' in meta:
        prompt = meta['prompt']
        if matches := re.findall(r'<lora:([^:]+):[0-9.]+>', prompt, flags=RegexFlag.MULTILINE):
            flag = True
            for match in matches:
                if match == name:
                    flag = False
                    break
            if flag: warning = f'*This prompt does **not** contain `<lora:{name}:1>`.*\n'
        else:
            warning = f'*This prompt does **not** contain `<lora:{name}:1>`.*\n'
        data['prompt'] = prompt

    populateData('steps', data, meta)
    if 'negativePrompt' in meta:
        neg = meta['negativePrompt']
        if len(neg) < 75:
            data['negative_prompt'] = meta['negativePrompt']
        else:
            data['negative_prompt'] = '*more than 75 tokens of schizo negs*'
    if 'Size' in meta:
        res = meta['Size'].split('x')
        data['width'] = res[0]
        data['height'] = res[1]
        if 'Upscale factor' in meta:
            data['width'] /= meta['Upscale factor']
            data['height'] /= meta['Upscale factor']
    if 'cfgScale' in meta:
        data['cfg_scale'] = meta['cfgScale']
    if 'sampler' in meta:
        sampler = meta['sampler']
        if 'Schedule type' in meta:
            sampler += ' ' + meta['Schedule type']
        if sampler in samplers:
            data['sampler_name'] = sampler

    if data:
        embed.add_field(name='Sample Prompt', value=warning + prettyPrintSettings(data, format=True), inline=False)


def download_civitai_image(filename: str, url: str):
    file = loraDir.joinpath(filename)
    if file.exists():
        return file
    if unthumb := re.search(r'/(width=\d+)/', url):
        url = url.replace(unthumb.group(1), 'original=true,quality=90')
    content = requests.get(url).content
    with open(file, 'wb') as f:
        f.write(content)
        f.close()
    return file


def loadLoraInfo(name: str) -> dict:
    with open(name, 'rb') as f:
        file = json.load(f)
        f.close()
        return file


def parseLoraInfo(embed: Embed, model: ModelVersion, filename: str) -> str:
    prompt = f'<lora:{filename}:1> '
    embed.set_author(name=prompt)
    mBase = '[🐴💖] ' if model.baseModel == BaseModel.PONY.value else '[🇽🇱] '
    embed.title = f'{mBase} {model.name}'
    #if model.description:
    #   descr = stripHTML(model.description)
    #   descr = wrap(descr, 400, expand_tabs=False, replace_whitespace=False, break_long_words=False,
    #                drop_whitespace=False)
    #   embed.description = descr[0]
    #f 'description' in model.model and model.description != model.model['description']:
    #   descr = wrap(stripHTML(model.model['description']), 400, expand_tabs=False, replace_whitespace=False,
    #                break_long_words=False,
    #                drop_whitespace=False)
    #  embed.add_field(name='About Version', value=descr[0], inline=False)
    if model.trainedWords:
        triggers = list()
        for w in model.trainedWords:
            triggers.extend(w.split(','))
        triggers = list(dict.fromkeys(triggers))
        prompt += ', '.join(triggers[:4] if len(triggers) > 4 else triggers)
        embed.add_field(name='Trigger Words', value=', '.join(triggers), inline=False)
    embed.add_field(name='Civitai Page', value=f'https://civitai.com/models/{model.modelId}?modelVersionId={model.id}')
    if model.images and model.images[0].meta is not None:
        sample_image_meta(embed, filename, model.images[0])
    return f'`{prompt}`'


@civitai.command("info")
async def c_info(ctx: Context, model_version: int):
    version = civclient.model_versions.get_model_version(model_version)
    file = version.files[0]
    for f in version.files:
        if f.primary:
            file = f
            break
    embed = Embed(colour=Colour.blurple())
    image = version.images[0]
    filename = file.name[:-12]
    img = download_civitai_image(filename + '.preview.jpg', image.url)
    prompt = parseLoraInfo(embed, version, filename)
    await ctx.reply(content=prompt, embed=embed, file=File(img))


@civitai.command('download')
async def c_download(ctx: Context, model_version: int):
    obj = requests.get('https://civitai.com/api/v1/model-versions/' + str(model_version)).json()
    if 'error' in obj:
        await ctx.reply(
            f'Couldn\'t find model with the id {model_version}, make sure you are passing a model **version** id and not the base one')
        return
    elif not obj['model']['type'] in ['LORA', 'LoCon']:
        await ctx.reply(f'This can only download LoRA and LoCon/LyCORIS tensors.')
        return
    file = obj['files'][0]
    filename = file['name']
    info = filename.replace('safetensors', 'civitai.info')
    if not obj['description']:
        model = civclient.models.get_model(obj['modelId'])
        description = model.description
    else:
        description = obj['description']
    obj['description'] = description
    sdinfo = filename.replace('safetensors', 'json')
    await c_info(ctx, model_version)
    result = download_file('https://civitai.com/api/download/models/' + str(model_version), str(loraDir), civToken)
    if result:
        with open(loraDir.joinpath(info), 'w') as f:
            json.dump(obj, f, indent=4)
            f.close()

        sdi = {
            'description': description,
            'sd version': 'SDXL',
            'activation text': obj['triggerWords'] if 'triggerWords' in obj else '',
            'preferred weight': 0
        }
        with open(loraDir.joinpath(sdinfo), 'w') as f:
            json.dump(sdi, f, indent=4)
            f.close()

        getLoras()
        await ctx.reply(result)
    else:
        await ctx.reply(f'Download for `{filename}` failed.')


@autism.command(name='samplers')
async def listSamplers(ctx: Context):
    s = 'Available samplers:'
    response = requests.get(url='http://127.0.0.1:7860/sdapi/v1/samplers').json()
    for obj in response:
        s += f' {obj['name']},'
    sendMessage(s, ref=ctx.message)


@autism.command(name='lora', aliases=['loras'])
async def listLoras(ctx: Context, *, message: str = ''):
    """
    **List all or find a specific installed LoRA**
    Reducing the lookup to a single LoRA will show detailed information about it with trigger words
    Command can be 'lora' or 'loras'

    `!autism loras` - list all of the installed LoRAs
    `!autism loras <string>` - filter LoRAs that contain a specific string in the filename;
    `!autism loras mc' returns '<lora:mcnsfwv7_8dim_pdxl:1>, <lora:MCXL:1>'
    `!autism lora ocarina` returns '<lora:Ocarina64-low_poly:1> low poly, an animated character'
    """
    s = ', '
    li = []
    if not message:
        for l in loras:
            li.append(f'`<lora:{l}:1>`')
    else:
        for l in loras:
            if message.lower() in l.lower():
                li.append(f'<lora:{l}:1>')
        if len(li) > 1:
            li = [f'`{l}`' for l in li]
    if not li:
        return await ctx.reply(f'Couldn\'t find any LoRAs for \"{message}\".')
    elif len(li) == 1:
        embed = Embed(colour=Colour.blurple())
        lora = loras[li[0].split(':')[1]]
        info = loadLoraInfo(lora[0])
        files = info['files']
        file = files[0]
        for f in files:
            if f['primary']:
                file = f
                break
        reply = parseLoraInfo(embed, civclient._model_version._parse_model_version(info), file['name'][:-12])
        img = lora[1]
        await ctx.reply(reply, embed=embed, file=None if img is None else File(img))
    else:
        reply = f'\n**Listed {len(li)} LoRAs.** *To view details for a specific LoRA, use filtering to reduce the amount of results down to 1.*'
        reply = wrap(s.join(li) + reply, 1900, expand_tabs=False, replace_whitespace=False, break_long_words=False,
                     drop_whitespace=False)
        for s in reply:
            await ctx.reply(s)


def stripHTML(s: str) -> str:
    if not s:
        return ''
    elif html := re.findall(r'<[^>\n]+>', s):
        for h in html:
            s = s.replace(h, '')
    return s


def timestamp():
    return Date.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def genAndRespond(obj: dict):
    SD = obj.pop('SD')
    genType = obj.pop('gen_type')
    dflt = dict(payload)
    if genType == 'img2img':
        dflt = dict(dflt | imgDefaults)
    author = SD['author']
    exists = author in userdata and genType in userdata[author]
    settings = dict(dflt | userdata[author][genType]) if exists else dflt
    try:
        burp = validateSettings(settings, obj, userdata[author][genType] if exists else dflt, genType=genType)
        if burp:
            sendMessage(burp, ref=SD['ref'])
            return
    except Exception as e:
        sendMessage(traceback.format_exception_only(e)[0], ref=SD['ref'])
    for s in ['prompt', 'negative_prompt']:
        match = re.search(placeholderRemover, settings[s])
        if match:
            settings[s] = settings[s].replace(match.group(1), '')

    edited = formatWildcards(formatLoras(settings['prompt']))
    obj['prompt'] = settings['prompt'] = edited

    settings = copy.deepcopy(settings | alwayson)
    extensionPayload(settings, genType)
    reply = f'```\n{prettyPrintSettings(obj, -1, userdata[author][genType] if exists else None, True)}```'
    response = requests.post(url=sdapi(genType), json=settings)
    if response.ok:
        response = response.json()
    else:
        sendMessage(message='The API isn\'t responding. This is probably very bad.')
        return
    info = json.loads(response['info'])
    reply = reply.replace('seed: -1', f'seed: {info['seed']}', 1)
    filepath = outputDir + author + '/'
    os.makedirs(filepath, exist_ok=True)
    filepath += SD['timestamp'] + '.png'
    if 'images' in response:
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(response['images'][0]))
            f.close()
        sendMessage(message=reply, file=File(filepath), ref=SD['ref'])
    else:
        sendMessage(message=reply + 'Output contains no image data, possibly due to cancelled generation.',
                    ref=SD['ref'])


def extensionPayload(settings: dict, genType: str):
    if 'adetailer' not in settings and '[SEP]' in settings['prompt']:
        settings['alwayson_scripts']['forge couple']['args'][0] = True
    if 'controlnet' in settings:
        settings.pop('controlnet')
        cnet = settings.pop('cnet_cfg') if 'cnet_cfg' in settings else None
        if 'init_images' in settings:
            cn_settings = dict(cn_payload)
            if cnet:
                cn_settings.update(cnet)
            if 'image_url' in settings:
                cn_settings['image'] = settings['init_images'][0]
            else:
                img = requests.get(settings['init_images'][0])
                image_bytes = BytesIO(img.content)
                encoded_image = base64.b64encode(image_bytes.read()).decode()
                cn_settings['image'] = encoded_image
            settings['alwayson_scripts']['ControlNet']['args'][0].update(cn_settings)

    if genType == 'img2img' and 'adetailer' in settings:
        settings.pop('adetailer')
        settings['alwayson_scripts'].pop('forge couple')
        ad_settings = settings['alwayson_scripts']['ADetailer']['args']
        ad_settings[0] = True
        ad_settings[1] = True
        ad_obj = {'ad_prompt': settings['prompt'], 'ad_negative_prompt': settings['negative_prompt'],
                  'ad_model': ad_models[settings['ad_model']] if 'ad_model' in settings else ad_models['mask']}
        if 'ad_denoising_strength' in settings:
            ad_obj['ad_denoising_strength'] = settings.pop('ad_denoising_strength')
        if 'ad_inpaint_width' in settings and 'ad_inpaint_height' in settings:
            ad_obj['ad_use_inpaint_width_height'] = True
            ad_obj['ad_inpaint_width'] = settings.pop('ad_inpaint_width')
            ad_obj['ad_inpaint_height'] = settings.pop('ad_inpaint_height')
        ad_obj['ad_use_sampler'] = True
        ad_obj['ad_sampler'] = settings['sampler_name']
        ad_settings.append(ad_obj)


def getSamplers():
    global samplers
    response = requests.get(url='http://127.0.0.1:7860/sdapi/v1/samplers').json()
    for obj in response:
        samplers.append(obj['name'])


def loadUserData():
    os.makedirs('data', exist_ok=True)
    for filename in os.listdir('data'):
        filename = os.fsdecode(filename)
        if filename.endswith('.json'):
            with open(os.path.join('data', filename)) as file:
                filename = filename[:-5]
                userdata[filename] = json.load(file)
                file.close()


def getLoras(refresh=True):
    global loras
    if refresh and requests.post(url=sdapi('refresh-loras')).ok:
        loras.clear()
    response = requests.get(url=sdapi('loras')).json()
    for obj in response:
        if 'metadata' in obj and 'ss_base_model_version' in obj['metadata'] and 'xl' in obj['metadata'][
            'ss_base_model_version'].lower():
            s = ''
            i = None
            t = None
            pth = Path(obj['path'].replace('safetensors', 'civitai.info'))
            if pth.exists():
                s = pth
                with open(pth, 'r') as f:
                    info = json.load(f)
                    triggers = list()
                    for w in info['trainedWords']:
                        if len(triggers) > 4: break
                        triggers.extend(w.split(','))
                    triggers = list(dict.fromkeys(triggers))
                    t = ','.join(triggers[:4] if len(triggers) > 4 else triggers)
            else:
                t = ''


            pth = pth.parent.glob(f'{pth.stem[:-8] + '.preview'}.*')
            if pth:
                for f in pth:
                    if f.suffix == '.png' or f.suffix == '.jpg':
                        i = f
                        break
            loras[obj['name']] = (s, i, t)


def getWildcards():
    global wildcardList
    wildcardList.clear()
    if wildcardDir:
        files = wildcardDir.glob('*.txt')
        for file in files:
            wildcardList.append(file.stem)

@bot.event
async def on_raw_reaction_add(evt: RawReactionActionEvent):
    if evt.channel_id == channelId and evt.emoji.name == '🔞':
        message = bot.get_message(evt.message_id) or await bot.get_channel(evt.channel_id).fetch_message(evt.message_id)
        if message.author.id == bot.user.id and message.attachments and 'SPOILER_' not in message.attachments[0].filename:
            file = await message.attachments[0].to_file(use_cached=True, spoiler=True)
            await message.edit(attachments=[])
            await message.edit(file=file)

@bot.event
async def on_raw_reaction_remove(evt: RawReactionActionEvent):
    if evt.channel_id == channelId and evt.emoji.name == '🔞':
        message = bot.get_message(evt.message_id) or await bot.get_channel(evt.channel_id).fetch_message(evt.message_id)
        if message.author.id == bot.user.id and not message.reactions and message.attachments and 'SPOILER_' in message.attachments[0].filename:
            file = await message.attachments[0].to_file(use_cached=True, spoiler=False)
            file.filename = file.filename.replace('SPOILER_', '')
            await message.edit(attachments=[])
            await message.edit(file=file)

def mainThread():
    global currentGen
    while True:
        if q.not_empty:
            try:
                currentGen = obj = q.get()[1]
                genThread = Thread(target=genAndRespond, args=[obj])
                genThread.start()
                genThread.join()
                currentGen = None
                q.task_done()
            except Exception as e:
                pass
        sleep(4)


mainT = Thread(target=mainThread, daemon=True)
mainT.start()
bot.run(mToken)
