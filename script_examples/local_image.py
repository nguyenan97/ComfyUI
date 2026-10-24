"""Render an SD 1.5 image through the local ComfyUI API and save its measurements."""
import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path
from urllib import request
from urllib.error import HTTPError

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = 'v1-5-pruned-emaonly-fp16.safetensors'


def api(base, path, payload=None):
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    req = request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    try:
        with request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(error.read().decode('utf-8', errors='replace')) from error


def ui_workflow(graph, node_info):
    layout = {
        '1': ([40, 240], [300, 110]), '2': ([400, 0], [370, 180]),
        '3': ([400, 210], [370, 140]), '4': ([400, 390], [300, 110]),
        '5': ([830, 140], [310, 280]), '6': ([1200, 150], [220, 80]),
        '7': ([1470, 150], [450, 520]),
    }
    nodes = {}
    links = []
    for key, value in graph.items():
        info = node_info[value['class_type']]
        position, size = layout[key]
        outputs = [{'name': name, 'type': kind, 'links': [], 'slot_index': index}
                   for index, (name, kind) in enumerate(zip(info['output_name'], info['output']))]
        nodes[key] = {'id': int(key), 'type': value['class_type'], 'pos': position, 'size': size,
                      'flags': {}, 'order': int(key) - 1, 'mode': 0, 'inputs': [], 'outputs': outputs,
                      'properties': {'Node name for S&R': value['class_type']}, 'widgets_values': []}
    for key, value in graph.items():
        node = nodes[key]
        for name, setting in value['inputs'].items():
            if isinstance(setting, list):
                source, slot = setting
                output = nodes[source]['outputs'][slot]
                link = len(links) + 1
                links.append([link, int(source), slot, int(key), len(node['inputs']), output['type']])
                node['inputs'].append({'name': name, 'type': output['type'], 'link': link})
                output['links'].append(link)
            else:
                node['widgets_values'].append(setting)
                if value['class_type'] == 'KSampler' and name == 'seed':
                    node['widgets_values'].append('fixed')
    return {'last_node_id': len(nodes), 'last_link_id': len(links), 'nodes': list(nodes.values()),
            'links': links, 'groups': [], 'config': {}, 'extra': {}, 'version': 0.4}


def generate(args):
    system = api(args.server, '/system_stats')['system']
    graph = {
        '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': CHECKPOINT}},
        '2': {'class_type': 'CLIPTextEncode', 'inputs': {'text': args.prompt, 'clip': ['1', 1]}},
        '3': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'text, watermark, logo, blurry, low quality, distorted, extra limbs', 'clip': ['1', 1]}},
        '4': {'class_type': 'EmptyLatentImage', 'inputs': {'width': args.width, 'height': args.height, 'batch_size': 1}},
        '5': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['2', 0], 'negative': ['3', 0], 'latent_image': ['4', 0], 'seed': args.seed, 'steps': args.steps, 'cfg': 7.0, 'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': 1.0}},
        '6': {'class_type': 'VAEDecode', 'inputs': {'samples': ['5', 0], 'vae': ['1', 2]}},
        '7': {'class_type': 'SaveImage', 'inputs': {'images': ['6', 0], 'filename_prefix': 'balanced_local'}},
    }
    workflow = ui_workflow(graph, api(args.server, '/object_info'))
    workflow_folder = ROOT / 'user' / 'default' / 'workflows'
    workflow_folder.mkdir(parents=True, exist_ok=True)
    (workflow_folder / 'balanced-sd15.json').write_text(json.dumps(workflow, indent=2), encoding='utf-8')
    started = time.monotonic()
    prompt_id = api(args.server, '/prompt', {'prompt': graph, 'client_id': uuid.uuid4().hex,
                                          'extra_data': {'extra_pnginfo': {'workflow': workflow}}})['prompt_id']
    print(f'Queued {prompt_id}: {args.width}x{args.height}, {args.steps} steps', flush=True)  # noqa: T201
    peak = 0
    observed_pids = set()
    while time.monotonic() - started < args.timeout:
        snapshot = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        if snapshot.returncode == 0:
            peak = max(peak, int(snapshot.stdout.strip().splitlines()[0]))
        processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        if processes.returncode == 0:
            observed_pids.update(int(line) for line in processes.stdout.splitlines() if line.strip().isdigit())
        item = api(args.server, f'/history/{prompt_id}').get(prompt_id)
        if item:
            status = item.get('status', {})
            if status.get('status_str') == 'error':
                raise RuntimeError(json.dumps(status, ensure_ascii=False))
            images = item.get('outputs', {}).get('7', {}).get('images', [])
            if images:
                image = images[0]
                output_folder = (ROOT / 'output').resolve()
                destination = (output_folder / image['subfolder'] / image['filename']).resolve()
                if not destination.is_relative_to(output_folder):
                    raise ValueError('Image response points outside the output folder')
                with Image.open(destination) as result:
                    deviation = ImageStat.Stat(result.convert('RGB')).stddev
                    actual_size = list(result.size)
                if max(deviation) < 1:
                    raise RuntimeError(f'Generated image appears blank: {destination}')
                record = {'image': str(destination), 'prompt_id': prompt_id, 'seconds': round(time.monotonic() - started, 2),
                          'width': args.width, 'height': args.height, 'actual_size': actual_size, 'steps': args.steps,
                          'seed': args.seed, 'sampler': 'dpmpp_2m', 'scheduler': 'karras', 'cfg': 7.0,
                          'peak_total_gpu_used_mib': peak, 'observed_compute_pids': sorted(observed_pids),
                          'pixel_stddev': deviation, 'prompt': args.prompt,
                          'server_argv': system.get('argv'), 'pytorch_version': system.get('pytorch_version')}
                destination.with_suffix('.api.json').write_text(json.dumps(graph, indent=2), encoding='utf-8')
                destination.with_suffix('.result.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
                print(json.dumps(record, indent=2), flush=True)  # noqa: T201
                return
        time.sleep(3)
    running = api(args.server, '/queue').get('queue_running', [])
    if any(item[1] == prompt_id for item in running):
        api(args.server, '/interrupt', {})
    else:
        api(args.server, '/queue', {'delete': [prompt_id]})
    raise TimeoutError(f'Render exceeded {args.timeout}s; requested cancellation of {prompt_id}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default='http://127.0.0.1:8188')
    parser.add_argument('--prompt', default='an original cheerful orange fox wearing a teal scarf, sitting in a green meadow with small flowers, clean storybook illustration, warm golden sunlight, soft colors, full body, detailed fur, balanced composition')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--timeout', type=int, default=600)
    generate(parser.parse_args())
