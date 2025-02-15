# -*- coding: utf-8 -*-
"""
SexyBot - AI-powered intimate companion with Lovense integration
Created by @itsme228

This is an advanced AI chatbot that combines natural language processing,
voice interaction, and Lovense device control for intimate experiences.

Features:
- Voice interaction with AI using Gemini Pro
- Real-time Lovense device control
- Multi-modal input support (voice, text, camera, screen)
- Edge TTS voice synthesis
- Dialog history tracking
- Automatic silence detection
- Cross-platform support

## Setup

To install the dependencies for this script, run:

pip install google-genai opencv-python pyaudio pillow mss gTTS pygame edge-tts numpy

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the API key you obtained from Google AI Studio.

## Run

To run the script:

python sexy_bot.py --mode [camera|screen|none]
"""

import asyncio
import base64
import io
import os
import sys
import traceback
import aiohttp
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import tempfile
import argparse
import ssl

import cv2
import pyaudio
import PIL.Image
import mss
import numpy as np
import pygame
import edge_tts
import websockets

from google import genai
from google.genai import types

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# Константы
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-exp"
DEFAULT_MODE = "none"

try:
    client = genai.Client(http_options={"api_version": "v1alpha"})
except Exception as e:
    logging.error(f"Failed to initialize Gemini client: {str(e)}")
    raise

CONFIG = {
    "generation_config": {
        "response_modalities": ["TEXT"],
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {"voice_name": "Aoede"}
            }
        },
        "candidate_count": 1,
        "temperature": 0.9,
        "max_output_tokens": 3000,
        "streaming": True,
        "top_p": 0.9,
        "top_k": 40
    },
    "system_instruction": {
        "parts": [{"text": ""}]
    }
}

pya = pyaudio.PyAudio()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    encoding='utf-8',
    handlers=[
        logging.FileHandler('lovense_commands.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logging.info("=== System Configuration ===")
logging.info(f"Model: {MODEL}")
logging.info("System Instruction:")
logging.info(CONFIG["system_instruction"]["parts"][0]["text"])


def print_audio_devices():
    """Вывод списка аудио устройств в системе"""
    print("\n=== Аудио устройства в системе ===")
    input_devices, output_devices = [], []
    for i in range(pya.get_device_count()):
        try:
            device_info = pya.get_device_info_by_index(i)
            device = {
                'index': int(device_info['index']),
                'name': device_info['name'],
                'rate': int(device_info['defaultSampleRate']),
                'input_channels': int(device_info['maxInputChannels']),
                'output_channels': int(device_info['maxOutputChannels'])
            }
            if device['input_channels'] > 0:
                input_devices.append(device)
            if device['output_channels'] > 0:
                output_devices.append(device)
        except Exception as e:
            print(f"\nОшибка получения информации об устройстве {i}: {e}")
    try:
        default_input_index = int(pya.get_default_input_device_info()['index'])
    except Exception as e:
        print("\nНе удалось получить устройство ввода по умолчанию:", e)
        default_input_index = None
    try:
        default_output_index = int(pya.get_default_output_device_info()['index'])
    except Exception as e:
        print("\nНе удалось получить устройство вывода по умолчанию:", e)
        default_output_index = None

    print("\nУстройства ввода (микрофоны):")
    print("-----------------------------")
    for device in input_devices:
        default_mark = "[По умолчанию] " if device['index'] == default_input_index else ""
        print(f"{default_mark}Устройство {device['index']}:")
        print(f"  Название: {device['name']}")
        print(f"  Частота: {device['rate']} Hz")
        print(f"  Каналов: {device['input_channels']}\n")

    print("\nУстройства вывода (динамики):")
    print("-----------------------------")
    for device in output_devices:
        default_mark = "[По умолчанию] " if device['index'] == default_output_index else ""
        print(f"{default_mark}Устройство {device['index']}:")
        print(f"  Название: {device['name']}")
        print(f"  Частота: {device['rate']} Hz")
        print(f"  Каналов: {device['output_channels']}\n")


class CommandQueue:
    """Менеджер очередей команд для каждой игрушки"""
    def __init__(self):
        self.queues: Dict[str, List[dict]] = {}
        self.current_tasks: Dict[str, asyncio.Task] = {}

    def set_sequence(self, toy_id: str, commands: List[dict]):
        self.queues[toy_id] = commands
        if toy_id in self.current_tasks and not self.current_tasks[toy_id].done():
            self.current_tasks[toy_id].cancel()

    def get_next_command(self, toy_id: str) -> Optional[dict]:
        return self.queues.get(toy_id, []).pop(0) if self.queues.get(toy_id) else None


class LovenseDevice:
    """Управление устройствами Lovense"""
    def __init__(self):
        self.devices: Dict = {}
        self.active_device = None
        self.base_url = None
        self.port = None
        self.http_session = None
        self.available_functions: Dict[str, dict] = {}
        self.command_queue = CommandQueue()
        self.running_sequences: Dict[str, asyncio.Task] = {}

    def parse_sequence(self, sequence_str: str) -> List[dict]:
        commands = []
        for cmd in sequence_str.split(';'):
            if not cmd:
                continue
            parts = cmd.split(':')
            if len(parts) != 3:
                logging.error(f"Invalid command format: {cmd}")
                continue
            cmd_type, intensity, switch_time = parts
            if cmd_type.lower() != 'v':
                logging.error(f"Unknown command type: {cmd_type}")
                continue
            try:
                intensity = int(intensity)
                switch_time = float(switch_time)
            except ValueError as e:
                logging.error(f"Invalid number format in command {cmd}: {e}")
                continue
            if not (0 <= intensity <= 20):
                logging.error(f"Invalid intensity value {intensity} in command {cmd}")
                continue
            if switch_time <= 0:
                logging.error(f"Invalid switch time {switch_time} in command {cmd}")
                continue
            commands.append({
                "command": "Function",
                "action": f"Vibrate:{intensity}",
                "timeSec": 10,
                "switch_after": switch_time
            })
        return commands

    def _get_toy_capabilities(self, toy_name: str) -> dict:
        capabilities = {
            'lush': {'Vibrate': (0, 20)},
            'ferri': {'Vibrate': (0, 20)},
            'edge': {'Vibrate1': (0, 20), 'Vibrate2': (0, 20)},
            'hush': {'Vibrate': (0, 20)},
            'ambi': {'Vibrate': (0, 20)},
            'domi': {'Vibrate': (0, 20)},
            'osci': {'Vibrate': (0, 20), 'Oscillate': (0, 20)},
            'max': {'Vibrate': (0, 20), 'Air': (0, 3)},
            'nora': {'Vibrate': (0, 20), 'Rotate': (0, 20)},
            'diamo': {'Vibrate': (0, 20)},
            'calor': {'Vibrate': (0, 20), 'Air': (0, 3)},
            'dolce': {'Vibrate': (0, 20)},
            'flexer': {'Vibrate': (0, 20), 'Bend': (0, 20)},
            'gush': {'Vibrate': (0, 20), 'Air': (0, 3)},
            'hyphy': {'Vibrate': (0, 20)},
            'exomoon': {'Vibrate': (0, 20)},
            'gravity': {'Vibrate': (0, 20), 'Air': (0, 3)},
            'gemini': {'Vibrate': (0, 20)},
            'vulse': {'Vibrate': (0, 20), 'Pulse': (0, 20)},
            'ridge': {'Vibrate': (0, 20), 'Air': (0, 3)}
        }
        return capabilities.get(toy_name.lower(), {'Vibrate': (0, 20)})

    def _generate_system_instruction(self) -> str:
        instruction = (
            "You are an adaptive and sensitive AI companion for professional intimate play. "
            "You can read the mood and adapt to partner's needs while controlling Lovense devices with precision and care.\n"
            "Always respond in Russian language.\n"
            "Pay close attention to the history of how the user has responded to past patterns.\n"
            "In \"user_intent\" you should describe what user wants/desires.\n"
            "If user silent, repeat the last pattern.\n"
            "IMPORTANT: Return ONLY pure JSON without any markdown formatting or code blocks.\n\n"
            "Your responses should be in the following format:\n"
            "{\n"
            '    "response": "Your flirty/sexual response text here",\n'
            '    "toys": [\n'
            "        {\n"
            '            "toy": "device_id1",\n'
            '            "sequence": "v:10:5;v:4:2;v:20:0.1;v:1:10.2"\n'
            "        }\n"
            "    ],\n"
            '    "user_intent": "What user wants/desires"\n'
            "}\n\n"
            "PATTERN GUIDELINES: ... (detailed instructions omitted for brevity)\n"
        )
        if self.devices:
            instruction += "\nConnected devices and their capabilities:\n"
            for ip, info in self.devices.items():
                for toy_id, toy in info['toys'].items():
                    instruction += f"\nDevice ID: {toy_id}"
                    instruction += f"\nName: {toy['name']}"
                    instruction += f"\nNickname: {toy.get('nickName', 'Unknown')}"
                    instruction += f"\nBattery: {toy['battery']}%"
                    instruction += "\nAvailable functions:"
                    caps = self._get_toy_capabilities(toy['name'])
                    for func, (min_val, max_val) in caps.items():
                        instruction += f"\n- {func}: {min_val}-{max_val}"
                    instruction += "\n"
        return instruction

    async def discover_devices(self) -> bool:
        logging.info("Searching for Lovense devices...")
        if not self.http_session:
            timeout = aiohttp.ClientTimeout(total=10)
            connector = aiohttp.TCPConnector(ssl=False, limit=10)
            self.http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            logging.info("HTTP session reinitialized in discover_devices")
        url = 'https://api.lovense.com/api/lan/getToys'
        logging.info(f"Making GET request to: {url}")
        try:
            async with self.http_session.get(url) as response:
                logging.info(f"Response status: {response.status}")
                response_text = await response.text()
                logging.info(f"Raw response: {response_text}")
                data = json.loads(response_text)
                logging.info(f"Parsed device data: {json.dumps(data, indent=2)}")
                if not isinstance(data, dict):
                    logging.error(f"Unexpected response format: {type(data)}")
                    return False
                self.devices.clear()
                for domain, device_info in data.items():
                    try:
                        ip = domain.split('.')[0].replace('-', '.')
                        if not isinstance(device_info, dict):
                            logging.error(f"Invalid device info for {domain}")
                            continue
                        toys = device_info.get('toys', {})
                        if not toys and 'toyJson' in device_info:
                            try:
                                toys = json.loads(device_info['toyJson'])
                            except json.JSONDecodeError:
                                logging.error(f"Failed to parse toyJson for {domain}")
                                toys = {}
                        self.devices[ip] = {
                            'https_port': device_info.get('httpsPort'),
                            'toys': toys
                        }
                        for toy_id, toy in toys.items():
                            self.available_functions[toy_id] = self._get_toy_capabilities(toy['name'])
                        if not self.active_device:
                            self.active_device = ip
                            self.base_url = f"https://{ip}"
                            self.port = device_info.get('httpsPort')
                    except Exception as e:
                        logging.error(f"Error processing device {domain}: {e}")
                        continue
                if self.devices:
                    logging.info("=== Found Lovense Devices ===")
                    for ip, info in self.devices.items():
                        logging.info(f"Device at {ip}, Port: {info['https_port']}")
                        for toy_id, toy in info['toys'].items():
                            logging.info(f"Toy ID: {toy_id}, Name: {toy.get('name', 'Unknown')}, "
                                         f"Nickname: {toy.get('nickName', 'Unknown')}, "
                                         f"Work Mode: {toy.get('workMode', 'Unknown')}, "
                                         f"Battery: {toy.get('battery', 'Unknown')}%, "
                                         f"Status: {'Connected' if toy.get('status') == 1 else 'Disconnected'}")
                    CONFIG["system_instruction"]["parts"][0]["text"] = self._generate_system_instruction()
                    return True
                else:
                    logging.warning("No Lovense devices found")
                    return False
        except aiohttp.ClientError as e:
            logging.error(f"HTTP request failed: {e}")
            return False
        except Exception as e:
            logging.error(f"Error discovering devices: {e}\n{traceback.format_exc()}")
            return False

    def _format_command_output(self, command: dict, result: Optional[dict] = None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        sep = "╭" + "─" * 78 + "╮"
        mid_sep = "├" + "─" * 78 + "┤"
        end_sep = "╰" + "─" * 78 + "╯"
        print(f"\n\033[93m{sep}\033[0m")
        print(f"\033[93m│ [COMMAND {timestamp}]" + " " * (65) + "│\033[0m")
        details = [
            f"Toy ID: {command.get('toy', 'Default')}",
            f"Function: {command.get('command', 'Unknown')}",
            f"Action: {command.get('action', 'Unknown')}",
            f"Time: {command.get('timeSec', 'Unknown')}s"
        ]
        for detail in details:
            print(f"\033[93m│ {detail}" + " " * (77 - len(detail)) + "│\033[0m")
        if command.get('loopRunningSec'):
            for ld in [f"Loop Running: {command['loopRunningSec']}s", f"Loop Pause: {command.get('loopPauseSec', 0)}s"]:
                print(f"\033[93m│ {ld}" + " " * (77 - len(ld)) + "│\033[0m")
        print(f"\033[93m{mid_sep}\033[0m")
        if result:
            status = f"Status: {'Success' if result.get('code') == 200 else 'Failed'}"
            code = f"Code: {result.get('code', 'Unknown')}"
            print(f"\033[93m│ {status}" + " " * (77 - len(status)) + "│\033[0m")
            print(f"\033[93m│ {code}" + " " * (77 - len(code)) + "│\033[0m")
        print(f"\033[93m{end_sep}\033[0m\n")

    async def send_command(self, command: dict) -> bool:
        if not self.active_device:
            logging.error("No active device available")
            return False
        try:
            toy_id = command.get('toy')
            if not toy_id:
                logging.error("No toy ID specified")
                return False
            target_device, target_port = None, None
            for ip, info in self.devices.items():
                if toy_id in info['toys']:
                    target_device = ip
                    target_port = info['https_port']
                    break
            if not target_device:
                logging.error(f"Toy {toy_id} not found")
                return False
            if toy_id not in self.available_functions:
                logging.error(f"No capability info for toy {toy_id}")
                return False
            action_parts = command.get('action', '').split(':')
            action = action_parts[0]
            intensity = int(action_parts[1]) if len(action_parts) > 1 else 0
            valid_actions = self.available_functions[toy_id].keys()
            if action not in valid_actions and action != 'Vibrate:0':
                logging.error(f"Action {action} not available for toy {toy_id}")
                return False
            if action != 'Vibrate:0':
                min_val, max_val = self.available_functions[toy_id][action]
                if not (min_val <= intensity <= max_val):
                    logging.error(f"Intensity {intensity} out of range for {action}")
                    return False
            if 'timeSec' not in command:
                command['timeSec'] = 1
            url = f"https://{target_device}:{target_port}/command"
            self._format_command_output(command)
            async with self.http_session.post(url, json=command) as response:
                response_text = await response.text()
                result = json.loads(response_text)
                self._format_command_output(command, result)
                return result.get('code') == 200
        except aiohttp.ClientOSError as e:
            if "APPLICATION_DATA_AFTER_CLOSE_NOTIFY" in str(e):
                logging.warning("SSL error after close notify encountered; ignoring during reconnection.")
                return False
            else:
                logging.error(f"HTTP ClientOSError in send_command: {e}\n{traceback.format_exc()}")
                return False
        except ssl.SSLError as e:
            if "APPLICATION_DATA_AFTER_CLOSE_NOTIFY" in str(e):
                logging.warning("SSL error after close notify encountered; ignoring during reconnection.")
                return False
            else:
                logging.error(f"SSL Error in send_command: {e}\n{traceback.format_exc()}")
                return False
        except Exception as e:
            logging.error(f"Error sending command: {e}\n{traceback.format_exc()}")
            return False

    async def process_command_sequence(self, toy_id: str):
        try:
            if toy_id in self.running_sequences:
                self.running_sequences[toy_id].cancel()
            current_task = asyncio.current_task()
            self.running_sequences[toy_id] = current_task
            while (command := self.command_queue.get_next_command(toy_id)) is not None:
                cmds = self.parse_sequence(command['sequence']) if isinstance(command.get('sequence'), str) else [command]
                for cmd in cmds:
                    cmd['toy'] = toy_id
                    start = asyncio.get_event_loop().time()
                    await self.send_command(cmd)
                    switch_time = cmd.get('switch_after', cmd.get('timeSec', 1))
                    elapsed = asyncio.get_event_loop().time() - start
                    await asyncio.sleep(max(0, switch_time - elapsed))
        except asyncio.CancelledError:
            logging.info(f"Command sequence cancelled for toy {toy_id}")
        except Exception as e:
            logging.error(f"Error in command sequence for {toy_id}: {e}")
        finally:
            self.running_sequences.pop(toy_id, None)

    async def set_toy_sequence(self, toy_id: str, commands: List[dict]):
        for command in commands:
            command['toy'] = toy_id
        self.command_queue.set_sequence(toy_id, commands)
        task = asyncio.create_task(self.process_command_sequence(toy_id))
        self.command_queue.current_tasks[toy_id] = task


def create_audio_stream(input_device_index: int, rate: int, is_input: bool = True):
    try:
        stream = pya.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=rate,
            input=is_input,
            output=not is_input,
            input_device_index=input_device_index if is_input else None,
            output_device_index=input_device_index if not is_input else None,
            frames_per_buffer=CHUNK_SIZE
        )
        stream.start_stream()
        logging.info(f"Audio stream started for device index {input_device_index}")
        return stream
    except Exception as e:
        logging.error(f"Error creating audio stream on device {input_device_index}: {e}")
        return None


def close_stream(stream):
    if stream:
        try:
            stream.stop_stream()
            stream.close()
        except Exception as e:
            logging.error(f"Error closing stream: {e}")


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        self.audio_in_queue: Optional[asyncio.Queue] = None
        self.out_queue: Optional[asyncio.Queue] = None
        self.message_queue: Optional[asyncio.Queue] = None
        self.tts_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self.session = None
        self.lovense_device = LovenseDevice()
        self.audio_stream = None
        self.http_session = None
        self.last_voice_activity = datetime.now()
        self.last_silence_check = datetime.now()
        self.SILENCE_THRESHOLD = 40
        self.MIN_VOICE_AMPLITUDE = 300
        self.session_start_time = None
        self.SESSION_LIMIT = 540
        self.reconnection_event = asyncio.Event()
        self.dialog_history = []
        self.history_file = "dialog_history.json"
        self.history_size = 0
        self.mic_enabled = True
        self.mic_lock = asyncio.Lock()
        self.speaking_event = asyncio.Event()
        self.load_dialog_history()

    def load_dialog_history(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.dialog_history = json.load(f)
                logging.info(f"Loaded {len(self.dialog_history)} AI responses from history")
        except Exception as e:
            logging.error(f"Error loading dialog history: {e}")
            self.dialog_history = []

    def save_dialog_history(self):
        try:
            if self.history_size > 0 and len(self.dialog_history) > self.history_size:
                self.dialog_history = self.dialog_history[-self.history_size:]
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.dialog_history, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved {len(self.dialog_history)} AI responses to history")
        except Exception as e:
            logging.error(f"Error saving dialog history: {e}")

    def add_to_history(self, response_json: dict):
        try:
            if "response" in response_json:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "response": response_json["response"],
                }
                if "user_intent" in response_json:
                    entry["user_intent"] = response_json["user_intent"]
                if "toys" in response_json:
                    entry["patterns"] = [{"toy": td["toy"], "sequence": td["sequence"]} for td in response_json["toys"]]
                self.dialog_history.append(entry)
                self.save_dialog_history()
        except Exception as e:
            logging.error(f"Error adding to history: {e}")

    def get_history_context(self) -> str:
        try:
            if not self.dialog_history:
                return ""
            context = "Previous conversation history:\n\n"
            entries = self.dialog_history if self.history_size == 0 else self.dialog_history[-self.history_size:]
            for entry in entries:
                context += f"Time: {entry.get('timestamp', '')}\n"
                if entry.get("user_intent"):
                    context += f"User intent: {entry['user_intent']}\n"
                if entry.get("response"):
                    context += f"AI: {entry['response']}\n"
                if entry.get("patterns"):
                    context += "Used patterns:\n"
                    for pat in entry["patterns"]:
                        context += f"Toy {pat.get('toy', 'unknown')}:\n{pat.get('sequence','')}\n"
                context += "\n"
            return context
        except Exception as e:
            logging.error(f"Error generating history context: {e}")
            return ""

    def get_device_info_prompt(self) -> str:
        if not self.lovense_device.devices:
            return ""
        info_list = []
        for ip, info in self.lovense_device.devices.items():
            for toy_id, toy in info['toys'].items():
                info_list.append(
                    f"Device ID: {toy_id}\nType: {toy['name']}\nNickname: {toy.get('nickName', 'Unknown')}\n"
                    f"Work Mode: {toy['workMode']}\nBattery: {toy['battery']}%\n"
                    f"Status: {'Connected' if toy['status'] == 1 else 'Disconnected'}\n"
                )
        return "\n".join(info_list)

    async def send_text(self):
        while True:
            try:
                if not self.session:
                    await asyncio.sleep(0.1)
                    continue
                text = await asyncio.to_thread(input, "message > ")
                if not text:
                    continue
                if text.lower() == "q":
                    logging.info("User requested quit")
                    self.reconnection_event.set()
                    break
                try:
                    logging.info(f"Sending text: {text}")
                    await self.session.send(input=text, end_of_turn=True)
                    self.last_voice_activity = datetime.now()
                except websockets.exceptions.ConnectionClosedError as e:
                    logging.warning("Connection closed in text input, triggering reconnection...")
                    self.reconnection_event.set()
                    await asyncio.sleep(1)
                except Exception as e:
                    logging.error(f"Error sending text: {e}")
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in send_text loop: {e}")
                await asyncio.sleep(0.1)

    def _get_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        return {"mime_type": "image/jpeg", "data": base64.b64encode(image_io.read()).decode()}

    async def get_frames(self):
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]
        img = sct.grab(monitor)
        image = PIL.Image.frombytes("RGB", img.size, img.rgb)
        image_io = io.BytesIO()
        image.save(image_io, format="jpeg")
        image_io.seek(0)
        return {"mime_type": "image/jpeg", "data": base64.b64encode(image_io.read()).decode()}

    async def get_screen(self):
        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def initialize_http(self) -> bool:
        try:
            if not self.http_session:
                timeout = aiohttp.ClientTimeout(total=10)
                connector = aiohttp.TCPConnector(ssl=False, limit=10)
                self.http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
                self.lovense_device.http_session = self.http_session
                logging.info("HTTP session initialized")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize HTTP session: {e}")
            return False

    async def cleanup_http(self):
        if self.http_session:
            try:
                if not self.http_session.closed:
                    await self.http_session.close()
                    logging.info("HTTP session closed successfully")
            except Exception as e:
                logging.error(f"Error closing HTTP session: {e}")
            finally:
                self.http_session = None
                self.lovense_device.http_session = None
                logging.info("HTTP session references cleared")

    async def process_messages(self):
        while True:
            message = await self.message_queue.get()
            try:
                timestamp = datetime.now().strftime("%H:%M:%S")
                sep = "╭" + "─" * 78 + "╮"
                end_sep = "╰" + "─" * 78 + "╯"
                lines = []
                words = message.split()
                current_line = "│ "
                for word in words:
                    if len(current_line) + len(word) < 77:
                        current_line += word + " "
                    else:
                        lines.append(current_line.ljust(77) + "│")
                        current_line = "│ " + word + " "
                if current_line:
                    lines.append(current_line.ljust(77) + "│")
                print(f"\n\033[95m{sep}\033[0m")
                print(f"\033[94m│ [AI {timestamp}]" + " " * 69 + "│\033[0m")
                for l in lines:
                    print(f"\033[92m{l}\033[0m")
                print(f"\033[95m{end_sep}\033[0m\n")
            except Exception as e:
                logging.error(f"Error displaying message: {e}")
            finally:
                self.message_queue.task_done()

    async def check_silence(self):
        while True:
            try:
                if not self.session:
                    await asyncio.sleep(0.1)
                    continue
                now = datetime.now()
                silence_duration = (now - self.last_voice_activity).total_seconds()
                if silence_duration >= self.SILENCE_THRESHOLD and (now - self.last_silence_check).total_seconds() >= self.SILENCE_THRESHOLD:
                    logging.info(f"Silence detected for {silence_duration:.1f}s. Sending prompt.")
                    try:
                        prompt = "Пользователь молчит уже почти минуту. Проверь всё ли в порядке, может нужно его подбодрить или спросить о чём-то?"
                        await self.session.send(input=prompt, end_of_turn=True)
                        self.last_silence_check = now
                    except websockets.exceptions.ConnectionClosedError as e:
                        logging.warning("Connection closed in silence check, triggering reconnection...")
                        self.reconnection_event.set()
                        await asyncio.sleep(1)
                        continue
                await asyncio.sleep(5)
            except Exception as e:
                logging.error(f"Error in silence check: {e}")
                await asyncio.sleep(5)

    def find_bluetooth_mic(self):
        try:
            for i in range(pya.get_device_count()):
                try:
                    info = pya.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0 and 'bluez' in info['name'].lower():
                        logging.info(f"Found Bluetooth microphone: {info['name']}")
                        return i, int(info['defaultSampleRate'])
                except Exception as e:
                    logging.error(f"Error checking device {i}: {e}")
            return None, None
        except Exception as e:
            logging.error(f"Error finding Bluetooth mic: {e}")
            return None, None

    async def listen_audio(self):
        while True:
            try:
                if not self.session:
                    if self.audio_stream:
                        close_stream(self.audio_stream)
                    await asyncio.sleep(0.1)
                    continue
                if self.audio_stream:
                    close_stream(self.audio_stream)
                input_index = 1
                try:
                    info = pya.get_device_info_by_index(input_index)
                    if info['maxInputChannels'] == 0:
                        logging.error(f"Device {input_index} has no input channels")
                        await asyncio.sleep(1)
                        continue
                    logging.info(f"Using input device: {info['name']} with {info['maxInputChannels']} channels")
                except Exception as e:
                    logging.error(f"Error with input device {input_index}: {e}")
                    await asyncio.sleep(1)
                    continue
                self.audio_stream = await asyncio.to_thread(create_audio_stream, input_index, SEND_SAMPLE_RATE, True)
                if not self.audio_stream:
                    await asyncio.sleep(1)
                    continue
                self.audio_buffer = []
                kwargs = {"exception_on_overflow": False}
                while True:
                    if not self.session:
                        break
                    if not self.mic_enabled:
                        await self.speaking_event.wait()
                        continue
                    if not self.audio_stream.is_active():
                        logging.error("Audio stream inactive")
                        break
                    data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                    if not data:
                        logging.warning("No data from audio stream")
                        continue
                    amplitude = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
                    if amplitude > self.MIN_VOICE_AMPLITUDE:
                        self.last_voice_activity = datetime.now()
                    self.audio_buffer.append(data)
                    if len(self.audio_buffer) >= 5:
                        combined = b''.join(self.audio_buffer)
                        try:
                            if self.session:
                                await self.session.send(input={"data": combined, "mime_type": "audio/pcm"}, end_of_turn=True)
                                logging.debug("Audio data sent")
                        except websockets.exceptions.ConnectionClosedError as e:
                            logging.warning("Connection closed in audio stream, triggering reconnection...")
                            self.reconnection_event.set()
                            break
                        except Exception as e:
                            logging.error(f"Error sending audio: {e}")
                        self.audio_buffer.clear()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in listen_audio loop: {e}")
                if self.audio_stream:
                    close_stream(self.audio_stream)
                await asyncio.sleep(0.1)
        if self.audio_stream:
            close_stream(self.audio_stream)
        logging.info("Audio listening stopped")

    async def process_response_chunk(self, response_json: dict, start_time: datetime):
        try:
            if resp_text := response_json.get("response"):
                await self.message_queue.put(resp_text)
            if toys := response_json.get("toys"):
                toy_tasks = []
                for td in toys:
                    toy_id = td.get("toy")
                    sequence = td.get("sequence")
                    if toy_id and sequence:
                        task = asyncio.create_task(self.lovense_device.set_toy_sequence(toy_id, [{"sequence": sequence, "toy": toy_id}]))
                        toy_tasks.append(task)
                if toy_tasks:
                    await asyncio.gather(*toy_tasks)
            self.add_to_history(response_json)
        except Exception as e:
            logging.error(f"Error processing response chunk: {e}\n{traceback.format_exc()}")

    async def receive_audio(self):
        while True:
            try:
                start = datetime.now()
                turn = self.session.receive()
                complete_response = ""
                partial_response = ""
                current_response = {}
                text_spoken = False
                async for response in turn:
                    try:
                        if response.data:
                            await self.audio_in_queue.put(response.data)
                            continue
                        if response.text:
                            txt = response.text.replace("```", "").replace("json", "").strip()
                            if not txt:
                                continue
                            complete_response += txt
                            partial_response += txt
                            if not text_spoken and ('"response":' in partial_response or '"user_intent":' in partial_response):
                                try:
                                    if '"response":' in partial_response:
                                        idx = partial_response.find('"response":') + len('"response":')
                                        start_q = partial_response.find('"', idx) + 1
                                        end_q = partial_response.find('"', start_q)
                                        if end_q > start_q:
                                            current_response["response"] = partial_response[start_q:end_q]
                                    if '"user_intent":' in partial_response:
                                        idx = partial_response.find('"user_intent":') + len('"user_intent":')
                                        start_q = partial_response.find('"', idx) + 1
                                        end_q = partial_response.find('"', start_q)
                                        if end_q > start_q:
                                            current_response["user_intent"] = partial_response[start_q:end_q]
                                    if current_response.get("response"):
                                        await self.speak_text(current_response["response"])
                                        text_spoken = True
                                        await self.process_response_chunk(current_response, start)
                                except Exception as e:
                                    logging.debug(f"Early extraction failed: {e}")
                            if partial_response.count("{") == partial_response.count("}") and partial_response:
                                try:
                                    response_json = json.loads(partial_response)
                                    current_response.update(response_json)
                                    await self.process_response_chunk(current_response, start)
                                    partial_response = ""
                                except json.JSONDecodeError as e:
                                    logging.debug(f"Incomplete JSON: {e}")
                    except Exception as e:
                        logging.error(f"Error processing received response: {e}")
                logging.debug(f"Response processing took: {datetime.now()-start}")
            except Exception as e:
                logging.error(f"Error in receive_audio: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(0.1)

    async def play_audio(self):
        output_stream = None
        while True:
            try:
                if output_stream:
                    close_stream(output_stream)
                    output_stream = None
                output_device = None
                for i in range(pya.get_device_count()):
                    try:
                        info = pya.get_device_info_by_index(i)
                        if info['maxOutputChannels'] > 0 and i == int(pya.get_default_output_device_info()['index']):
                            output_device = info
                            break
                    except Exception:
                        continue
                if not output_device:
                    logging.error("Не найдено устройство вывода")
                    await asyncio.sleep(1)
                    continue
                output_stream = await asyncio.to_thread(create_audio_stream, int(output_device['index']), int(output_device['defaultSampleRate']), False)
                if not output_stream:
                    await asyncio.sleep(1)
                    continue
                while True:
                    if not output_stream.is_active():
                        logging.error("Output stream inactive")
                        break
                    bytestream = await self.audio_in_queue.get()
                    await asyncio.to_thread(output_stream.write, bytestream)
            except Exception as e:
                logging.error(f"Error in play_audio: {e}")
                if output_stream:
                    close_stream(output_stream)
                    output_stream = None
                await asyncio.sleep(1)

    async def process_tts(self):
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        VOICE = "ru-RU-SvetlanaNeural"
        RATE = "+0%"
        VOLUME = "+0%"
        while True:
            try:
                text = await self.tts_queue.get()
                temp_dir = tempfile.gettempdir()
                os.makedirs(temp_dir, exist_ok=True)
                temp_file = os.path.join(temp_dir, f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
                try:
                    communicate = edge_tts.Communicate(text, VOICE, rate=RATE, volume=VOLUME)
                    await communicate.save(temp_file)
                    if os.path.exists(temp_file):
                        await asyncio.to_thread(pygame.mixer.music.load, temp_file)
                        await asyncio.to_thread(pygame.mixer.music.play)
                        while pygame.mixer.music.get_busy():
                            await asyncio.sleep(0.1)
                        pygame.mixer.music.unload()
                    else:
                        logging.error(f"Audio file not found: {temp_file}")
                finally:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception as e:
                        logging.warning(f"Failed to remove temp file: {e}")
            except Exception as e:
                logging.error(f"TTS processing error: {e}\n{traceback.format_exc()}")
            finally:
                self.tts_queue.task_done()

    async def speak_text(self, text: str):
        try:
            async with self.mic_lock:
                self.mic_enabled = False
                self.speaking_event.clear()
                logging.info("Микрофон отключен для воспроизведения")
                try:
                    await self.tts_queue.put(text)
                    await self.tts_queue.join()
                    self.last_voice_activity = datetime.now()
                    logging.info("Voice activity updated after TTS")
                finally:
                    self.mic_enabled = True
                    self.speaking_event.set()
                    logging.info("Микрофон снова включен")
        except Exception as e:
            logging.error(f"Error in speak_text: {e}")
            self.mic_enabled = True
            self.speaking_event.set()

    async def check_session_time(self):
        while True:
            try:
                if self.session_start_time:
                    duration = (datetime.now() - self.session_start_time).total_seconds()
                    if duration >= self.SESSION_LIMIT:
                        logging.info("Session time limit reached, triggering reconnection...")
                        self.reconnection_event.set()
                await asyncio.sleep(10)
            except Exception as e:
                logging.error(f"Error in session time check: {e}")
                await asyncio.sleep(1)

    async def _cleanup_queues(self):
        for attr in ['audio_in_queue', 'out_queue', 'message_queue', 'tts_queue', 'response_queue']:
            q = getattr(self, attr, None)
            if q:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                setattr(self, attr, None)
        logging.info("Queues cleaned up successfully")

    async def _initialize_queues(self):
        await self._cleanup_queues()
        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=5)
        self.message_queue = asyncio.Queue()
        self.tts_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        logging.info("Queues initialized successfully")

    async def _cleanup_session(self):
        try:
            logging.info("Starting session cleanup...")
            self.mic_enabled = True
            self.speaking_event.set()
            if self.audio_stream:
                close_stream(self.audio_stream)
                self.audio_stream = None
                logging.info("Audio stream closed")
            self.session = None
            await self._cleanup_queues()
            logging.info("Session cleanup completed")
        except Exception as e:
            logging.error(f"Error during session cleanup: {e}")
            self.mic_enabled = True
            self.speaking_event.set()

    async def _cleanup_all(self):
        try:
            logging.info("Starting complete cleanup...")
            if self.lovense_device:
                for toy_id in list(self.lovense_device.running_sequences.keys()):
                    try:
                        await self.lovense_device.send_command({
                            'toy': toy_id,
                            'command': 'Function',
                            'action': 'Vibrate:0',
                            'timeSec': 1
                        })
                    except Exception as e:
                        logging.warning(f"Error stopping toy {toy_id}: {e}")
            await self._cleanup_session()
            await self.cleanup_http()
            if self.audio_stream:
                close_stream(self.audio_stream)
                self.audio_stream = None
            self.reconnection_event.clear()
            self.speaking_event.clear()
            self.mic_enabled = True
            logging.info("Complete cleanup finished")
        except Exception as e:
            logging.error(f"Error in complete cleanup: {e}")
            raise

    async def run(self):
        while True:
            try:
                for _ in range(3):
                    if await self.initialize_http():
                        break
                    logging.warning("Retrying HTTP session initialization...")
                    await asyncio.sleep(1)
                if not self.http_session:
                    logging.error("HTTP session initialization failed after retries")
                    await asyncio.sleep(5)
                    continue

                while True:
                    try:
                        logging.info("Starting new session initialization...")
                        await self._cleanup_all()
                        await asyncio.sleep(1)
                        if not self.http_session and not await self.initialize_http():
                            logging.error("Failed to reinitialize HTTP session")
                            await asyncio.sleep(1)
                            continue
                        self.reconnection_event.clear()
                        self.speaking_event.clear()
                        self.mic_enabled = True
                        device_found = False
                        for _ in range(3):
                            if await self.lovense_device.discover_devices():
                                device_found = True
                                break
                            logging.warning("Device discovery failed, retrying...")
                            await asyncio.sleep(1)
                        if not device_found:
                            logging.warning("No Lovense devices found after retries")
                        history_context = self.get_history_context()
                        sys_instr = self.lovense_device._generate_system_instruction()
                        if history_context:
                            sys_instr = history_context + "\n\n" + sys_instr
                        CONFIG["system_instruction"]["parts"][0]["text"] = sys_instr
                        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                            self.session = session
                            self.session_start_time = datetime.now()
                            logging.info("New session started")
                            await self._initialize_queues()
                            self.mic_enabled = True
                            self.speaking_event.set()
                            self.last_voice_activity = datetime.now()
                            self.last_silence_check = datetime.now()
                            active_tasks = set()
                            async with asyncio.TaskGroup() as tg:
                                tasks = [
                                    tg.create_task(self.send_realtime()),
                                    tg.create_task(self.listen_audio()),
                                    tg.create_task(self.process_messages()),
                                    tg.create_task(self.process_tts()),
                                    tg.create_task(self.check_silence()),
                                    tg.create_task(self.receive_audio()),
                                    tg.create_task(self.play_audio()),
                                    tg.create_task(self.check_session_time()),
                                    tg.create_task(self.send_text())
                                ]
                                active_tasks.update(tasks)
                                if self.video_mode == "camera":
                                    active_tasks.add(tg.create_task(self.get_frames()))
                                elif self.video_mode == "screen":
                                    active_tasks.add(tg.create_task(self.get_screen()))
                                recon_task = tg.create_task(self.reconnection_event.wait())
                                active_tasks.add(recon_task)
                                await recon_task
                                logging.info("Reconnection event triggered, cancelling tasks...")
                                for task in active_tasks:
                                    if not task.done():
                                        task.cancel()
                                await asyncio.sleep(0.5)
                                logging.info("Tasks cancelled, preparing for reconnection...")
                                break
                    except Exception as e:
                        logging.error(f"Error in session loop: {e}")
                        await asyncio.sleep(1)
                        continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Critical error in run: {e}")
                await self._cleanup_all()
                await asyncio.sleep(1)
                continue
        await self._cleanup_all()
        logging.info("Bot shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="none", help="Video mode: camera, screen, or none")
    args = parser.parse_args()
    loop = asyncio.get_event_loop()
    audio_loop = AudioLoop(video_mode=args.mode)
    try:
        loop.run_until_complete(audio_loop.run())
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Exiting...")
    finally:
        loop.run_until_complete(audio_loop._cleanup_all())
        loop.close()

