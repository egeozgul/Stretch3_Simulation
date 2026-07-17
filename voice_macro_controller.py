#!/usr/bin/env python3
"""Voice or text command bridge for two Stretch macro controllers.

Examples:
  robot one get lettuce one
  robot two get tomato 3
  both robots get onion two
  robot 1 stay

The parser maps natural phrases to macro actions from actions.yaml and executes
them through the existing InteractiveController logic on /stretch and /stretch2.
"""

import argparse
from difflib import SequenceMatcher
import re
import threading
import time

import rclpy

from interactive_controller import InteractiveController


NUMBER_WORDS = {
    'one': '1',
    'two': '2',
    'three': '3',
    'first': '1',
    'second': '2',
    'third': '3',
}

OBJECT_WORDS = {
    'lettuce': 'lettuce',
    'let us': 'lettuce',
    'let is': 'lettuce',
    'let as': 'lettuce',
    'let this': 'lettuce',
    'led us': 'lettuce',
    'letas': 'lettuce',
    'lettice': 'lettuce',
    'lettis': 'lettuce',
    'lettuces': 'lettuce',
    'lattice': 'lettuce',
    'latest': 'lettuce',
    'tomato': 'tomato',
    'tomatoes': 'tomato',
    'tomoate': 'tomato',
    'tomate': 'tomato',
    'tomatoe': 'tomato',
    'onion': 'onion',
    'onions': 'onion',
    'union': 'onion',
    'unions': 'onion',
    'anyon': 'onion',
    'an yon': 'onion',
    'on yun': 'onion',
    'own yun': 'onion',
    'ownion': 'onion',
    'onien': 'onion',
    'onyon': 'onion',
    'anions': 'onion',
    'plate': 'plate',
}

OBJECT_FUZZY_WORDS = {
    'lettuce': ('lettuce', 'lettice', 'lettis', 'lattice', 'let us', 'let is', 'let as', 'led us'),
    'onion': ('onion', 'onions', 'union', 'unions', 'anyon', 'on yun', 'own yun', 'ownion', 'onien', 'onyon'),
}

OBJECT_FUZZY_THRESHOLD = {
    'lettuce': 0.70,
    'onion': 0.72,
}

GET_WORDS = ('get', 'grab', 'pick', 'pickup', 'take', 'fetch')

INDEX_WORDS = {
    '1': '1',
    'one': '1',
    'first': '1',
    '2': '2',
    'two': '2',
    'to': '2',
    'too': '2',
    'second': '2',
    '3': '3',
    'three': '3',
    'tree': '3',
    'free': '3',
    'third': '3',
}

ROBOT_ALIASES = {
    'robot one': ['/stretch'],
    'robot 1': ['/stretch'],
    'stretch one': ['/stretch'],
    'stretch 1': ['/stretch'],
    'first robot': ['/stretch'],
    'robot two': ['/stretch2'],
    'robot 2': ['/stretch2'],
    'robot2': ['/stretch2'],
    'robot too': ['/stretch2'],
    'robot to': ['/stretch2'],
    'stretch two': ['/stretch2'],
    'stretch 2': ['/stretch2'],
    'stretch2': ['/stretch2'],
    'stretch too': ['/stretch2'],
    'stretch to': ['/stretch2'],
    'second robot': ['/stretch2'],
    'both robots': ['/stretch', '/stretch2'],
    'both robot': ['/stretch', '/stretch2'],
    'both': ['/stretch', '/stretch2'],
    'all robots': ['/stretch', '/stretch2'],
}


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    for word, digit in NUMBER_WORDS.items():
        text = re.sub(rf'\b{word}\b', digit, text)
    return text


def compact_text(text):
    return text.replace(' ', '')


def fuzzy_object_match(phrase):
    """Map accent/transcription variants to known pickable objects."""
    if phrase in OBJECT_WORDS:
        return OBJECT_WORDS[phrase]

    compact_phrase = compact_text(phrase)
    if len(compact_phrase) < 4:
        return None

    best_object = None
    best_score = 0.0
    for object_name, variants in OBJECT_FUZZY_WORDS.items():
        for variant in variants:
            score = SequenceMatcher(None, compact_phrase, compact_text(variant)).ratio()
            if score > best_score:
                best_object = object_name
                best_score = score

    if best_object and best_score >= OBJECT_FUZZY_THRESHOLD[best_object]:
        return best_object
    return None


def extract_object_and_index(normalized):
    object_index = '1'
    index_pattern = r'(1|2|3|one|two|to|too|three|tree|free|first|second|third)'

    for word, canonical in sorted(OBJECT_WORDS.items(), key=lambda item: -len(item[0])):
        escaped_word = re.escape(word)
        match_after = re.search(
            rf'\b{escaped_word}\b\s*(?:number\s*)?(?:{index_pattern})?\b',
            normalized
        )
        match_before = re.search(
            rf'\b{index_pattern}\s+{escaped_word}\b',
            normalized
        )
        if match_after or match_before:
            if match_after and match_after.group(1):
                object_index = INDEX_WORDS.get(match_after.group(1), '1')
            elif match_before:
                object_index = INDEX_WORDS.get(match_before.group(1), '1')
            return canonical, object_index

    tokens = normalized.split()
    for size in (2, 1):
        for start in range(0, len(tokens) - size + 1):
            phrase_tokens = tokens[start:start + size]
            if any(token in INDEX_WORDS or token in GET_WORDS for token in phrase_tokens):
                continue

            object_name = fuzzy_object_match(' '.join(phrase_tokens))
            if not object_name:
                continue

            if start + size < len(tokens):
                next_token = tokens[start + size]
                object_index = INDEX_WORDS.get(next_token, object_index)
            if start > 0:
                previous_token = tokens[start - 1]
                object_index = INDEX_WORDS.get(previous_token, object_index)
            return object_name, object_index

    return None, object_index


def parse_voice_command(text):
    """Return (namespaces, macro_name) or (None, None)."""
    normalized = normalize_text(text)

    namespaces = None
    for phrase, robot_namespaces in sorted(ROBOT_ALIASES.items(), key=lambda item: -len(item[0])):
        normalized_phrase = normalize_text(phrase)
        if re.search(rf'\b{re.escape(normalized_phrase)}\b', normalized):
            namespaces = robot_namespaces
            normalized = re.sub(rf'\b{re.escape(normalized_phrase)}\b', ' ', normalized, count=1)
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            break

    # Remove any leftover robot selector tokens before extracting object index.
    normalized = re.sub(r'\b(robot|stretch)\s*(1|2|one|two|to|too)\b', ' ', normalized)
    normalized = re.sub(r'\b(1|2)\s*(robot|stretch)\b', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    if namespaces is None:
        namespaces = ['/stretch']

    if any(word in normalized.split() for word in ('stay', 'wait', 'stop', 'idle')):
        return namespaces, 'stay'

    if 'deliver' in normalized:
        return namespaces, 'deliver'
    if 'knife' in normalized:
        return namespaces, 'go_to_knife'
    if 'cut' in normalized:
        return namespaces, 'cut_fruit'
    if 'plate' in normalized and any(word in normalized.split() for word in ('get', 'grab', 'pick', 'pickup')):
        return namespaces, 'get_plate'
    if 'plate' in normalized:
        return namespaces, 'plate_fruit'

    object_name, object_index = extract_object_and_index(normalized)

    if object_name in ('lettuce', 'tomato', 'onion'):
        if any(word in normalized.split() for word in GET_WORDS):
            return namespaces, f'get_{object_name}{object_index}'

    return None, None


class VoiceMacroController:
    """Owns one InteractiveController per robot namespace."""

    def __init__(self, actions_file='actions.yaml'):
        self.controllers = {
            '/stretch': InteractiveController(actions_file=actions_file, robot_namespace='/stretch'),
            '/stretch2': InteractiveController(actions_file=actions_file, robot_namespace='/stretch2'),
        }

    def execute(self, namespaces, macro_name):
        unknown = [ns for ns in namespaces if ns not in self.controllers]
        if unknown:
            print(f"Unknown robot namespace(s): {unknown}")
            return False

        results = {}

        def run_one(ns):
            controller = self.controllers[ns]
            print(f"[{ns}] executing {macro_name}")
            results[ns] = controller._execute_macro_action(macro_name, {})

        threads = [threading.Thread(target=run_one, args=(ns,), daemon=True) for ns in namespaces]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        ok = all(results.get(ns, False) for ns in namespaces)
        status = 'OK' if ok else 'FAILED'
        print(f"{status}: {macro_name} on {', '.join(namespaces)}")
        return ok

    def destroy(self):
        for controller in self.controllers.values():
            controller.destroy_node()


def iter_text_commands():
    print("Text command mode. Type 'exit' to quit.")
    while True:
        try:
            text = input('voice> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.lower() in ('exit', 'quit'):
            return
        if text:
            yield text


def iter_voice_commands(phrase_time_limit=5.0):
    try:
        import speech_recognition as sr
    except ImportError as exc:
        raise RuntimeError(
            "speech_recognition is not installed. Install it or run with --text."
        ) from exc

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        print("Calibrating microphone...")
        recognizer.adjust_for_ambient_noise(source, duration=1.0)

    print("Voice command mode. Say commands like 'robot one get lettuce one'. Ctrl+C to quit.")
    while True:
        with microphone as source:
            print("Listening...")
            audio = recognizer.listen(source, phrase_time_limit=phrase_time_limit)
        try:
            text = recognizer.recognize_google(audio)
            yield text
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as exc:
            print(f"Speech recognition request failed: {exc}")
            time.sleep(1.0)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actions-file', default='actions.yaml')
    parser.add_argument('--text', action='store_true', help='Use typed text instead of microphone')
    parser.add_argument('--phrase-time-limit', type=float, default=5.0)
    parsed = parser.parse_args(args)

    rclpy.init()
    bridge = VoiceMacroController(actions_file=parsed.actions_file)
    command_iter = iter_text_commands() if parsed.text else iter_voice_commands(parsed.phrase_time_limit)

    try:
        for heard in command_iter:
            print(f"Heard: {heard}")
            namespaces, macro_name = parse_voice_command(heard)
            if not namespaces or not macro_name:
                print("I could not map that to a macro action.")
                continue
            print(f"Parsed -> robots={namespaces}, macro={macro_name}")
            bridge.execute(namespaces, macro_name)
    finally:
        bridge.destroy()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
