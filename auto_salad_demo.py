#!/usr/bin/env python3
"""Automated salad demo.

Run for stretch1 (default):
    python3 auto_salad_demo.py

Run for stretch2 only:
    python3 auto_salad_demo.py --stretch2
"""

import argparse
import time
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor

from interactive_controller import InteractiveController

STRETCH1_SEQUENCE = [
    'get_tomato1',       'cut_tomato1',       'get_chopped_tomato1',  'plate_tomato1',
    'get_lettuce1',      'cut_lettuce1',      'get_chopped_lettuce1', 'plate_lettuce1',
    'get_onion1',        'cut_onion1',        'get_chopped_onion1',   'plate_onion1',
]

STRETCH2_SEQUENCE = [
    'r2_get_tomato1',    'r2_cut_tomato1',    'r2_get_chopped_tomato1',  'r2_plate_tomato1',
    'r2_get_lettuce1',   'r2_cut_lettuce1',   'r2_get_chopped_lettuce1', 'r2_plate_lettuce1',
    'r2_get_onion1',     'r2_cut_onion1',     'r2_get_chopped_onion1',   'r2_plate_onion1',
]


def run_sequence(controller: InteractiveController, sequence: list, label: str) -> bool:
    print(f'\n{"="*60}')
    print(f'  {label}: starting salad sequence ({len(sequence)} macros)')
    print(f'{"="*60}')
    for macro in sequence:
        print(f'\n[{label}] >>> {macro}')
        ok = controller._execute_macro_action(macro, {})
        if not ok:
            print(f'[{label}] FAILED at "{macro}" — aborting sequence.')
            return False
        print(f'[{label}] waiting 50 s before next macro…')
        time.sleep(50.0)
    print(f'\n[{label}] ✓ All macros complete!')
    return True


def main():
    parser = argparse.ArgumentParser(description='Automated salad demo')
    parser.add_argument('--stretch2', action='store_true',
                        help='Run stretch2 sequence instead of stretch1')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    if args.stretch2:
        namespace = '/stretch2'
        sequence = STRETCH2_SEQUENCE
        controller = InteractiveController(robot_namespace=namespace, show_welcome=False)
    else:
        namespace = '/stretch'
        sequence = STRETCH1_SEQUENCE
        controller = InteractiveController(robot_namespace=namespace, show_welcome=False)

    executor = MultiThreadedExecutor()
    executor.add_node(controller)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        run_sequence(controller, sequence, namespace)
        print(f'\nDemo complete — {namespace} finished the salad sequence.')
    except KeyboardInterrupt:
        print('\nInterrupted by user.')
    finally:
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
