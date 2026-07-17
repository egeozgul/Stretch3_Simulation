"""
Script to view statistics of logged episodes.
"""

import pickle
import sys
import argparse
import numpy as np


def print_statistics(episodes):
    """Print detailed statistics across all episodes."""
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)
    
    returns = [ep['return'] for ep in episodes]
    lengths = [ep['length'] for ep in episodes]
    
    print("\nReturns:")
    print(f"  Mean:   {np.mean(returns):.2f}")
    print(f"  Std:    {np.std(returns):.2f}")
    print(f"  Min:    {np.min(returns):.2f}")
    print(f"  Max:    {np.max(returns):.2f}")
    print(f"  Median: {np.median(returns):.2f}")
    
    print("\nLengths:")
    print(f"  Mean:   {np.mean(lengths):.2f}")
    print(f"  Std:    {np.std(lengths):.2f}")
    print(f"  Min:    {int(np.min(lengths))}")
    print(f"  Max:    {int(np.max(lengths))}")
    print(f"  Median: {int(np.median(lengths))}")
    
    # Analyze human messages if present
    message_events = []
    for episode in episodes:
        # Track legacy format per episode
        legacy_prev_messages = []
        
        for trans in episode['transitions']:
            # New format: single message per step
            if 'human_message' in trans['info'] and trans['info']['human_message']:
                message_events.append(trans['info']['human_message'])
            # Legacy format: cumulative list (detect new messages)
            elif 'human_messages' in trans['info']:
                current_messages = trans['info']['human_messages']
                if len(current_messages) > len(legacy_prev_messages):
                    new_messages = current_messages[len(legacy_prev_messages):]
                    message_events.extend(new_messages)
                legacy_prev_messages = current_messages
    
    if message_events:
        print("\nHuman Messages (actual sends):")
        print(f"  Total messages sent: {len(message_events)}")
        print(f"  Unique messages: {len(set(message_events))}")
        
        # Count message frequency
        from collections import Counter
        message_counts = Counter(message_events)
        print(f"\n  Messages sent:")
        for msg, count in message_counts.most_common(10):
            print(f"    '{msg}': {count} times")
    
    # Analyze action distribution
    all_actions = []
    for episode in episodes:
        for trans in episode['transitions']:
            all_actions.extend(trans['actions'])
    
    if all_actions:
        print("\nAction Distribution:")
        from collections import Counter
        action_counts = Counter(all_actions)
        for action, count in sorted(action_counts.items()):
            percentage = (count / len(all_actions)) * 100
            print(f"  Action {action}: {count} times ({percentage:.1f}%)")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="View episode statistics")
    parser.add_argument('filepath', type=str, help="Path to pickle file")
    
    args = parser.parse_args()
    
    # Load data
    try:
        with open(args.filepath, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {args.filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    episodes = data['episodes']
    
    # Display statistics
    print_statistics(episodes)


if __name__ == "__main__":
    main()

