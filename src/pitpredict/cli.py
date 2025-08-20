#!/usr/bin/env python3
"""
Command Line Interface for PitPredict
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pitpredict.models.final_position_predict import train_final_position_model, predict_race_positions
from pitpredict.models.train_dnf import train_dnf_model


def train_models():
    """Training CLI for all models"""
    parser = argparse.ArgumentParser(description='Train PitPredict Models')
    parser.add_argument('--model', choices=['final', 'dnf', 'pit', 'all'], 
                       default='all', help='Which model to train')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    if args.model in ['final', 'all']:
        print("🎯 Training Final Position Model...")
        train_final_position_model()
        
    if args.model in ['dnf', 'all']:
        print("⚡ Training DNF Model...")
        train_dnf_model()
        
    if args.model == 'pit':
        print("🔧 Pit Stop Model currently under maintenance...")


def predict_race():
    """Prediction CLI"""
    parser = argparse.ArgumentParser(description='Predict Race Outcomes')
    parser.add_argument('race_id', type=str, help='Race ID (e.g., 2024_21)')
    parser.add_argument('--model', choices=['final', 'dnf'], 
                       default='final', help='Which model to use')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    if args.model == 'final':
        results = predict_race_positions(args.race_id)
        
        if args.output:
            results.to_csv(args.output, index=False)
            print(f"💾 Results saved to {args.output}")
        
        return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_models()
    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict_race()
    else:
        print("Usage: python cli.py [train|predict] [options]")
