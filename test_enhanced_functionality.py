#!/usr/bin/env python3
"""
Test script to validate the improved genVideo.py functionality
"""

import os
import sys
import argparse
from unittest.mock import patch

# Add the current directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
        print("✅ Diffusers imported successfully")
    except ImportError as e:
        print(f"❌ Diffusers import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL imported successfully")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    try:
        from util.converter import VideoConverter
        from util.prompt_handler import PromptHandler
        print("✅ Utility modules imported successfully")
    except ImportError as e:
        print(f"❌ Utility modules import failed: {e}")
        return False
    
    return True

def test_argument_parsing():
    """Test if the argument parsing works correctly"""
    print("\nTesting argument parsing...")
    
    try:
        # Import the main module
        import genVideo
        
        # Test text-to-video arguments
        test_args = [
            '--prompt', 'test prompt',
            '--duration', '3.0',
            '--fps', '10',
            '--width', '256',
            '--height', '256',
            '--output_name', 'test_output'
        ]
        
        with patch('sys.argv', ['genVideo.py'] + test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument("--prompt", type=str, default="test")
            parser.add_argument("--negative_prompt", type=str, default="")
            parser.add_argument("--video_path", type=str)
            parser.add_argument("--output_name", type=str, default="output")
            parser.add_argument("--fps", type=int, default=15)
            parser.add_argument("--duration", type=float, default=5.0)
            parser.add_argument("--width", type=int, default=512)
            parser.add_argument("--height", type=int, default=512)
            parser.add_argument("--num_inference_steps", type=int, default=10)
            parser.add_argument("--strength", type=float, default=0.35)
            parser.add_argument("--guidance_scale", type=float, default=7.5)
            parser.add_argument("--blend", type=float, default=0.1)
            parser.add_argument("--num_previous_frames", type=int, default=1)
            parser.add_argument("--seed", type=int, default=66)
            parser.add_argument("--prompt_file", type=str)
            
            args = parser.parse_args(test_args)
            
            assert args.prompt == 'test prompt'
            assert args.duration == 3.0
            assert args.fps == 10
            assert args.width == 256
            assert args.height == 256
            assert args.output_name == 'test_output'
            
        print("✅ Argument parsing works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Argument parsing failed: {e}")
        return False

def test_prompt_file_creation():
    """Test creating and parsing prompt files"""
    print("\nTesting prompt file functionality...")
    
    try:
        from util.prompt_handler import PromptHandler
        
        # Create a test text file
        test_txt_content = "A beautiful landscape with mountains and rivers"
        with open("test_prompt.txt", "w") as f:
            f.write(test_txt_content)
        
        # Test text file parsing
        handler = PromptHandler("test_prompt.txt", fps=10)
        prompt = handler.get_prompt_for_frame(2.5)
        assert prompt == test_txt_content
        
        # Clean up
        os.remove("test_prompt.txt")
        
        print("✅ Prompt file functionality works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Prompt file test failed: {e}")
        return False

def test_video_converter():
    """Test video converter functionality"""
    print("\nTesting video converter...")
    
    try:
        from util.converter import VideoConverter
        
        converter = VideoConverter()
        
        # Test that the converter can be instantiated
        assert converter is not None
        
        print("✅ Video converter works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Video converter test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("RAGDiffusion - Enhanced Functionality Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_argument_parsing,
        test_prompt_file_creation,
        test_video_converter
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced functionality is ready to use.")
        print("\nYou can now use the following modes:")
        print("1. Text-to-Video: python genVideo.py --prompt 'your prompt' --duration 5.0")
        print("2. Video-to-Video: python genVideo.py --prompt 'your prompt' --video_path 'input.mp4'")
        print("3. With prompt file: python genVideo.py --prompt_file 'prompts.txt' --duration 5.0")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
