import os
from pysrt import SubRipFile

class PromptHandler:
    def __init__(self, file_path, fps=10):
        """
        Initialize the PromptHandler with the path to the prompt file and FPS.
        :param file_path: Path to the input prompt file (either .txt or .srt).
        :param fps: Frames per second of the video (used for timestamp calculations).
        """
        self.file_path = file_path
        self.fps = fps
        self.prompts = []
        self.is_srt = file_path.endswith(".srt")
        self.previous_prompt = None
        if self.is_srt:
            print("Prompt parsing as .srt file")
            self._parse_srt()
        else:
            print("Prompt parsing as .txt file")
            self._parse_txt()

    def _parse_srt(self):
        """Parse an SRT file and store prompts with their timestamps."""
        srt_data = SubRipFile.open(self.file_path)
        for item in srt_data:
            start_time = item.start.ordinal / 1000  # Convert to seconds
            end_time = item.end.ordinal / 1000  # Convert to seconds
            prompt = item.text.strip()
            self.prompts.append((start_time, end_time, prompt))

    def _parse_txt(self):
        """Parse a plain text file and store prompts."""
        temp_prompt = ""
        with open(self.file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                temp_prompt += line.strip()
    
        prompt = [temp_prompt]
        
        self.prompts.append((0, float("inf"), prompt))  # Default to a single prompt covering the entire video
                

    def get_prompt_for_frame(self, frame_time):
        """
        Get the prompt corresponding to the frame's timestamp.
        :param frame_time: The timestamp of the frame in seconds.
        :return: The prompt for the frame, or None if no prompt matches.
        """
        if self.is_srt:
            for start_time, end_time, prompt in self.prompts:
                if start_time <= frame_time <= end_time:
                    self.previous_prompt = prompt
                    return prompt
            return self.previous_prompt
        else:
            return self.prompts[0][2][0] # Return the single prompt for .txt files