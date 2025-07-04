"""Text processing and chunking utilities."""

import re
import uuid
from typing import List, Tuple, Dict
from .storage import TextChunk


class TextProcessor:
    """Handles text processing, parsing, and chunking of call transcripts."""
    
    def __init__(self, chunk_size: int = 256):
        self.chunk_size = chunk_size
        # Regex patterns for parsing call transcripts
        self.timestamp_pattern = r'\[(\d{2}:\d{2})\]'
        self.speaker_pattern = r'\[(\d{2}:\d{2})\]\s*([^:]+):\s*(.+)'
    
    def parse_transcript(self, content: str) -> List[Dict]:
        """
        Parse transcript content into structured segments.
        Handles multi-line speaker segments including bullet points and formatted lists.
        Filters out system/action lines like *screen share* or *Call ends*.
        
        Returns:
            List of dicts with 'timestamp', 'speaker', 'content'
        """
        segments = []
        lines = content.strip().split('\n')
        current_segment = None
        
        # Pattern to match system/action lines like [HH:MM] *action*
        action_pattern = r'\[\d{2}:\d{2}\]\s*\*.*\*\s*$'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip system/action lines like [05:12] *screen share: ROI.xlsx*
            if re.match(action_pattern, line):
                continue
                
            # Try to match speaker pattern [HH:MM] Speaker: Content
            match = re.match(self.speaker_pattern, line)
            if match:
                # If we have a previous segment, save it
                if current_segment:
                    segments.append(current_segment)
                
                timestamp, speaker, text = match.groups()
                
                # Clean up speaker name (remove parenthetical info)
                speaker_clean = re.sub(r'\s*\([^)]*\)', '', speaker).strip()
                
                # Start new segment
                current_segment = {
                    'timestamp': timestamp,
                    'speaker': speaker_clean,
                    'content': text.strip()
                }
            else:
                # This is a continuation line (bullet point, numbered list, etc.)
                if current_segment:
                    # Add the line to the current segment's content
                    current_segment['content'] += '\n' + line
        
        # Don't forget the last segment
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def extract_participants(self, content: str) -> List[str]:
        """Extract unique participant names from transcript."""
        segments = self.parse_transcript(content)
        participants = set()
        
        for segment in segments:
            speaker = segment['speaker']
            # Handle cases like "AE (Jordan)" -> "Jordan (AE)"
            if '(' in speaker:
                parts = speaker.split('(')
                if len(parts) >= 2:
                    role = parts[0].strip()
                    name = parts[1].replace(')', '').strip()
                    participants.add(f"{name} ({role})")
            else:
                participants.add(speaker)
        
        return sorted(list(participants))
    
    def _create_chunk(self, current_chunk: List[Dict], call_id: str, chunk_index: int) -> TextChunk:
        """Create a TextChunk from a list of chunk segments."""
        chunk_content = '\n'.join([seg['text'] for seg in current_chunk])
        return TextChunk(
            chunk_id=str(uuid.uuid4()),
            call_id=call_id,
            content=chunk_content,
            speaker=self._get_primary_speaker(current_chunk),
            timestamp=self._get_chunk_timestamp(current_chunk),
            chunk_index=chunk_index
        )

    def create_chunks(self, call_id: str, content: str) -> Tuple[List[TextChunk], List[str]]:
        """
        Split transcript into chunks suitable for embedding and extract participants.
        
        Strategy:
        1. Parse into speaker segments
        2. Group segments by chunk_size tokens (approximately)
        3. Preserve speaker context and timestamps
        4. Extract unique participants during processing
        
        Returns:
            Tuple of (chunks, participants)
        """
        segments = self.parse_transcript(content)
        chunks = []
        participants = set()
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for segment in segments:
            # Add speaker to participants set
            participants.add(segment['speaker'])
            
            segment_text = f"[{segment['timestamp']}] {segment['speaker']}: {segment['content']}"
            # Rough token estimation (1 token ≈ 4 characters)
            segment_tokens = len(segment_text) // 4
            
            # Create segment dict with parsed info
            segment_info = {
                'text': segment_text,
                'timestamp': segment['timestamp'],
                'speaker': segment['speaker'],
                'tokens': segment_tokens
            }
            
            # If adding this segment would exceed chunk size, finalize current chunk
            if current_tokens + segment_tokens > self.chunk_size and current_chunk:
                chunk = self._create_chunk(current_chunk, call_id, chunk_index)
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk = []
                current_tokens = 0
                chunk_index += 1
            
            current_chunk.append(segment_info)
            current_tokens += segment_tokens
        
        # Handle remaining chunk
        if current_chunk:
            chunk = self._create_chunk(current_chunk, call_id, chunk_index)
            chunks.append(chunk)
        
        return chunks, sorted(list(participants))
    