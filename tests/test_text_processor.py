"""Tests for the text processor module."""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from text_processor import TextProcessor


class TestTextProcessor:
    """Test cases for TextProcessor class."""
    
    def setup_method(self):
        """Set up text processor."""
        self.processor = TextProcessor(chunk_size=256)
    
    def test_parse_transcript(self):
        """Test parsing transcript content."""
        content = """[00:00] AE (Jordan): Good morning, Priya!
[00:05] Prospect (Priya): Hey Jordan. Busy as always.
[00:11] AE: Before we jump in, quick agenda check."""
        
        segments = self.processor.parse_transcript(content)
        
        assert len(segments) == 3
        assert segments[0]['timestamp'] == '00:00'
        assert segments[0]['speaker'] == 'AE (Jordan)'
        assert segments[0]['content'] == 'Good morning, Priya!'
        
        assert segments[1]['timestamp'] == '00:05'
        assert segments[1]['speaker'] == 'Prospect (Priya)'
        assert segments[1]['content'] == 'Hey Jordan. Busy as always.'
    
    def test_extract_participants(self):
        """Test extracting participant names."""
        content = """[00:00] AE (Jordan): Hello
[00:05] Prospect (Priya): Hi there
[00:10] SE (Luis): Thanks
[00:15] AE (Jordan): Let's continue"""
        
        participants = self.processor.extract_participants(content)
        
        assert len(participants) == 3
        assert "Jordan (AE)" in participants
        assert "Priya (Prospect)" in participants
        assert "Luis (SE)" in participants
    
    def test_create_chunks(self):
        """Test creating text chunks from transcript."""
        content = """[00:00] Speaker A: This is the first message.
[00:05] Speaker B: This is the second message.
[00:10] Speaker A: This is the third message.
[00:15] Speaker B: This is the fourth message."""
        
        chunks = self.processor.create_chunks("test-call", content)
        
        assert len(chunks) > 0
        assert all(chunk.call_id == "test-call" for chunk in chunks)
        assert all(chunk.chunk_id is not None for chunk in chunks)
        assert chunks[0].chunk_index == 0
        
        # Check that content is preserved
        all_content = '\n'.join(chunk.content for chunk in chunks)
        assert "Speaker A" in all_content
        assert "Speaker B" in all_content
    
    def test_get_primary_speaker(self):
        """Test determining primary speaker in a chunk."""
        chunk_lines = [
            "[00:00] Speaker A: First message",
            "[00:05] Speaker A: Second message", 
            "[00:10] Speaker B: One message"
        ]
        
        primary_speaker = self.processor._get_primary_speaker(chunk_lines)
        assert primary_speaker == "Speaker A"
    
    def test_get_chunk_timestamp(self):
        """Test extracting chunk timestamp."""
        chunk_lines = [
            "[00:05] Speaker A: Message",
            "[00:10] Speaker B: Another message"
        ]
        
        timestamp = self.processor._get_chunk_timestamp(chunk_lines)
        assert timestamp == "00:05"
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        content = "We discussed pricing and a 15% discount. There are security concerns about the integration."
        
        keywords = self.processor.extract_keywords(content)
        
        assert "pricing" in keywords
        assert "discount" in keywords
        assert "security" in keywords
        assert "integration" in keywords
        assert "15%" in keywords
    
    def test_empty_content(self):
        """Test handling empty content."""
        segments = self.processor.parse_transcript("")
        assert len(segments) == 0
        
        participants = self.processor.extract_participants("")
        assert len(participants) == 0
        
        chunks = self.processor.create_chunks("test", "")
        assert len(chunks) == 0
    
    def test_malformed_transcript(self):
        """Test handling malformed transcript content."""
        content = "This is not a properly formatted transcript\nJust plain text"
        
        segments = self.processor.parse_transcript(content)
        assert len(segments) == 0  # Should not match any segments
        
        participants = self.processor.extract_participants(content)
        assert len(participants) == 0
