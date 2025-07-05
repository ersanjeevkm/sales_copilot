"""Test configuration and utilities."""

import os
import tempfile
import shutil
from typing import Dict, Any


class TestConfig:
    """Configuration for test environment."""
    
    @staticmethod
    def get_test_data_dir() -> str:
        """Get path to test data directory."""
        return os.path.join(os.path.dirname(__file__), 'test_data')
    
    @staticmethod
    def get_test_db_path() -> str:
        """Get path for test database."""
        return os.path.join(tempfile.gettempdir(), 'test_sales_copilot.db')
    
    @staticmethod
    def get_test_index_path() -> str:
        """Get path for test FAISS index."""
        return os.path.join(tempfile.gettempdir(), 'test_faiss_index')


def create_test_transcript(content: str, filename: str = "test_call.txt") -> str:
    """Create a temporary test transcript file."""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return file_path


def cleanup_test_files(*file_paths: str) -> None:
    """Clean up test files and directories."""
    for path in file_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)


SAMPLE_CALL_TRANSCRIPT = """[00:00] AE (Jordan): Good morning, Priya! Appreciate you carving out a full hour. How's the quarter treating you so far?

[00:05] Prospect (Priya – RevOps Director): Hey Jordan. Busy as always—pipeline is healthy, but I'm drowning in call recordings.

[00:11] AE: Totally hear that. Before we jump in, quick agenda check: I'll recap what we learned on our discovery email thread, Luis will run a live product demo, then we'll map next steps. Sound good?

[00:21] Prospect: Perfect.

[00:23] AE: Great. From our emails, your reps use ZoomInfo for contact discovery, Outreach for sequencing, and Salesforce as the CRM. You record ~500 calls a week but only manually review 5%. Primary pains you flagged were: 1) slow onboarding of new AEs, 2) lack of structured insight for coaching, and 3) no way to surface buying signals automatically. Did I miss anything?

[00:45] Prospect: That's accurate. Also, sales leadership wants a "single pane" where they see risk on every opp without scrubbing hour-long calls.

[00:55] AE: Awesome. That lines up exactly with what our AI Copilot solves. Quick intro—Luis is our Sales Engineer and ex-AE. He'll share screen now.

[01:05] SE (Luis): Thanks Jordan. Priya, can you see my browser?

[01:07] Prospect: Yup, clear.

[01:09] SE: Cool. Let's start with the dashboard. These tiles show total calls ingested, adoption by rep, and "deal health" on the right. Each health score is computed via a regression on verbal cues—next steps, pricing talk, risk phrases like "budget freeze," etc."""

EMPTY_TRANSCRIPT = ""

INVALID_FORMAT_TRANSCRIPT = """This is just regular text without any timestamp format.
Some random content that doesn't follow the expected pattern.
No speakers, no timestamps."""
