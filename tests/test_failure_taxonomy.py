import sys
from pathlib import Path
sys.path.insert(0, str((Path(__file__).parent.parent / "scripts" / "analysis").resolve()))


def test_classify_failure_types():
    from failure_taxonomy import classify
    assert classify(score=1.0, output="The answer is 42", target="42") == "correct"
    assert classify(score=0.0, output="I cannot help with that.", target="42") == "refusal"
    assert classify(score=0.0, output="", target="42") == "empty"
    assert classify(score=0.0, output="The answer is 41", target="42") == "wrong_answer"
    long = "word " * 600
    assert classify(score=0.0, output=long, target="42") == "runaway"
