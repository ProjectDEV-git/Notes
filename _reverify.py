"""Re-run the documented acceptance test under today's defaults."""
import time
from notetaker import summarize as S, config
from notetaker.asr import read_transcript

log = open("/tmp/rv.out", "w", buffering=1)
def say(m): print(m, flush=True); log.write(m + "\n")

path = "/home/gamingrf/.local/share/notetaker/sessions/2026-08-28_1258_aristotle-logic-lecture/transcript.jsonl"
segs = read_transcript(path)
transcript = " ".join(s.text for s in segs)
say(f"input: {len(segs)} segments, {len(transcript.split())} words, "
    f"{len(S.build_windows(segs))} windows")

t0 = time.time()
notes = S.summarize_segments(segs, title="Aristotle Logic Lecture", model=config.TEST_MODEL)
took = time.time() - t0
words = len(notes.markdown.split())
say(f"summarizing: {took/60:.0f}m {took%60:.0f}s   (README said 3m 31s)")
say(f"output: {len(transcript.split())} words -> {words} words "
    f"({words/len(transcript.split())*100:.1f}%)   (README said 171 words / 17.3%)")
say("")
say(notes.markdown)
