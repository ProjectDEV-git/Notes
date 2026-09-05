"""The acceptance recording, with the narration filter and prompt guard."""
import time
from notetaker import summarize as S, config
from notetaker.asr import read_transcript
log=open("/tmp/f.out","w",buffering=1)
def say(m): print(m,flush=True); log.write(m+"\n")
p="/home/gamingrf/.local/share/notetaker/sessions/2026-08-28_1258_aristotle-logic-lecture/transcript.jsonl"
segs=read_transcript(p)
words=len(" ".join(s.text for s in segs).split())
t0=time.time()
n=S.summarize_segments(segs,title="Aristotle Logic Lecture",model=config.TEST_MODEL)
took=time.time()-t0
out=len(n.markdown.split())
say(f"summarizing: {took/60:.0f}m {took%60:.0f}s")
say(f"output: {words} -> {out} words ({out/words*100:.1f}%)")
say("")
say(n.markdown)
