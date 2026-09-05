"""Does the new guard stop names being miscast as vocabulary?"""
from notetaker import summarize as S, config
from notetaker.asr import read_transcript
log=open("/tmp/g.out","w",buffering=1)
def say(m): print(m,flush=True); log.write(m+"\n")
path="/home/gamingrf/.local/share/notetaker/sessions/2026-08-28_1258_aristotle-logic-lecture/transcript.jsonl"
segs=read_transcript(path)
n=S.summarize_segments(segs,title="Aristotle Logic Lecture",model=config.TEST_MODEL)
say(n.markdown)
